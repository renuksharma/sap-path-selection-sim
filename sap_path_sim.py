"""
sap_path_sim.py -- Reproducible simulation of path-selection policies for a
dual-connection RISE with SAP design (private primary + IPsec backup).

Everything reported in the accompanying article is produced by this file.
Run:  python sap_path_sim.py --out results
Requirements: Python 3.10+, numpy, scikit-learn.

Model summary (all parameters are in CFG below):
  * 60-minute scenarios, 5-second ticks, 30-second decision windows.
  * Two customer-edge paths to the SAP private-cloud ingress:
      primary = private circuit (1,000 Mbps), backup = IPsec tunnel (CFG capacity).
  * Tier 1 SAP transactions (Fiori/GUI dialog steps, postings, synchronous
    RFC/OData calls) are sampled every tick on BOTH paths with common random
    numbers, so every policy faces identical conditions.
  * Response = SAP server time + round_trips x RTT + TCP retransmission
    penalties + jitter noise. A transaction FAILS if its path is down or the
    response exceeds the client timeout (4,000 ms).
  * SLO breach = Tier 1 P95 response in a 30 s window > 1,500 ms.
  * Policies share the same eligibility gates, BFD-style hard-failure
    detection, failback rule and workload admission. Only the trigger differs.
  * The risk models are logistic regressions trained on a TRAIN set; policy
    thresholds are tuned on a separate VALIDATION set; all reported results
    come from a held-out TEST set with different seeds.
"""
from __future__ import annotations
import argparse, json, csv, os, math
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CFG = dict(
    tick_s=5, window_ticks=6, scenario_s=3600,
    tx_per_tick=20,                 # Tier 1 transactions sampled per tick (4/s)
    round_trips=12,                 # sequential network round trips per Tier 1 step (Fiori OData/GUI)
    rto_ms=300.0,                   # initial TCP RTO; doubles on each consecutive loss
    timeout_ms=3000.0,              # client/interface timeout => failed transaction
    slo_p95_ms=1800.0,              # Tier 1 SLO, 30 s window P95
    server_median_ms=350.0, server_sigma=0.5,
    primary=dict(base_ms=48.0, jitter_ms=2.0, loss_pct=0.02, cap_mbps=1000.0),
    backup=dict(base_ms=61.0, jitter_ms=4.0, loss_pct=0.05, cap_mbps=600.0),
    demand_mbps=(450.0, 800.0),     # uniform per-scenario peak demand
    tier_share=dict(t1=0.40, t2=0.33, t3=0.27),
    switch_reset_frac=0.15,         # share of in-flight Tier 1 tx reset by a path change
    failback_windows=20,            # 10 minutes of stable primary before failback
    horizon_windows=10,             # 5-minute prediction horizon
    persistence=2,                  # consecutive windows required (60 s)
    thr=dict(lat_ms=95.0, loss_pct=1.0, jit_ms=18.0),   # SD-WAN SLA-class style thresholds
    norm=dict(lat=(40.0, 140.0), loss=(0.0, 3.0), jit=(0.0, 30.0)),
    fault_mix={                     # scenario type -> probability
        "progressive_latency": 0.16, "loss_burst": 0.14, "jitter_instability": 0.12,
        "combined": 0.12, "hard_failure": 0.10, "benign_transient": 0.10,
        "mtu_blackhole": 0.06,
        "app_side_slowdown": 0.08, "shared_risk": 0.04,
        "backup_ineligible": 0.04, "stale_telemetry": 0.04,
    },
    large_payload_share=0.35,       # share of Tier 1 tx with payloads large enough to hit a PMTU black hole
    false_change_cost=20.0,         # tuning objective: 1 false path change ~ 20 failed Tier 1 tx
)

T = CFG["scenario_s"] // CFG["tick_s"]          # 720 ticks
W = T // CFG["window_ticks"]                      # 120 windows
N = CFG["tx_per_tick"]
POLICIES = ["operator", "bfd_only", "threshold", "weighted", "pred_network", "sap_aware"]
LABEL = {"operator": "Operator-guided", "bfd_only": "BFD-only failover",
         "threshold": "Threshold + hysteresis", "weighted": "Weighted score + persistence",
         "pred_network": "Predictive (network-only)", "sap_aware": "SAP-aware predictive"}


# ----------------------------------------------------------------------------
# Scenario generation
# ----------------------------------------------------------------------------
@dataclass
class Path:
    lat_add: np.ndarray; loss: np.ndarray; jit: np.ndarray; down: np.ndarray
    pmtu: np.ndarray = None   # fraction of large-payload flows black-holed (MTU/PMTUD failure)


def _ramp(t, t0, rise, hold, fall, amp):
    y = np.zeros_like(t, dtype=float)
    a = (t >= t0) & (t < t0 + rise); y[a] = amp * (t[a] - t0) / rise
    b = (t >= t0 + rise) & (t < t0 + rise + hold); y[b] = amp
    c = (t >= t0 + rise + hold) & (t < t0 + rise + hold + fall)
    y[c] = amp * (1 - (t[c] - t0 - rise - hold) / fall)
    return y


def gen_scenario(seed: int, sid: int, ftype: str | None = None, backup_cap=None):
    rng = np.random.default_rng([seed, sid])
    cfg = CFG
    if ftype is None:
        types, p = zip(*cfg["fault_mix"].items())
        ftype = types[rng.choice(len(types), p=np.array(p) / sum(p))]
    t = np.arange(T) * cfg["tick_s"]
    P = Path(np.zeros(T), np.full(T, cfg["primary"]["loss_pct"]), np.full(T, cfg["primary"]["jitter_ms"]), np.zeros(T, bool), np.zeros(T))
    B = Path(np.zeros(T), np.full(T, cfg["backup"]["loss_pct"]), np.full(T, cfg["backup"]["jitter_ms"]), np.zeros(T, bool), np.zeros(T))
    app_add = np.zeros(T)
    eligible_b = np.ones(T, bool)
    stale_w = np.zeros(W, bool)
    t0 = rng.uniform(600, 2100)
    meta = dict(type=ftype, onset_s=round(float(t0), 1))

    if ftype in ("progressive_latency", "backup_ineligible", "stale_telemetry"):
        rise, hold = rng.uniform(300, 900), rng.uniform(600, 1200)
        P.lat_add += _ramp(t, t0, rise, hold, 120, rng.uniform(60, 140))
        P.loss += _ramp(t, t0 + rise * 0.6, rise * 0.4, hold, 120, rng.uniform(0.0, 0.8))
        if ftype == "backup_ineligible":           # IPsec SA / certificate / prefix gate fails
            eligible_b[(t >= t0) & (t < t0 + rise + hold)] = False
        if ftype == "stale_telemetry":             # collector outage overlapping the fault
            ws = int((t0 + rise * 0.3) // 30); stale_w[ws: ws + int(rng.uniform(6, 20))] = True
    elif ftype == "loss_burst":
        pre = rng.uniform(60, 180); dur = rng.uniform(180, 600)
        pre_mask = (t >= t0 - pre) & (t < t0)
        P.loss[pre_mask] += rng.uniform(0.3, 0.9) * (rng.random(pre_mask.sum()) < 0.5)
        P.loss[(t >= t0) & (t < t0 + dur)] += rng.uniform(1.5, 6.0)
    elif ftype == "jitter_instability":
        rise, hold = rng.uniform(120, 480), rng.uniform(300, 900)
        env = _ramp(t, t0, rise, hold, 60, 1.0)
        P.jit += env * rng.uniform(20, 45)
        P.lat_add += env * rng.uniform(15, 50) * (0.5 + 0.5 * np.sin(t / 40.0))
        P.loss += env * rng.uniform(0.5, 1.5)
    elif ftype == "combined":
        rise, hold = rng.uniform(300, 900), rng.uniform(600, 1200)
        env = _ramp(t, t0, rise, hold, 120, 1.0)
        P.lat_add += env * rng.uniform(40, 100); P.loss += env * rng.uniform(0.8, 3.0); P.jit += env * rng.uniform(8, 20)
    elif ftype == "hard_failure":
        P.down[(t >= t0) & (t < t0 + rng.uniform(120, 600))] = True
    elif ftype == "benign_transient":
        dur = rng.uniform(30, 75)
        m = (t >= t0) & (t < t0 + dur)
        if rng.random() < 0.5: P.lat_add[m] += rng.uniform(40, 70)
        else: P.loss[m] += rng.uniform(1.0, 2.0)
    elif ftype == "mtu_blackhole":                # e.g. tunnel/MSS change: small probes pass, large payloads stall
        P.pmtu[(t >= t0) & (t < t0 + rng.uniform(600, 1500))] = rng.uniform(0.3, 0.9)
    elif ftype == "app_side_slowdown":            # SAP-side issue: route change cannot help
        app_add += _ramp(t, t0, rng.uniform(120, 600), rng.uniform(600, 1200), 120, rng.uniform(900, 2500))
    elif ftype == "shared_risk":                  # shared last mile / facility
        rise, hold = rng.uniform(200, 600), rng.uniform(600, 1200)
        env = _ramp(t, t0, rise, hold, 120, 1.0)
        L = rng.uniform(1.5, 4.0); A = rng.uniform(40, 90)
        P.loss += env * L; B.loss += env * L * rng.uniform(0.8, 1.2)
        P.lat_add += env * A; B.lat_add += env * A

    demand = rng.uniform(*cfg["demand_mbps"])
    crit = demand * (cfg["tier_share"]["t1"] + cfg["tier_share"]["t2"])
    # Common random numbers for transactions (identical across paths/policies)
    crn = dict(
        server=rng.lognormal(math.log(cfg["server_median_ms"]), cfg["server_sigma"], (T, N)),
        uloss=rng.random((T, N, cfg["round_trips"])),
        jn=rng.standard_normal((T, N)),
        probe_jn=rng.standard_normal((T, 2, 3)),
        sap_probe=rng.lognormal(math.log(250), 0.3, (W, 2, 2)),
        sap_uloss=rng.random((W, 2, 2, 6)),
        reset=rng.random((T, N)),
        ularge=rng.random((T, N)),
        probe_large=rng.random((W, 2, 2)),
        operator_delay_s=rng.uniform(300, 900),
    )
    cap_b = backup_cap if backup_cap is not None else cfg["backup"]["cap_mbps"]
    sc = dict(meta=meta, P=P, B=B, app_add=app_add, eligible_b=eligible_b, stale_w=stale_w,
              demand=demand, crit=crit, crn=crn, cap_b=cap_b)
    precompute(sc)
    return sc


def _queue(load, cap):
    """Piecewise queueing: light M/M/1-style delay below 95% utilization,
    rising to 60 ms at 100%; above 100% the excess is dropped (loss %)."""
    rho = load / cap
    if rho < 0.95:
        return 1.0 * rho / (1 - rho), 0.0
    if rho < 1.0:
        return 19.0 + (rho - 0.95) / 0.05 * 41.0, 0.0
    return 60.0, min(30.0, (rho - 1.0) / rho * 100.0)


def _tx(sc, path: Path, base_ms, q_ms, q_loss):
    c, crn = CFG, sc["crn"]
    rtt = base_ms + path.lat_add + q_ms
    loss = np.clip(path.loss + q_loss, 0, 100) / 100.0
    # consecutive losses per round trip are geometric; each adds a doubled RTO
    p = np.clip(loss, 1e-9, 0.999)[:, None, None]
    nloss = np.floor(np.log(crn["uloss"]) / np.log(p))
    penalty = (c["rto_ms"] * (2.0 ** np.minimum(nloss, 6) - 1)).sum(axis=2)
    resp = crn["server"] + sc["app_add"][:, None] + c["round_trips"] * rtt[:, None] \
        + penalty + np.abs(crn["jn"]) * path.jit[:, None] * 2
    stalled = crn["ularge"] < (path.pmtu * c["large_payload_share"])[:, None]
    resp[stalled] = c["timeout_ms"] + 1.0
    resp[path.down] = np.inf
    return resp


def precompute(sc):
    c = CFG
    P, B = sc["P"], sc["B"]
    qp, lp = _queue(sc["demand"], c["primary"]["cap_mbps"])
    qbf, lbf = _queue(sc["demand"], sc["cap_b"])     # backup carrying all production tiers
    qbs, lbs = _queue(sc["crit"], sc["cap_b"])       # backup with Tier 3 shed (admission)
    sc["resp"] = dict(P=_tx(sc, P, c["primary"]["base_ms"], qp, lp),
                      B_full=_tx(sc, B, c["backup"]["base_ms"], qbf, lbf),
                      B_shed=_tx(sc, B, c["backup"]["base_ms"], qbs, lbs))
    # Network probes (3 per tick) -> window telemetry; probes see idle backup
    tel = {}
    for k, (path, base, q, ql) in {"P": (P, c["primary"]["base_ms"], qp, lp),
                                   "B": (B, c["backup"]["base_ms"], 0.0, 0.0)}.items():
        idx = 0 if k == "P" else 1
        lat = base + path.lat_add[:, None] + q + np.abs(sc["crn"]["probe_jn"][:, idx, :]) * path.jit[:, None]
        lat_w = lat.reshape(W, -1)
        p95 = np.percentile(lat_w, 95, axis=1)
        jit = np.abs(np.diff(lat_w, axis=1)).mean(axis=1)
        loss = (path.loss + ql).reshape(W, -1).mean(axis=1)
        down = path.down.reshape(W, -1).any(axis=1)
        p95[down] = 1000.0; loss[down] = 100.0
        # SAP synthetic probes (Fiori launchpad tx + RFC_PING) through this path
        rtt_w = base + path.lat_add.reshape(W, -1).mean(axis=1) + q
        lw = np.clip(loss / 100.0, 0, 1)
        lost = (sc["crn"]["sap_uloss"][:, idx] < lw[:, None, None]).sum(axis=2)
        fiori = sc["crn"]["sap_probe"][:, idx] + 0.8 * sc["app_add"].reshape(W, -1).mean(axis=1)[:, None] \
            + 6 * rtt_w[:, None] + lost * c["rto_ms"]
        pm = path.pmtu.reshape(W, -1).mean(axis=1)
        fiori = np.where(sc["crn"]["probe_large"][:, idx] < pm[:, None], 10000.0, fiori)   # probe payload > path MTU
        fiori = fiori.mean(axis=1); fiori[down] = 10000.0
        rfc = rtt_w + 2.0 + lost[:, 0] * c["rto_ms"]; rfc[down] = 10000.0
        tel[k] = dict(p95=p95, jit=jit, loss=loss, down=down, fiori=fiori, rfc=rfc)
    for k in tel:
        for m in ("fiori", "rfc"):
            tel[k][m + "_ratio"] = tel[k][m] / np.median(tel[k][m][:10])
    sc["tel"] = tel
    # Ground truth for the primary path (counterfactual: if traffic stayed on primary)
    rp = sc["resp"]["P"].reshape(W, -1)
    p95p = np.percentile(np.where(np.isfinite(rp), rp, 1e6), 95, axis=1)
    sc["truth_breach"] = (p95p > c["slo_p95_ms"]) | tel["P"]["down"]
    sc["truth_p95"] = p95p
    sc["user_ratio_P"] = p95p / np.median(p95p[:10])


# ----------------------------------------------------------------------------
# Features and risk models
# ----------------------------------------------------------------------------
def _norm(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def features(tel_path, user_ratio=None, sap=False):
    n = CFG["norm"]
    L = _norm(tel_path["p95"], *n["lat"]); P = _norm(tel_path["loss"], *n["loss"]); J = _norm(tel_path["jit"], *n["jit"])
    slope = np.zeros(W); slope[3:] = (L[3:] - L[:-3]) / 3
    lmax = np.array([P[max(0, i - 3): i + 1].max() for i in range(W)])
    cols = [L, P, J, slope, lmax]
    if sap:
        fr = np.clip(tel_path["fiori_ratio"] - 1, 0, 5)
        rr = np.clip(tel_path["rfc_ratio"] - 1, 0, 5)
        fs = np.zeros(W); fs[3:] = (fr[3:] - fr[:-3]) / 3
        ur = np.clip((user_ratio if user_ratio is not None else tel_path["fiori_ratio"]) - 1, 0, 5)
        cols += [fr, rr, fs, ur]
    return np.column_stack(cols)


FEAT_NAMES = ["latency_p95_norm", "loss_norm", "jitter_norm", "latency_trend_90s", "loss_max_2min"]
FEAT_NAMES_SAP = FEAT_NAMES + ["fiori_probe_ratio", "rfc_probe_ratio", "fiori_trend_90s", "tier1_user_p95_ratio"]


def labels(sc):
    b = sc["truth_breach"]; h = CFG["horizon_windows"]
    return np.array([b[i: i + h + 1].any() for i in range(W)])


class RiskModel:
    def __init__(self, sap: bool):
        self.sap = sap
        self.m = LogisticRegression(C=1.0, max_iter=2000)  # ~30% positive windows; threshold tuned on validation

    def fit(self, scs):
        X = np.vstack([features(s["tel"]["P"], s["user_ratio_P"], self.sap) for s in scs])
        y = np.concatenate([labels(s) for s in scs])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.m.fit((X - self.mu) / self.sd, y)
        return self

    def predict(self, X):
        return self.m.predict_proba((X - self.mu) / self.sd)[:, 1]


# ----------------------------------------------------------------------------
# Policy execution
# ----------------------------------------------------------------------------
def healthy_net(tel, w):
    th = CFG["thr"]
    return (not tel["down"][w]) and tel["p95"][w] <= th["lat_ms"] and tel["loss"][w] <= th["loss_pct"] and tel["jit"][w] <= th["jit_ms"]


def weighted_score(tel, w, weights, cost):
    n = CFG["norm"]
    return (weights[0] * _norm(tel["p95"][w], *n["lat"]) + weights[1] * _norm(tel["loss"][w], *n["loss"])
            + weights[2] * _norm(tel["jit"][w], *n["jit"]) + weights[3] * cost)


def run_policy(sc, policy, prm, models, admission=True, gates=True, log=False, attribution=True):
    c = CFG; tel = sc["tel"]; crn = sc["crn"]
    respB = sc["resp"]["B_shed" if admission else "B_full"]; respP = sc["resp"]["P"]
    if policy in ("pred_network", "sap_aware"):
        sap = policy == "sap_aware"
        rP = models[policy].predict(features(tel["P"], sc["user_ratio_P"], sap))
        # The model is trained on the primary path; the backup is validated with the
        # same SLA-class health check used by every other policy (no distribution shift).
    active = "P"; fail = 0; resets = 0; switches = []; failbacks = 0; streak = 0
    healthy_streak = 0; unsafe = 0; blocked_gate = 0; held_stale = 0; held_app = 0
    op_due = None; impact_windows = 0; warnings = np.zeros(W, bool); events = []
    win_resp = []
    for k in range(T):
        w_done = k // c["window_ticks"] - 1
        # ---- decisions at window boundaries -----------------------------------
        if k % c["window_ticks"] == 0 and k > 0:
            w = w_done
            # impact accounting for the finished window
            wr = np.concatenate(win_resp); win_resp = []
            if np.percentile(np.where(np.isfinite(wr), wr, 1e6), 95) > c["slo_p95_ms"]:
                impact_windows += 1
            elig = bool(sc["eligible_b"][k - 1]) or not gates
            stale = bool(sc["stale_w"][w]) and gates
            want = False; reason = ""
            if active == "P":
                if policy == "operator":
                    viol = not healthy_net(tel["P"], w) and (w == 0 or not healthy_net(tel["P"], w - 1))
                    if viol and op_due is None:
                        op_due = k * c["tick_s"] + crn["operator_delay_s"]
                    if op_due is not None and k * c["tick_s"] >= op_due:
                        want = not healthy_net(tel["P"], w); op_due = None if not want else op_due
                        reason = "operator_action"
                elif policy == "threshold":
                    cond = not healthy_net(tel["P"], w) and healthy_net(tel["B"], w)
                    streak = streak + 1 if cond else 0
                    want = streak >= c["persistence"]; reason = "sla_threshold"
                elif policy == "weighted":
                    sp = weighted_score(tel["P"], w, prm["weights"], 0.0)
                    sb = weighted_score(tel["B"], w, prm["weights"], prm["backup_cost"])
                    cond = (sp - sb) > prm["margin"]
                    streak = streak + 1 if cond else 0
                    want = streak >= c["persistence"]; reason = "score_margin"
                elif policy in ("pred_network", "sap_aware"):
                    warn = rP[w] > prm["theta"]; warnings[w] = warn
                    cond = warn and healthy_net(tel["B"], w)
                    if policy == "sap_aware" and cond:
                        # attribution: SAP probes degraded on BOTH paths while primary network is normal
                        fp, fb = tel["P"]["fiori_ratio"][w], tel["B"]["fiori_ratio"][w]
                        if attribution and healthy_net(tel["P"], w) and fp > 1.5 and fb >= 0.8 * fp:
                            cond = False; held_app += 1
                        elif attribution and tel["B"]["fiori_ratio"][w] > 1.5:
                            cond = False
                    streak = streak + 1 if cond else 0
                    want = streak >= c["persistence"]; reason = "predicted_slo_threat"
                if want and stale:
                    want = False; held_stale += 1; streak = 0
                if want and not elig:
                    want = False; blocked_gate += 1; streak = 0
                if want:
                    if not sc["eligible_b"][k - 1]: unsafe += 1
                    active = "B"; switches.append(k); streak = 0; healthy_streak = 0
                    resets += int((crn["reset"][k] < c["switch_reset_frac"]).sum())
                    if log: events.append(dict(t_s=k * c["tick_s"], action="failover", reason=reason, window=w))
            else:  # on backup: failback / forced return
                ok = healthy_net(tel["P"], w) and (policy != "sap_aware" or tel["P"]["fiori_ratio"][w] < 1.3)
                healthy_streak = healthy_streak + 1 if ok else 0
                must = (gates and not sc["eligible_b"][k - 1]) or tel["B"]["down"][w]
                if (healthy_streak >= c["failback_windows"] or must) and not tel["P"]["down"][w]:
                    active = "P"; failbacks += 1; healthy_streak = 0
                    resets += int((crn["reset"][k] < c["switch_reset_frac"]).sum())
                    if log: events.append(dict(t_s=k * c["tick_s"], action="failback", reason="forced" if must else "stable_primary", window=w))
        # ---- BFD-style hard failure detection (all policies), 1 tick ----------
        if k > 0 and active == "P" and sc["P"].down[k - 1] and not sc["B"].down[k - 1] and (sc["eligible_b"][k - 1] or not gates):
            active = "B"; switches.append(k); healthy_streak = 0
            if log: events.append(dict(t_s=k * c["tick_s"], action="failover", reason="bfd_down", window=k // 6))
        r = respP[k] if active == "P" else respB[k]
        fail += int((~np.isfinite(r) | (r > c["timeout_ms"])).sum())
        win_resp.append(r)
    fail += resets
    return dict(fail=fail, resets=resets, switches=switches, failbacks=failbacks, impact_min=impact_windows * 0.5,
                unsafe=unsafe, blocked_gate=blocked_gate, held_stale=held_stale, held_app=held_app,
                warnings=warnings, events=events)


def classify_switches(sc, res):
    """A failover is FALSE if the primary had no SLO breach (and was not down)
    from the switch window through the next 10 windows."""
    b = sc["truth_breach"]; false = 0; lead = []
    first_breach = np.argmax(b) if b.any() else None
    for k in res["switches"]:
        w = k // CFG["window_ticks"]
        if not b[max(0, w - 1): w + CFG["horizon_windows"] + 1].any():
            false += 1
    if first_breach is not None and b.sum() >= 2 and res["switches"]:
        k0 = res["switches"][0]
        lead.append(first_breach * 30 - k0 * CFG["tick_s"])   # +ve: switched before user impact
    return false, lead


# ----------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------
SEEDS = dict(train=1001, valid=2002, test=3003)
SIZES = dict(train=400, valid=200, test=600)
BATCH = 30


def build(split, n=None, **kw):
    return [gen_scenario(SEEDS[split], i, **kw) for i in range(n or SIZES[split])]


def evaluate(scs, policy, prm, models, **kw):
    rows = []
    for i, sc in enumerate(scs):
        r = run_policy(sc, policy, prm, models, **kw)
        fc, lead = classify_switches(sc, r)
        degr = sc["truth_breach"].sum() >= 2
        helpful = degr and sc["meta"]["type"] not in ("app_side_slowdown", "shared_risk")
        rows.append(dict(sid=i, type=sc["meta"]["type"], policy=policy, fail=r["fail"], resets=r["resets"],
                         impact_min=r["impact_min"], failovers=len(r["switches"]), false_changes=fc,
                         detected=int(helpful and len(r["switches"]) > fc), helpful=int(helpful),
                         lead_s=lead[0] if lead else "", unsafe=r["unsafe"], blocked_gate=r["blocked_gate"],
                         held_stale=r["held_stale"], held_app=r["held_app"]))
    return rows


def objective(rows):
    return np.mean([r["fail"] + CFG["false_change_cost"] * r["false_changes"] for r in rows])


DEFAULT_PRM = dict(operator={}, bfd_only={}, threshold={},
                   weighted=dict(weights=(0.40, 0.35, 0.15, 0.10), backup_cost=0.30, margin=0.10),
                   pred_network=dict(theta=0.5), sap_aware=dict(theta=0.5))


def tune(valid, models, log):
    prm = {k: dict(v) for k, v in DEFAULT_PRM.items()}
    grid = dict(weighted=("margin", [0.03, 0.06, 0.10, 0.15, 0.20, 0.30]),
                pred_network=("theta", [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]),
                sap_aware=("theta", [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]))
    for pol, (key, vals) in grid.items():
        best = None
        for v in vals:
            p = dict(prm[pol]); p[key] = v
            obj = objective(evaluate(valid, pol, p, models))
            log.append(dict(policy=pol, param=key, value=v, objective=round(obj, 2)))
            if best is None or obj < best[0]: best = (obj, v)
        prm[pol][key] = best[1]
    return prm


def ci95(x):
    x = np.asarray(x, float); m = x.mean()
    if len(x) < 2: return m, m, m
    h = 2.093 * x.std(ddof=1) / math.sqrt(len(x))     # t(0.975, df=19)
    return m, m - h, m + h


def main(out):
    os.makedirs(out, exist_ok=True)
    print("building scenarios ...")
    train, valid, test = build("train"), build("valid"), build("test")
    models = {"pred_network": RiskModel(False).fit(train), "sap_aware": RiskModel(True).fit(train)}
    coef = {k: dict(zip(FEAT_NAMES_SAP if k == "sap_aware" else FEAT_NAMES, np.round(m.m.coef_[0], 3).tolist()),
                    intercept=round(float(m.m.intercept_[0]), 3)) for k, m in models.items()}
    tune_log = []; prm = tune(valid, models, tune_log)
    print("tuned:", {k: v for k, v in prm.items() if v})

    all_rows = []
    for pol in POLICIES:
        all_rows += evaluate(test, pol, prm[pol], models)
    ablation = evaluate(test, "sap_aware", prm["sap_aware"], models, admission=False)
    for r in ablation: r["policy"] = "sap_aware_no_admission"
    gate_off = evaluate(test, "sap_aware", prm["sap_aware"], models, gates=False)
    for r in gate_off: r["policy"] = "sap_aware_no_gates"
    no_attr = evaluate(test, "sap_aware", prm["sap_aware"], models, attribution=False)
    for r in no_attr: r["policy"] = "sap_aware_no_attribution"
    rows = all_rows + ablation + gate_off + no_attr
    with open(f"{out}/test_rows.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    # Summary with 95% CI across 20 batches of 30 scenarios
    summary = {}
    for pol in POLICIES + ["sap_aware_no_admission", "sap_aware_no_gates", "sap_aware_no_attribution"]:
        R = [r for r in rows if r["policy"] == pol]
        batches = [R[i:i + BATCH] for i in range(0, len(R), BATCH)]
        def per_batch(key): return [sum(r[key] for r in b) for b in batches]
        leads = [r["lead_s"] for r in R if r["lead_s"] != ""]
        s = {}
        for key in ("fail", "false_changes", "failovers", "impact_min", "detected", "unsafe", "blocked_gate", "held_stale", "held_app", "resets"):
            m, lo, hi = ci95(per_batch(key)); s[key] = dict(mean=round(m, 2), lo=round(lo, 2), hi=round(hi, 2),
                                                             min=int(min(per_batch(key))), max=int(max(per_batch(key))))
        s["helpful_total"] = sum(r["helpful"] for r in R); s["detected_total"] = sum(r["detected"] for r in R)
        s["median_lead_s"] = float(np.median(leads)) if leads else None
        s["preemptive_share"] = round(float(np.mean([l > 0 for l in leads])), 3) if leads else None
        s["by_type"] = {}
        for ty in CFG["fault_mix"]:
            Rt = [r for r in R if r["type"] == ty]
            s["by_type"][ty] = dict(n=len(Rt), fail_per_scn=round(np.mean([r["fail"] for r in Rt]), 2),
                                    false_changes=sum(r["false_changes"] for r in Rt),
                                    impact_min_per_scn=round(np.mean([r["impact_min"] for r in Rt]), 2))
        summary[pol] = s

    # Window-level confusion matrix for SAP-aware warnings on the test set
    m = models["sap_aware"]; th = prm["sap_aware"]["theta"]
    tp = fp = fn = tn = 0
    for sc in test:
        pred = m.predict(features(sc["tel"]["P"], sc["user_ratio_P"], True)) > th; y = labels(sc)
        tp += int((pred & y).sum()); fp += int((pred & ~y).sum()); fn += int((~pred & y).sum()); tn += int((~pred & ~y).sum())
    mn = models["pred_network"]; thn = prm["pred_network"]["theta"]; cm_n = [0, 0, 0, 0]
    for sc in test:
        pred = mn.predict(features(sc["tel"]["P"])) > thn; y = labels(sc)
        cm_n[0] += int((pred & y).sum()); cm_n[1] += int((pred & ~y).sum()); cm_n[2] += int((~pred & y).sum()); cm_n[3] += int((~pred & ~y).sum())
    conf = dict(sap_aware=dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(tp / max(1, tp + fp), 3), recall=round(tp / max(1, tp + fn), 3)),
                pred_network=dict(tp=cm_n[0], fp=cm_n[1], fn=cm_n[2], tn=cm_n[3],
                                  precision=round(cm_n[0] / max(1, cm_n[0] + cm_n[1]), 3), recall=round(cm_n[0] / max(1, cm_n[0] + cm_n[2]), 3)))

    # Threshold sensitivity (test set) for SAP-aware
    sens = []
    for th_ in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]:
        R = evaluate(test, "sap_aware", dict(theta=th_), models)
        sens.append(dict(theta=th_, fail_per_scn=round(np.mean([r["fail"] for r in R]), 2),
                         false_changes=sum(r["false_changes"] for r in R), failovers=sum(r["failovers"] for r in R)))
    # Weight perturbation for weighted policy (+/-25%, 30 draws, first 300 test scenarios)
    rng = np.random.default_rng(7); wsens = []
    base = np.array(prm["weighted"]["weights"])
    for d in range(30):
        wts = base * rng.uniform(0.75, 1.25, 4); wts = wts / wts.sum()
        p = dict(prm["weighted"]); p["weights"] = tuple(wts)
        R = evaluate(test[:300], "weighted", p, models)
        wsens.append(dict(draw=d, fail_per_scn=round(np.mean([r["fail"] for r in R]), 2), false_changes=sum(r["false_changes"] for r in R)))
    # Backup capacity sweep (first 300 test scenarios), with and without admission
    cap = []
    for cb in [300, 400, 500, 600, 700, 800, 1000]:
        scs = [gen_scenario(SEEDS["test"], i, backup_cap=cb) for i in range(300)]
        for adm in (True, False):
            R = evaluate(scs, "sap_aware", prm["sap_aware"], models, admission=adm)
            onB = [r for r in R if r["failovers"] > 0]
            cap.append(dict(backup_mbps=cb, admission=adm, fail_per_scn=round(np.mean([r["fail"] for r in R]), 2),
                            impact_min_per_scn=round(np.mean([r["impact_min"] for r in R]), 3)))

    json.dump(dict(config=CFG, seeds=SEEDS, sizes=SIZES, tuned=prm, coefficients=coef, tuning=tune_log,
                   summary=summary, confusion=conf, theta_sensitivity=sens, weight_sensitivity=wsens,
                   capacity_sweep=cap), open(f"{out}/summary.json", "w"), indent=1, default=str)
    print("done ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results")
    main(ap.parse_args().out)
