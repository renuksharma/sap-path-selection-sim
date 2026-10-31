"""analyze.py -- turns results/test_rows.csv + results/summary.json into the
tables and figures used in the article. Run after sap_path_sim.py:
    python analyze.py --res results --fig figures
"""
import argparse, csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sap_path_sim as S

# Okabe-Ito colorblind-safe palette
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", grey="#8C8C8C")
PCOL = dict(operator=OI["grey"], bfd_only=OI["black"], threshold=OI["orange"], weighted=OI["purple"],
            pred_network=OI["sky"], sap_aware=OI["blue"])
GROUPS = dict(
    recoverable=["progressive_latency", "loss_burst", "jitter_instability", "combined", "stale_telemetry"],
    hard=["hard_failure"], benign=["benign_transient"], mtu=["mtu_blackhole"],
    not_routable=["app_side_slowdown", "shared_risk", "backup_ineligible"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def boot(x, reps=2000, seed=11):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = rng.choice(x, (reps, len(x))).mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load_rows(res):
    rows = list(csv.DictReader(open(f"{res}/test_rows.csv")))
    for r in rows:
        for k in ("fail", "resets", "failovers", "false_changes", "detected", "helpful", "unsafe", "blocked_gate", "held_stale", "held_app"):
            r[k] = int(r[k])
        r["impact_min"] = float(r["impact_min"]); r["lead_s"] = float(r["lead_s"]) if r["lead_s"] != "" else None
    return rows


def tables(rows):
    out = {}
    pols = S.POLICIES + ["sap_aware_no_admission", "sap_aware_no_gates", "sap_aware_no_attribution"]
    for p in pols:
        R = [r for r in rows if r["policy"] == p]
        g = lambda grp: [r for r in R if r["type"] in GROUPS[grp]]
        rec = g("recoverable")
        leads = [r["lead_s"] for r in rec if r["lead_s"] is not None]
        o = dict(
            n_recoverable=len(rec),
            fail_rec=boot([r["fail"] for r in rec]),
            impact_rec=boot([r["impact_min"] for r in rec]),
            fail_hard=boot([r["fail"] for r in g("hard")]),
            fail_mtu=boot([r["fail"] for r in g("mtu")]), impact_mtu=boot([r["impact_min"] for r in g("mtu")]),
            n_mtu=len(g("mtu")), mtu_switch_scn=sum(1 for r in g("mtu") if r["failovers"] > 0),
            false_changes_all=sum(r["false_changes"] for r in R),
            false_benign=sum(r["false_changes"] for r in g("benign")),
            n_benign=len(g("benign")),
            failovers_all=sum(r["failovers"] for r in R),
            resets_all=sum(r["resets"] for r in R),
            detected=sum(r["detected"] for r in rec), helpful=sum(r["helpful"] for r in rec),
            median_lead_s=float(np.median(leads)) if leads else None,
            p25_lead_s=float(np.percentile(leads, 25)) if leads else None,
            p75_lead_s=float(np.percentile(leads, 75)) if leads else None,
            preemptive_share=float(np.mean([l > 0 for l in leads])) if leads else None,
            unsafe_moves=sum(r["unsafe"] for r in R),
            gate_blocks_scn=sum(1 for r in R if r["blocked_gate"] > 0),
            stale_hold_scn=sum(1 for r in R if r["held_stale"] > 0),
            app_switch_scn=sum(1 for r in R if r["type"] == "app_side_slowdown" and r["failovers"] > 0),
            n_app=sum(1 for r in R if r["type"] == "app_side_slowdown"),
            by_type={t: dict(n=len([r for r in R if r["type"] == t]),
                             fail=float(np.mean([r["fail"] for r in R if r["type"] == t])),
                             impact=float(np.mean([r["impact_min"] for r in R if r["type"] == t])),
                             false=sum(r["false_changes"] for r in R if r["type"] == t))
                     for t in S.CFG["fault_mix"]})
        # paired comparison against threshold+hysteresis on recoverable scenarios
        if p != "threshold":
            T = {r["sid"]: r for r in rows if r["policy"] == "threshold"}
            d_imp = [r["impact_min"] - T[r["sid"]]["impact_min"] for r in rec]
            d_fail = [r["fail"] - T[r["sid"]]["fail"] for r in rec]
            o["paired_vs_threshold"] = dict(impact=boot(d_imp), fail=boot(d_fail),
                                            better=int(sum(x < 0 for x in d_imp)), worse=int(sum(x > 0 for x in d_imp)))
        out[p] = o
    return out


# --------------------------------------------------------------------------- figures
def fig_architecture(path):
    fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 110); ax.set_ylim(0, 56); ax.axis("off")
    def box(x, y, w, h, title, sub, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=11, weight="bold", color="#16324F")
        ax.text(x + w / 2, y + h - 7.2, sub, ha="center", va="top", fontsize=8.8, color="#333333", linespacing=1.35)
    # zones
    ax.add_patch(FancyBboxPatch((1, 1), 50, 54, boxstyle="round,pad=0.2,rounding_size=1.5", fc="#F1F6FA", ec="#9FB3C8", lw=1, ls="--"))
    ax.text(3, 53, "CUSTOMER-MANAGED", fontsize=9, color="#5B6573", weight="bold")
    ax.add_patch(FancyBboxPatch((79, 1), 30, 54, boxstyle="round,pad=0.2,rounding_size=1.5", fc="#F7F7F7", ec="#9FB3C8", lw=1, ls="--"))
    ax.text(81, 53, "SAP-MANAGED (RISE)", fontsize=9, color="#5B6573", weight="bold")
    box(4, 36, 20, 14, "Users & apps", "Fiori / SAP GUI\nCPI, middleware,\nfile & print", "white", "#16324F")
    box(28, 36, 20, 14, "Path controller", "risk model, gates,\nhysteresis, failback,\naudit log", "#E8F1FA", OI["blue"])
    box(4, 7, 20, 22, "Telemetry", "BFD + SLA probes\nFiori synthetic tx\nRFC_PING per path\nST03N / Cloud ALM\nreal-user P95", "white", "#16324F")
    box(28, 7, 20, 22, "Customer edge", "SD-WAN app policy\n(outbound steering)\nBGP prepend/MED\n(return path)\nstateful FW cluster", "white", "#16324F")
    box(82, 18, 24, 20, "SAP Cloud ERP\nPrivate ingress", "\n\nS/4HANA app servers\nWeb Dispatcher\nrouting inside\nSAP's boundary", "white", "#5B6573")
    # paths
    ax.annotate("", xy=(82, 33), xytext=(48, 22), arrowprops=dict(arrowstyle="-|>", lw=3, color=OI["blue"]))
    ax.text(62, 31.5, "Primary: private circuit\n1,000 Mbps", fontsize=9, color=OI["blue"], ha="center")
    ax.annotate("", xy=(82, 23), xytext=(48, 14), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=OI["orange"], ls="--"))
    ax.text(66, 9.0, "Backup: IPsec tunnel\n600 Mbps (sized for Tier 1+2)", fontsize=9, color="#9A6700", ha="center")
    # controls
    for (x1, y1, x2, y2) in [(24, 43, 28, 43), (24, 18, 28, 40), (38, 36, 38, 29)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#5B6573"))
    ax.text(39, 32.3, "route\npreference", fontsize=8, color="#5B6573")
    ax.text(94, 6, "Synthetic probes traverse BOTH paths\nto the same SAP endpoints", fontsize=8.5, color="#5B6573", ha="center")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_timeline(path, res_summary):
    prm = res_summary["tuned"]
    train = S.build("train")
    models = {"pred_network": S.RiskModel(False).fit(train), "sap_aware": S.RiskModel(True).fit(train)}
    # first progressive-latency scenario in the TEST set
    sid = next(i for i in range(S.SIZES["test"]) if S.gen_scenario(S.SEEDS["test"], i)["meta"]["type"] == "progressive_latency")
    sc = S.gen_scenario(S.SEEDS["test"], sid)
    rt = S.run_policy(sc, "threshold", prm["threshold"], models, log=True)
    rs = S.run_policy(sc, "sap_aware", prm["sap_aware"], models, log=True)
    risk = models["sap_aware"].predict(S.features(sc["tel"]["P"], sc["user_ratio_P"], True))
    tmin = np.arange(S.W) * 0.5 + 0.5
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax[0].plot(tmin, sc["truth_p95"], color=OI["black"], lw=1.8, label="Tier 1 P95 if traffic stays on primary")
    ax[0].axhline(S.CFG["slo_p95_ms"], color=OI["vermillion"], ls="--", lw=1.4, label="SLO 1,800 ms")
    ax[0].set_ylabel("Tier 1 P95 (ms)"); ax[0].set_ylim(1000, max(2900, sc["truth_p95"].max() * 1.05))
    def first_fo(r): return next((e["t_s"] / 60 for e in r["events"] if e["action"] == "failover"), None)
    ft, fs = first_fo(rt), first_fo(rs)
    for x, c, lab in [(fs, OI["blue"], "SAP-aware failover"), (ft, OI["orange"], "Threshold+hysteresis failover")]:
        if x: ax[0].axvline(x, color=c, lw=2.2, label=f"{lab} ({x:.1f} min)"); ax[1].axvline(x, color=c, lw=2.2)
    ax[0].legend(loc="upper left", fontsize=8.8, frameon=False)
    ax[0].set_title(f"Test scenario {sid} (progressive latency, seed {S.SEEDS['test']})", fontsize=11, loc="left")
    ax[1].plot(tmin, risk, color=OI["blue"], lw=1.6); ax[1].axhline(prm["sap_aware"]["theta"], color=OI["grey"], ls=":", lw=1.3)
    ax[1].set_ylabel("Predicted\nSLO-threat risk"); ax[1].set_xlabel("Minutes into scenario"); ax[1].set_ylim(-0.03, 1.05)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return dict(sid=sid, onset_min=sc["meta"]["onset_s"] / 60, threshold_failover_min=ft, sap_failover_min=fs,
                first_breach_min=float(np.argmax(sc["truth_breach"]) * 0.5) if sc["truth_breach"].any() else None,
                fail_threshold=rt["fail"], fail_sap=rs["fail"], events_sap=rs["events"][:3])


def fig_results(path, tab):
    pols = S.POLICIES
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    y = np.arange(len(pols))[::-1]
    for i, p in enumerate(pols):
        m, lo, hi = tab[p]["impact_rec"]
        ax[0].errorbar(m, y[i], xerr=[[m - lo], [hi - m]], fmt="o", color=PCOL[p], ms=8, capsize=4, lw=2)
        ax[1].barh(y[i], tab[p]["false_changes_all"], color=PCOL[p], height=0.6)
        ax[1].text(tab[p]["false_changes_all"] + 0.6, y[i], str(tab[p]["false_changes_all"]), va="center", fontsize=9)
    for a in ax: a.set_yticks(y); a.set_yticklabels([S.LABEL[p] for p in pols])
    ax[1].set_yticklabels([])
    ax[0].set_xlabel("Minutes Tier 1 SLO breached per scenario\n(recoverable degradations, mean and 95% CI)")
    ax[1].set_xlabel("False path changes, all 600 test scenarios")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_capacity(path, summ):
    cap = summ["capacity_sweep"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for adm, col, lab, ls in [(True, OI["blue"], "Tier 3 shed on backup (admission control)", "-"),
                              (False, OI["orange"], "All production tiers moved to backup", "--")]:
        xs = [c["backup_mbps"] for c in cap if c["admission"] == adm]; ys = [c["fail_per_scn"] for c in cap if c["admission"] == adm]
        ax.plot(xs, ys, ls, marker="o", color=col, lw=2, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("Backup capacity (Mbps); primary = 1,000 Mbps")
    ax.set_ylabel("Failed Tier 1 transactions\nper scenario (log scale)")
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_threshold(path, summ):
    s = summ["theta_sensitivity"]; th = [x["theta"] for x in s]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].plot(th, [x["false_changes"] for x in s], "o-", color=OI["vermillion"], lw=2)
    ax[0].set_ylabel("False path changes (600 scenarios)"); ax[0].set_ylim(bottom=0)
    ax[1].plot(th, [x["fail_per_scn"] for x in s], "s-", color=OI["blue"], lw=2)
    ax[1].set_ylabel("Failed Tier 1 tx per scenario"); ax[1].set_ylim(80, 100)
    for a_ in ax: a_.set_xlabel("Risk threshold \u03b8 (SAP-aware policy, test set)")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results"); ap.add_argument("--fig", default="figures")
    a = ap.parse_args(); os.makedirs(a.fig, exist_ok=True)
    summ = json.load(open(f"{a.res}/summary.json")); rows = load_rows(a.res)
    tab = tables(rows)
    fig_architecture(f"{a.fig}/fig1_architecture.png")
    tl = fig_timeline(f"{a.fig}/fig2_timeline.png", summ)
    fig_results(f"{a.fig}/fig3_results.png", tab)
    fig_capacity(f"{a.fig}/fig4_capacity.png", summ)
    fig_threshold(f"{a.fig}/fig5_threshold.png", summ)
    json.dump(dict(tables=tab, timeline=tl), open(f"{a.res}/analysis.json", "w"), indent=1, default=str)
    print(json.dumps(dict(timeline=tl), indent=1, default=str)[:1500])
