"""Writes results/example_record.json: the real inputs behind the SAP-aware failover
in test scenario 0 (used for the worked example in the article)."""
import json, numpy as np, sap_path_sim as S
summ = json.load(open("results/summary.json"))
train = S.build("train")
m = S.RiskModel(True).fit(train)
sc = S.gen_scenario(S.SEEDS["test"], 0)
X = S.features(sc["tel"]["P"], sc["user_ratio_P"], True)
r = m.predict(X)
res = S.run_policy(sc, "sap_aware", summ["tuned"]["sap_aware"], {"sap_aware": m, "pred_network": m}, log=True)
ev = next(e for e in res["events"] if e["action"] == "failover"); w = ev["window"]
z = (X[w] - m.mu) / m.sd
contrib = dict(zip(S.FEAT_NAMES_SAP, np.round(m.m.coef_[0] * z, 2).tolist()))
P, B = sc["tel"]["P"], sc["tel"]["B"]
rec = dict(t_s=ev["t_s"], decision="failover", reason=ev["reason"], decided_on_window=w,
           risk_primary=round(float(r[w]), 4), risk_prev_window=round(float(r[w-1]), 4), theta=summ["tuned"]["sap_aware"]["theta"],
           primary=dict(lat_p95_ms=round(float(P["p95"][w]), 1), loss_pct=round(float(P["loss"][w]), 2), jitter_ms=round(float(P["jit"][w]), 1),
                        fiori_ratio=round(float(P["fiori_ratio"][w]), 2), rfc_ratio=round(float(P["rfc_ratio"][w]), 2),
                        tier1_p95_ms=round(float(sc["truth_p95"][w]), 0)),
           backup=dict(eligible=bool(sc["eligible_b"][ev["t_s"] // 5 - 1]), lat_p95_ms=round(float(B["p95"][w]), 1),
                       loss_pct=round(float(B["loss"][w]), 2), jitter_ms=round(float(B["jit"][w]), 1), fiori_ratio=round(float(B["fiori_ratio"][w]), 2)),
           features=dict(zip(S.FEAT_NAMES_SAP, np.round(X[w], 3).tolist())), logit_contributions=contrib,
           intercept=round(float(m.m.intercept_[0]), 2), logit=round(float(m.m.intercept_[0] + (m.m.coef_[0] * z).sum()), 2))
json.dump(rec, open("results/example_record.json", "w"), indent=1)
print(json.dumps(rec, indent=1))
