# SAP-aware path selection simulator (RISE with SAP)

Reproduces every number and figure in the article *When Should SAP Traffic Leave the Primary Link?*

```
pip install -r requirements.txt
python sap_path_sim.py --out results      # ~5 min on a laptop
python example_record.py
python analyze.py --res results --fig figures
```

* Seeds: train 1001 (400 scenarios), validation 2002 (200), test 3003 (600).
* All model parameters are in `CFG` at the top of `sap_path_sim.py`.
* `results/test_rows.csv` holds one row per (scenario, policy) on the test set.
* `results/summary.json` holds config, tuned parameters, model coefficients, tuning log, sensitivity sweeps.
* `results/analysis.json` holds the bootstrap tables used in the article.

All data are synthetic. Nothing here is an SAP benchmark or a measurement from any customer or employer environment.
