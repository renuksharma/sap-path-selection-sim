# Alt text for DZone image upload

**fig1_architecture.png**: Architecture: customer-managed telemetry and path controller set route preference on the customer edge; a 1,000 Mbps private circuit and a 600 Mbps IPsec backup reach the SAP-managed private-cloud ingress; synthetic SAP probes traverse both paths.

**fig2_timeline.png**: Tier 1 P95 over 60 minutes in test scenario 0 rising past the 1,800 ms SLO at about minute 38; SAP-aware failover at 38.5 min, threshold failover at 39.5 min; lower panel shows predicted risk rising from minute 33.

**fig3_results.png**: Left: SLO-breach minutes per recoverable scenario with 95% CIs for six policies; right: false path changes across 600 test scenarios (operator 11, BFD-only 0, threshold 56, weighted 21, predictive network-only 7, SAP-aware 7).

**fig4_capacity.png**: Failed Tier 1 transactions per scenario versus backup capacity (log scale): with Tier 3 shed, failures flatten at 600 Mbps; without admission control the same floor needs about 800 Mbps.

**fig5_threshold.png**: False path changes fall from 44 to 7 as the risk threshold rises from 0.3 to 0.99; failed Tier 1 transactions per scenario change only slightly.
