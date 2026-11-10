# Limitations and Responsible Interpretation

1. **Record-level split:** The dataset was split by records. Verified device-level or capture-session-level grouping was not available.
2. **Exploratory audit scope:** Earlier exploratory feature audits used the complete dataset before partitioning. This limits the independence of validation-based feature selection.
3. **TCP feature dependence:** Ablation experiments showed substantial sensitivity to TCP sequence-related features, especially `tcp.seq`. This dependence does not alone prove leakage, but may reflect dataset-specific patterns.
4. **Class-level weaknesses:** DDoS_HTTP precision was approximately 38.02%; Fingerprinting F1 approximately 50.90%; Uploading recall approximately 48.48%; Password recall approximately 60.37%.
5. **MITM support:** Duplicate removal affected MITM records, and the final test support was only 60 records.
6. **No external validation:** The final test result does not demonstrate performance on unseen devices, capture sessions, buildings or network environments.
7. **Uncalibrated scores:** Model probability-like outputs displayed by the application are not calibrated confidence values.
8. **Offline replay:** Simulation replays dataset records. It does not capture live packets or monitor a real network interface.
9. **Prototype scope:** Alerts are model predictions requiring human review. They are not independently verified incidents.

The project is an educational defensive prototype, not a production intrusion detection appliance.
