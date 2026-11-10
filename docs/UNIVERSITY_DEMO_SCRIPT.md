# University Demonstration Script (about 5–7 minutes)

## 1. Project objective

“Smart buildings contain IoT and IIoT devices that exchange network traffic. This project demonstrates a local, CPU-based machine-learning prototype that classifies traffic as normal or attack and then predicts an attack category.”

## 2. Dataset

Open **Dataset Explorer**. Explain that the original Edge-IIoTset CSV has 157,800 records and 63 columns. Phase 3 retained 36 source features and the preprocessing pipeline produced 43 transformed columns. Show the full-dataset label charts, not only the first 1,500-row preview. The target tasks are binary `Attack_label` and multiclass `Attack_type` with 15 categories.

## 3. Model evaluation

Open **Model Evaluation**. State the final held-out random-record split results:

- Binary Random Forest: 93.64% accuracy and 88.96% Macro-F1.
- Multiclass Decision Tree: 83.51% accuracy and 82.69% Macro-F1.

Mention that the final test set was evaluated once after model freezing. State that record-level splitting and earlier full-data exploratory audits limit claims about unseen devices or sessions.

## 4. Manual inference

Open **Traffic Analyzer → Manual single record**. The form initializes 36 feature inputs with per-feature training medians/modes; edit a few inputs and run the two models. Explain that it is a manually entered feature vector, not a packet capture. The binary model outputs Normal/Attack. The multiclass model always selects one of the 15 known categories; it has no validated Unknown class. Any displayed predict_proba scores are uncalibrated model scores.

## 5. Batch inference

Upload a CSV with all 36 Phase 3 source feature columns, or select a validation example. Labels, when present for a validation demonstration, are reference-only and are not model inputs. Export the predictions CSV.

## 6. Offline replay and alerts

Open **Live Simulation**. Choose a stratified sample from the validation split and run it in batches. Explain that this is offline replay—not live network capture. Open **Threat Alerts** and show that attack predictions are labelled “review required”, not confirmed incidents. Download the alert history.

## 7. Edge performance

Open **Edge Performance**. Show CPU/memory usage, processing time and records/second measured on the current laptop. Explain that the measurement depends on the machine, batch size and current system load.

## 8. Limitations

- No live packet capture, active scanning, attack execution or production building control.
- Predictions are model outputs, not verified incidents.
- A classifier always chooses a known category; unknown/novel attack detection is not implemented.
- The split is record-level, not verified device/session-level.
- Earlier exploratory audits used the full dataset; the final test score does not prove generalization to unseen buildings.
- The MITM class has relatively few records after duplicate removal; interpret perfect-looking scores cautiously.
