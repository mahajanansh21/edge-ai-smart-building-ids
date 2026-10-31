# Initial architecture

```text
Public labelled IoT/IIoT dataset
        |
        v
Schema inspection and data-quality analysis
        |
        v
Leakage-aware split + train-fitted preprocessing
        |
        v
Classical ML model comparison
        |
        v
Saved model + preprocessing pipeline
        |
        v
Local CPU inference service
        |
        +--> CSV traffic analyzer
        +--> Offline dataset replay
        +--> Local prediction/alert log
                 |
                 v
          Streamlit dashboard
```

The first deployment target is a Windows laptop acting as an edge-computing simulation. Actual Raspberry Pi or smart-building hardware deployment is outside the initial scope.
