# Baseline Model Artifacts

Training command:

```bash
/Users/ajneya/Desktop/FYP/scam-webapp/.venv/bin/python /Users/ajneya/Desktop/FYP/scam-webapp/scripts/train_baseline_model.py
```

Produced files:
- `/Users/ajneya/Desktop/FYP/scam-webapp/data/models/baseline_model.joblib`
- `/Users/ajneya/Desktop/FYP/scam-webapp/data/models/baseline_metrics.json`

Runtime path:
- `src/model_runtime.py` loads the model from `scam-webapp/data/models/baseline_model.joblib`.
- If model/dependencies are unavailable, app falls back to `mock_model`.

To run app with trained model:

```bash
cd /Users/ajneya/Desktop/FYP/scam-webapp
.venv/bin/streamlit run app.py
```
