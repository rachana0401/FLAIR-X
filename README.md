## FLAIR-X: Federated Learning with Adaptive Privacy

FLAIR-X is a federated learning system for diabetic retinopathy classification across multiple hospitals with adaptive privacy controls. It simulates three hospitals (A, B, C), trains sequentially, adaptively adjusts differential privacy noise per client based on risk, and logs all privacy/training metrics to a privacy ledger visualized in a Streamlit dashboard.

### Key Features
- Adaptive privacy engine that increases/decreases DP noise per client based on risk
- Sequential federated training loop (no Ray) that works reliably on macOS/Windows/Linux
- Privacy ledger (`src/ledger/privacy_ledger.json`) with append-only round logs and events
- Streamlit dashboard with privacy/training visualizations

---

## What’s Already Done

- Federated client (`src/federated/client.py`) with local training, evaluation, and DP noise injection
- Model factory (`src/federated/model.py`) supporting `resnet18`/`efficientnet_b0`
- Data utilities (`src/federated/utils.py`) for loading datasets and training/evaluation helpers
- Adaptive privacy strategy (Flower) prototype (`src/federated/server.py`) with risk-based noise adjustments
- Sequential training pipeline (`src/main.py`) that trains across hospitals and logs rounds to ledger
- Privacy ledger module (`src/ledger/logger.py`) to log rounds and events; summary APIs
- Adaptive privacy engine (`src/privacy/adaptive_engine.py`)
- DP wrapper (`src/privacy/dp_wrapper.py`) with gradient clipping/noise and rough privacy cost
- Privacy auditor (`src/privacy/audit.py`) for risk assessment and anomaly detection
- Visualization utilities (`src/visualization/plots.py`) and Streamlit dashboard (`src/visualization/dashboard.py`)

---

## What’s Left / Future Work

- Replace simplified risk heuristics with stronger privacy auditing (e.g., Rényi DP accountant, membership inference tests)
- Integrate a production-grade DP library (e.g., Opacus) for precise accounting and per-batch noise/clipping
- Add model checkpointing to `outputs/models/` every N rounds (config-driven)
- Extend dashboard with live refresh toggle and historical comparisons across runs
- Add unit tests for ledger logger, privacy auditor, and visualization functions
- Optional: Enable Flower simulation mode to run parallel clients for speed

---

## Intended Direction

- Improve privacy accounting and risk scoring fidelity
- Broaden model options and augmentations for robustness
- Add exportable reports from the dashboard (PDF/CSV)
- Parameterize data splits and hospital counts via config/CLI

---

## Project Structure

```
src/
  federated/
    client.py            # Hospital client (training/eval + DP noise)
    model.py             # Model factory (ResNet/EfficientNet)
    server.py            # Flower strategy prototype (adaptive privacy)
    utils.py             # Data loaders, training/eval helpers
  ledger/
    logger.py            # Privacy ledger writer/reader
    privacy_ledger.json  # Append-only ledger data
  privacy/
    adaptive_engine.py   # Adaptive privacy engine (risk → noise)
    dp_wrapper.py        # DP ops (clipping, noise, privacy cost)
    audit.py             # Risk assessment and anomaly detection
  visualization/
    plots.py             # Plot helpers
    dashboard.py         # Streamlit dashboard
  config.py              # Hyperparameters, paths, dashboard config
  main.py                # Sequential FL entrypoint (logs to ledger)
```

Data directories (already included): `data/hospital_{A,B,C}/{train,val,test}/0..4/*.png`

---

## Setup

1) Create/activate the virtual environment (already provided as `myenv/`):
```bash
source /Users/rachana/Desktop/FLAIR-X/myenv/bin/activate
```

2) Install dependencies (first time only):
```bash
pip install -r /Users/rachana/Desktop/FLAIR-X/requirements.txt
```

If `python` is not found, use `python3`. If `streamlit` is not found, you can use the absolute path to the venv’s binary (see below).

---

## How to Run Training

Run the sequential federated simulation:
```bash
python /Users/rachana/Desktop/FLAIR-X/src/main.py
```

Tips for a quick run:
- In `src/config.py`, lower compute for fast smoke tests:
  - `NUM_ROUNDS = 1`
  - `NUM_EPOCHS_PER_ROUND = 1`

Outputs during training:
- Round-by-round logs printed to terminal
- Risk-based noise adjustments per hospital
- Privacy ledger updated at `src/ledger/privacy_ledger.json`

---

## How to Run the Dashboard

In another terminal (venv activated), run:
```bash
python -m streamlit run /Users/rachana/Desktop/FLAIR-X/src/visualization/dashboard.py
```

Or run Streamlit directly from the venv without activating it:
```bash
/Users/rachana/Desktop/FLAIR-X/myenv/bin/streamlit run /Users/rachana/Desktop/FLAIR-X/src/visualization/dashboard.py
```

Open the URL printed by Streamlit (default `http://localhost:8501`).

If the dashboard shows “No data available,” start training first and refresh the page after the first round completes. The dashboard reads from `src/ledger/privacy_ledger.json`.

Change port if busy:
```bash
streamlit run /Users/rachana/Desktop/FLAIR-X/src/visualization/dashboard.py --server.port 8502
```

---

## Configuration

Edit `src/config.py`:
- `NUM_ROUNDS`, `NUM_EPOCHS_PER_ROUND`, `BATCH_SIZE`, `LEARNING_RATE`
- Privacy: `INITIAL_NOISE_MULTIPLIER`, thresholds and adjustment rates
- Paths and dashboard settings

Model selection:
```python
MODEL_NAME = "resnet18"  # or "efficientnet_b0"
```

---

## Troubleshooting

- Command not found: streamlit
  - Activate venv, or use absolute path:
    ```bash
    /Users/rachana/Desktop/FLAIR-X/myenv/bin/streamlit run /Users/rachana/Desktop/FLAIR-X/src/visualization/dashboard.py
    ```

- Dashboard shows “No data available”
  - Ensure training ran at least one round
  - Check ledger size and contents:
    ```bash
    wc -c /Users/rachana/Desktop/FLAIR-X/src/ledger/privacy_ledger.json
    cat /Users/rachana/Desktop/FLAIR-X/src/ledger/privacy_ledger.json
    ```

- Slow training
  - Reduce `NUM_ROUNDS` and `NUM_EPOCHS_PER_ROUND`, or lower `IMAGE_SIZE` in config

- Missing packages
  - Re-run: `pip install -r requirements.txt`

---

## Roadmap Checklist

- [x] Sequential FL training and client implementations
- [x] Adaptive privacy engine and server-side noise control
- [x] Privacy ledger and dashboard visualization
- [ ] Advanced DP accounting (Opacus / Rényi DP)
- [ ] Membership inference tests and stronger auditing
- [ ] Checkpointing and experiment tracking (e.g., MLflow/W&B)
- [ ] Parallelized simulations and scalability tests

---

## License

Internal research project; license to be defined.


