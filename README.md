FLAIR-X: Federated Learning with Adaptive Privacy

FLAIR-X is a federated learning framework for diabetic retinopathy classification across multiple hospitals.
It adaptively adjusts differential privacy noise per client based on risk and logs all metrics to a privacy ledger visualized through a Streamlit dashboard.

⚙️ Key Features

Adaptive differential privacy engine (risk-based noise control)

Sequential federated training (works on macOS/Windows/Linux)

Append-only privacy ledger for all rounds

Streamlit dashboard for live training & privacy visualization

🏗️ Structure
src/
  federated/     # Clients, models, server, utils
  privacy/       # DP engine, wrapper, audit tools
  ledger/        # Logger + privacy_ledger.json
  visualization/ # Dashboard + plots
  config.py      # Settings
  main.py        # FL entry point


Data: data/hospital_{A,B,C}/{train,val,test}/

🚀 Setup
python3 -m venv myenv
source myenv/bin/activate         # or myenv\Scripts\activate (Windows)
pip install -r requirements.txt

🧩 Run Training
python src/main.py


For quick tests, edit src/config.py:

NUM_ROUNDS = 1
NUM_EPOCHS_PER_ROUND = 1


Outputs:

Terminal logs per round

Adaptive noise updates

Ledger saved at src/ledger/privacy_ledger.json

📊 Run Dashboard
streamlit run src/visualization/dashboard.py


Default: http://localhost:8501

If busy:

streamlit run src/visualization/dashboard.py --server.port 8502

⚙️ Config Highlights

In src/config.py:

Model: "resnet18" or "efficientnet_b0"

Privacy: noise multiplier, thresholds

Training: rounds, epochs, batch size

🧰 Troubleshooting

Streamlit not found:
source myenv/bin/activate or
myenv/bin/streamlit run src/visualization/dashboard.py

No data on dashboard:
Run training first → refresh page

Slow training:
Lower image size, rounds, or epochs
