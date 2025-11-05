"""
FLAIR-X Configuration
Central configuration for all hyperparameters and paths
"""
from pathlib import Path

# ============================================
# Paths
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
LEDGER_PATH = PROJECT_ROOT / "src" / "ledger" / "privacy_ledger.json"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# Model Configuration
# ============================================
NUM_CLASSES = 5
IMAGE_SIZE = 224
MODEL_NAME = "resnet18"  # or "efficientnet_b0"

# ============================================
# Training Configuration
# ============================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS_PER_ROUND = 5  # Local epochs per FL round
DEVICE = "cuda"  # Will auto-switch to "cpu" if no GPU

# ============================================
# Federated Learning Configuration
# ============================================
NUM_CLIENTS = 3
NUM_ROUNDS = 10
MIN_AVAILABLE_CLIENTS = 3  # All 3 hospitals must participate
MIN_FIT_CLIENTS = 3
MIN_EVALUATE_CLIENTS = 3

HOSPITAL_IDS = ["A", "B", "C"]

# ============================================
# Privacy Configuration
# ============================================
INITIAL_EPSILON = 1.0
INITIAL_DELTA = 1e-5
INITIAL_NOISE_MULTIPLIER = 0.5
MAX_GRAD_NORM = 1.0

# Risk thresholds
HIGH_RISK_THRESHOLD = 0.7
LOW_RISK_THRESHOLD = 0.3

# Noise adjustment rates
NOISE_INCREASE_RATE = 0.2  # Increase noise by 20% if high risk
NOISE_DECREASE_RATE = 0.1  # Decrease noise by 10% if low risk

# ============================================
# Dashboard Configuration
# ============================================
DASHBOARD_PORT = 8501
DASHBOARD_REFRESH_INTERVAL = 2  # seconds

# ============================================
# Logging
# ============================================
LOG_LEVEL = "INFO"
SAVE_MODEL_EVERY_N_ROUNDS = 2