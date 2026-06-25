"""
Fix for issue #2 (config settings bug) + issue #4 (logging).
Central configuration for all AIX experiments.
"""
import os

# ── Model ──────────────────────────────────────────────────────────────────
VOCAB_SIZE    = int(os.getenv("VOCAB_SIZE",    "20000"))
EMBED_DIM     = int(os.getenv("EMBED_DIM",     "128"))
LSTM_UNITS    = int(os.getenv("LSTM_UNITS",    "128"))
DROPOUT_RATE  = float(os.getenv("DROPOUT_RATE", "0.4"))
L2_WEIGHT     = float(os.getenv("L2_WEIGHT",   "0.001"))
MAX_SEQ_LEN   = int(os.getenv("MAX_SEQ_LEN",   "256"))

# ── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE    = int(os.getenv("BATCH_SIZE",   "64"))
EPOCHS        = int(os.getenv("EPOCHS",       "30"))
LEARNING_RATE = float(os.getenv("LR",         "0.001"))
PATIENCE      = int(os.getenv("PATIENCE",     "5"))    # EarlyStopping
LR_PATIENCE   = int(os.getenv("LR_PATIENCE",  "3"))    # ReduceLROnPlateau

# ── MIA Attack ──────────────────────────────────────────────────────────────
SHADOW_MODELS = int(os.getenv("SHADOW_MODELS", "4"))
MIA_EPOCHS    = int(os.getenv("MIA_EPOCHS",   "10"))

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR      = os.getenv("DATA_DIR",    "data/")
MODEL_DIR     = os.getenv("MODEL_DIR",   "models/")
RESULTS_DIR   = os.getenv("RESULTS_DIR", "results/")

# ── Differential Privacy (optional) ─────────────────────────────────────────
DP_ENABLED    = os.getenv("DP_ENABLED", "false").lower() == "true"
DP_L2_CLIP    = float(os.getenv("DP_L2_CLIP",  "1.0"))
DP_NOISE_MULT = float(os.getenv("DP_NOISE",    "0.1"))
DP_MICROBATCH = int(os.getenv("DP_MICROBATCH", "256"))
