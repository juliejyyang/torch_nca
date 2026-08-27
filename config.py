#@title Cellular Automata Parameters

CHANNEL_N = 16          # RBGA + 12 hidden
TARGET_PADDING = 16
TARGET_SIZE = 40
BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5
TARGET_EMOJIS = ["🪻", "🌷", "🌹"]
K = len(TARGET_EMOJIS)
EXPERIMENT_TYPE = "Regenerating"
EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]
USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch
TRAIN_STEPS = 8000      # training.py runs this many steps (matches the 8000-step convention used below)
