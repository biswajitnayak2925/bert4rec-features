import os

DATA_DIR = "./data"
RATINGS_PATH  = os.path.join(DATA_DIR, "Final_Ratings.csv")
ID_PATH       = os.path.join(DATA_DIR, "model_id.csv")
GENRE_PATH    = os.path.join(DATA_DIR, "model_genre.csv")
METADATA_PATH = os.path.join(DATA_DIR, "model_metadata.csv")
PLOT_PATH     = os.path.join(DATA_DIR, "model_plot.csv")


OUTPUT_DIR = "./outputs_publication"

MAX_SEQ_LEN = 20
MIN_USER_INTERACTIONS = 5
MAX_USER_INTERACTIONS = None

HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_HEADS = 4
INNER_SIZE = 256
DROPOUT = 0.2
MASK_PROB = 0.2
GENRE_DIM = 32
METADATA_DIM = 32
PLOT_PROJ_DIM = 64

BATCH_SIZE = 128
NUM_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 0.0

PATIENCE = 20
EVAL_EVERY = 4
NUM_EVAL_NEGATIVES = 100
TOP_K = 10


SEEDS = [42]

VARIANTS = ["M1_id", "M2_genre", "M3_metadata", "M4_plot", "M5_all"]
SPARSITY_LEVELS = [1.0, 0.50, 0.25, 0.10]

DEVICE = "cuda"
