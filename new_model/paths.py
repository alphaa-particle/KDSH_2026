import os

# Absolute path of new_model/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
