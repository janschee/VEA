import os
from typing import Literal

ROOT: str = os.path.abspath(os.path.dirname(__file__))
API_KEY: str = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
METADATA_CSV: str = os.path.join(ROOT, "./thumbnails/metadata.csv")
METADATA_JSON: str = os.path.join(ROOT, "./thumbnails/metadata.json")

MODE: Literal["train", "test"] = "train"
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 15
LEARNING_RATE: float = 10e-6
