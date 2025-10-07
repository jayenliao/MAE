import random
import torch
import numpy as np
import csv, os
from typing import List, Dict, Any

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CSVLogger:
    def __init__(self, filepath: str, fieldnames: List[str]):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._init_file()

    def _init_file(self):
        new_file = not os.path.exists(self.filepath)
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if new_file:
                writer.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
