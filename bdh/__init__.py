# FILE: bdh/__init__.py
from transformers import AutoTokenizer

# 1. The Tokenizer Wrapper (Keep this so main.py can load a tokenizer)
class BDHTokenizer:
    @staticmethod
    def from_pretrained(model_name):
        if model_name == "bdh-base":
            model_name = "distilbert-base-uncased"
        return AutoTokenizer.from_pretrained(model_name)

# 2. Import the Real Model Architecture
from .bdh import BDH, BDHConfig