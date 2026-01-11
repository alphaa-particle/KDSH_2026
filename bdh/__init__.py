from transformers import GPT2TokenizerFast

class BDHTokenizer:
    @staticmethod
    def from_pretrained(model_name="bdh-base"):
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

from .bdh import BDH, BDHConfig
