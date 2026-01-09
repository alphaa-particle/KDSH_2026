import torch
import torch.nn as nn
import json
from transformers import DistilBertModel, DistilBertConfig

class BDHConsistencyClassifier(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        
        # Load the config file from the bdh folder
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
        except FileNotFoundError:
            # Fallback defaults if config is missing
            print("Warning: Config not found, using defaults.")
            cfg = {'vocab_size': 30522, 'hidden_size': 768, 'dropout': 0.1, 'num_labels': 2}

        # Setup the Transformer Backbone
        hf_config = DistilBertConfig(
            vocab_size=cfg.get('vocab_size', 30522), 
            dim=cfg.get('hidden_size', 768),
            dropout=cfg.get('dropout', 0.1),
            n_layers=6, 
            n_heads=12
        )
        self.encoder = DistilBertModel(hf_config)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(cfg.get('hidden_size', 768), cfg.get('hidden_size', 768)),
            nn.ReLU(),
            nn.Dropout(cfg.get('dropout', 0.1)),
            nn.Linear(cfg.get('hidden_size', 768), cfg.get('num_labels', 2))
        )

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Encode text
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. Get the [CLS] token (first token) representation
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # 3. Classify
        logits = self.classifier(cls_token)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
        return logits, loss