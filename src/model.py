import torch
import torch.nn as nn
import json
from bdh import BDH, BDHConfig

class BDHConsistencyClassifier(nn.Module):

    
    def __init__(self, config_path, class_weights=None):
        super().__init__()
        
        try:
            with open(config_path, 'r') as f:
                json_cfg = json.load(f)
        except FileNotFoundError:
            json_cfg = {}

        self.config = BDHConfig(
            vocab_size=json_cfg.get('vocab_size', 30522),
            n_embd=json_cfg.get('hidden_size', 256),
            n_layer=json_cfg.get('n_layers', 6),
            n_head=json_cfg.get('n_heads', 4),
            dropout=json_cfg.get('dropout', 0.1),
            mlp_internal_dim_multiplier=json_cfg.get('mlp_internal_dim_multiplier', 128)
        )

        self.encoder = BDH(self.config)
        self.class_weights = class_weights
        self.classifier = nn.Sequential(
            nn.Linear(self.config.n_embd, self.config.n_embd),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.n_embd, 2)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Get Hidden States
        # Shape comes out as: (Batch, 1, Sequence_Length, Hidden_Dim)
        hidden_states, _ = self.encoder(input_ids)
        
        # 2. Extract the First Token (CLS)
        # We need to squeeze the singleton dimension (1) first -> (Batch, Seq, Dim)
        # Then take the 0th index of the sequence -> (Batch, Dim)
        cls_token = hidden_states.squeeze(1)[:, 0, :]
        
        # 3. Classify
        logits = self.classifier(cls_token)
        
        # 4. Calculate Loss
        loss = None
        if labels is not None:
            # Check if weights exist
            if self.class_weights is not None:
                weight_tensor = torch.tensor(self.class_weights).to(input_ids.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
            else:
                loss_fct = nn.CrossEntropyLoss()
                
            loss = loss_fct(logits, labels)
            
        return logits, loss