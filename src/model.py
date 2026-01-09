import torch
import torch.nn as nn
import json
from bdh import BDH, BDHConfig

class BDHConsistencyClassifier(nn.Module):
    def __init__(self, config_path, class_weights=None):
        super().__init__()
        
        # 1. Load Configuration FIRST
        # You cannot create self.encoder until self.config exists!
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

        # 2. Initialize Architecture
        self.encoder = BDH(self.config)
        self.class_weights = class_weights
        
        self.classifier = nn.Sequential(
            nn.Linear(self.config.n_embd, self.config.n_embd),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.n_embd, 2)
        )

        # 3. Initialize Weights
        # This MUST come last, after all layers are defined
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights with a small normal distribution.
        This helps the model start training effectively from scratch.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Get Hidden States
        hidden_states, _ = self.encoder(input_ids)
        
        # 2. Extract the First Token (CLS equivalent)
        # Shape: (Batch, 1, Seq, Dim) -> (Batch, Seq, Dim) -> (Batch, Dim)
        cls_token = hidden_states.squeeze(1)[:, 0, :]
        
        # 3. Classify
        logits = self.classifier(cls_token)
        
        # 4. Calculate Loss
        loss = None
        if labels is not None:
            if self.class_weights is not None:
                # Ensure weights are on the same device as the input
                weight_tensor = torch.tensor(self.class_weights).to(input_ids.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
            else:
                loss_fct = nn.CrossEntropyLoss()
                
            loss = loss_fct(logits, labels)
            
        return logits, loss