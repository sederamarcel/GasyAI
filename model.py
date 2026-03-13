import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class SmallMalagasyLLM(nn.Module):
    def __init__(self, vocab_size=16000, hidden_size=768, num_layers=20, 
                 num_heads=12, num_kv_heads=4, use_checkpointing=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_checkpointing = use_checkpointing
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size*4,
                batch_first=True,
                dropout=0.1,
                activation='gelu'
            )
            self.layers.append(layer)
            
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        x = self.embedding(x)
        
        # Gradient checkpointing for memory efficiency
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
                
        x = self.ln_f(x)
        return self.lm_head(x)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
