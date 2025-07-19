import torch
import torch.nn as nn

class BERT4Rec(nn.Module):
    def __init__(self, item_vocab_size, max_seq_len, d_model=128, n_heads=2, n_layers=2, dropout=0.1):
        super().__init__()
        self.item_vocab_size = item_vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.item_embedding = nn.Embedding(item_vocab_size + 2, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, item_vocab_size + 2)

    def forward(self, input_ids, attention_mask=None):
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        
        if attention_mask is None:
            attention_mask = (input_ids != 0)
        out = self.transformer_encoder(
            x, src_key_padding_mask=~attention_mask.bool()
        )
        logits = self.output(out)
        return logits
