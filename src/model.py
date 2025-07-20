import torch
import torch.nn as nn

class BERT4Rec(nn.Module):
    def __init__(self, item_vocab_size, max_seq_len, d_model=128, n_heads=2, n_layers=2, dropout=0.1):
        super().__init__()
        self.item_vocab_size = item_vocab_size
        self.max_seq_len = max_seq_len

        self.mask_token = item_vocab_size
        self.pad_token = item_vocab_size + 1
        self.vocab_size = item_vocab_size + 2

        self.item_embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_token)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, self.vocab_size)

    def forward(self, input_ids):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        attention_mask = (input_ids != self.pad_token)
        out = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)
        logits = self.output(out)
        return logits