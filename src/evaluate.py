import torch
import numpy as np

def evaluate_next_item(model, test_data, max_seq_len, pad_token, K=10):
    model.eval()
    hits, ndcgs = [], []
    with torch.no_grad():
        for user_id, seq, true_items in test_data:
            seq = seq[-max_seq_len:]
            pad_len = max_seq_len - len(seq)
            input_seq = [pad_token] * pad_len + seq
            input_seq = torch.tensor(input_seq).unsqueeze(0).to(next(model.parameters()).device)
            logits = model(input_seq)
            scores = logits[0, -1]
            topk = scores.topk(K).indices.cpu().numpy()
            true_item = true_items[0]
            hits.append(int(true_item in topk))
            if true_item in topk:
                rank = list(topk).index(true_item)
                ndcgs.append(1 / np.log2(rank + 2))
            else:
                ndcgs.append(0)
    hr = np.mean(hits)
    ndcg = np.mean(ndcgs)
    print(f"Hit Rate@{K}: {hr:.4f}  NDCG@{K}: {ndcg:.4f}")
    return hr, ndcg