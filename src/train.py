import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def fit(
    model, item_vocab_size, train_loader, test_seqs, epochs, learning_rate, weight_decay=1e-5,
    device='cpu', max_seq_len=None, pad_token=None, early_stopping_patience=3
):
    import copy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model = model.to(device)
    hr, ndcg = 0, 0
    vocab_size = item_vocab_size + 2

    best_hr = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (masked_seqs, labels) in enumerate(train_loader):
            masked_seqs, labels = masked_seqs.to(device), labels.to(device)
            logits = model(masked_seqs)
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)
            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

        if test_seqs is not None and max_seq_len is not None and pad_token is not None:
            from src.evaluate import evaluate_next_item
            hr, ndcg = evaluate_next_item(
                model, test_seqs, max_seq_len, pad_token, K=10
            )
            print(f"Hit Rate@10: {hr:.4f}  NDCG@10: {ndcg:.4f}")

            if hr > best_hr:
                best_hr = hr
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement in HR@10 for {patience_counter} epochs.")
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered!")
                    model.load_state_dict(best_model_wts)
                    break

    model.load_state_dict(best_model_wts)
    return model, hr, ndcg