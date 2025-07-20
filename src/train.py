import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def fit(model, item_vocab_size, train_loader, test_seqs, epochs, learning_rate, device='cpu', max_seq_len=None, pad_token=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    vocab_size = item_vocab_size + 2

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for masked_seqs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            masked_seqs, labels = masked_seqs.to(device), labels.to(device)
            
            logits = model(masked_seqs)
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)
            loss = F.cross_entropy(logits, labels, ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * masked_seqs.size(0)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

        if test_seqs is not None and max_seq_len is not None and pad_token is not None:
            from src.evaluate import evaluate_next_item
            hr, ndcg = evaluate_next_item(
                model, test_seqs, max_seq_len, pad_token, K=10
            )