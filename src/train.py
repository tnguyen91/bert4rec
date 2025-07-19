import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def fit(model, item_vocab_size, train_loader, epochs, learning_rate, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for masked_seqs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            masked_seqs, labels = masked_seqs.to(device), labels.to(device)
            logits = model(masked_seqs)

            logits = logits.view(-1, item_vocab_size)
            labels = labels.view(-1)
            loss = F.cross_entropy(logits, labels, ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * masked_seqs.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

