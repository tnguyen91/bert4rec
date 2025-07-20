from torch.utils.data import DataLoader
from src.model import BERT4Rec
from src.data_loader import BERT4RecDataset
from src.utils import train_test_split_sequences, load_dataset, preprocess_data
from src.train import fit
import torch

def main():
    ratings, anime = load_dataset()
    df = preprocess_data(ratings_df=ratings)

    print(df)
    anime2id = {aid: idx for idx, aid in enumerate(df['anime_id'].unique())}
    df['anime_idx'] = df['anime_id'].map(anime2id)

    user_sequences = df.groupby('user_id')['anime_idx'].apply(list).tolist()
    print(len(user_sequences))

    train_sequences, test_seqs = train_test_split_sequences(user_sequences, n_test=1)
    print(train_sequences[:3])
    print(test_seqs[:3])


    item_vocab_size = len(anime2id)
    max_seq_len = 100
    batch_size = 128
    mask_prob = 0.15
    epochs = 20
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_token = item_vocab_size
    pad_token = item_vocab_size + 1

    train_dataset = BERT4RecDataset(
        train_sequences,
        max_seq_len=max_seq_len,
        mask_prob=mask_prob,
        item_vocab_size=item_vocab_size,
        seed=42,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = BERT4Rec(
        item_vocab_size=item_vocab_size,
        max_seq_len=max_seq_len,
        d_model=128,
        n_heads=2,
        n_layers=2,
        dropout=0.1
    )

    fit(
        model=model,
        item_vocab_size=item_vocab_size,
        train_loader=train_loader,
        test_seqs=test_seqs,
        epochs=epochs,
        learning_rate=lr,
        device=device,
        max_seq_len=max_seq_len,
        pad_token=pad_token
    )

    torch.save(model.state_dict(), 'bert4rec.pth')

if __name__ == '__main__':
    main()
