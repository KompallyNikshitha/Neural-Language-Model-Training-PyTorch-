import torch
import re

def load_and_preprocess(path):
    """Load and clean text data"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    text = re.sub(r'[^a-zA-Z0-9\s.,;!?\'"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    vocab = sorted(set(tokens))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    encoded = [word2idx[word] for word in tokens]
    return encoded, word2idx, idx2word

def split_data(encoded, train_ratio=0.8, val_ratio=0.1):
    train_end = int(len(encoded) * train_ratio)
    val_end = int(len(encoded) * (train_ratio + val_ratio))
    train_data = encoded[:train_end]
    val_data = encoded[train_end:val_end]
    test_data = encoded[val_end:]
    return train_data, val_data, test_data

def get_batches(data, seq_len, batch_size):
    for i in range(0, len(data) - seq_len, seq_len):
        x = data[i:i+seq_len]
        y = data[i+1:i+seq_len+1]
        yield torch.tensor(x).unsqueeze(0), torch.tensor(y).unsqueeze(0)
