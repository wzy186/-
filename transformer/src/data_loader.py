import torch
from torch.utils.data import Dataset, DataLoader
import requests
import numpy as np


class TinyShakespeare(Dataset):
    """Tiny Shakespeare dataset loader"""
    
    def __init__(self, seq_len=128):
        self.seq_len = seq_len
        self.data = self._download_data()
        self.vocab, self.vocab_size = self._build_vocab()
        self.data_encoded = self._encode_data()
        
    def _download_data(self):
        """Download Tiny Shakespeare dataset"""
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except:
            # Fallback to sample data if download fails
            return "KING: What means this?\nHAMLET: I know not.\n"
    
    def _build_vocab(self):
        """Build character vocabulary"""
        chars = sorted(list(set(self.data)))
        vocab = {ch: i for i, ch in enumerate(chars)}
        print(f"Vocabulary size: {len(vocab)}")
        return vocab, len(vocab)
    
    def _encode_data(self):
        """Encode text data to token indices"""
        encoded = [self.vocab[ch] for ch in self.data]
        return torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self):
        return len(self.data_encoded) // self.seq_len
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        sequence = self.data_encoded[start_idx:end_idx]
        return sequence[:-1], sequence[1:]  # input, target


def create_data_loader(batch_size=32, seq_len=128, train_ratio=0.9):
    """Create data loaders for training and validation"""
    dataset = TinyShakespeare(seq_len)
    
    # Split dataset
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    vocab_info = {
        'stoi': dataset.vocab,
        'itos': {i: ch for ch, i in dataset.vocab.items()},
        'size': dataset.vocab_size
    }
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    return train_loader, val_loader, vocab_info
