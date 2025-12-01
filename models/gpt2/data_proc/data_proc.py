import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class BinaryTokenizer:
    def __init__(self):
        self.stoi = {"0": 0, "1": 1}
        self.itos = {0: "0", 1: "1"}

    def tokenize(self, binary_data):
        if isinstance(binary_data, str):
            return [self.stoi[b] for b in binary_data]
        return binary_data  # Already numeric

    def detokenize(self, tokens):
        return "".join([self.itos[t] for t in tokens])


class NBitsTokenizer:
    def __init__(self, n_bits):
        self.target_bits = n_bits

    def _create_stoi_mapping(self, n_bits):
        max_val = 2**n_bits
        stoi = {"{:0{}b}".format(i, n_bits): i for i in range(max_val)}
        return stoi

    def tokenize(self, binary_data):
        assert (
            len(binary_data) % self.target_bits == 0
        ), "Length of binary_data must be multiple of n_bits."

        return [
            int(binary_data[i : i + self.target_bits], 2)
            for i in range(0, len(binary_data), self.target_bits)
        ]

    def detokenize(self, tokens):
        return "".join(
            [format(token, "0{}b".format(self.target_bits)) for token in tokens]
        )


class BinaryDataset(Dataset):
    """Dataset for binary sequence prediction with configurable target bits."""
    
    def __init__(
        self, binary_data, config, start_index=0, end_index=None, is_training=False
    ):
        data_slice = binary_data[start_index:end_index]
        np_data = np.frombuffer(data_slice, dtype=np.uint8)
        self.bits = np.unpackbits(np_data)
        
        if is_training and config["is_autoregressive"]:
            self.target_bits = 1
        else:
            self.target_bits = config["target_bits"]
        
        self.seqlen = config["seqlen"]
        self.step = config["step"]
        self.batch_size = config["batch_size"]
        self.num_classes = 2 ** self.target_bits
        self.powers = 2 ** np.arange(self.target_bits - 1, -1, -1)
        
        # Precompute target indices offset matrix (relative to each position)
        self.target_offsets = np.arange(self.seqlen)[:, None] + np.arange(1, self.target_bits + 1)

    def __len__(self):
        total_steps = (len(self.bits) - self.seqlen - self.target_bits) // self.step
        num_batches = total_steps // self.batch_size
        return num_batches * self.batch_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        
        start = idx * self.step
        end = start + self.seqlen + self.target_bits
        bits_slice = self.bits[start:end]
        
        input_sequence = torch.from_numpy(bits_slice[:self.seqlen].astype(np.int64))
        
        target_bits_matrix = bits_slice[self.target_offsets]
        target_indices = np.dot(target_bits_matrix, self.powers)
        
        target_sequence = torch.from_numpy(target_indices.astype(np.int64))
        target_one_hot = F.one_hot(target_sequence, num_classes=self.num_classes).float()
        
        return input_sequence, target_one_hot


def load_and_prepare_data(config, num_workers=4):
    with open(config["filename"], "rb") as f:
        binary_data = f.read(config["num_bytes"])

    total_length = len(binary_data)
    train_length = int(total_length * config["train_ratio"])

    train_dataset = BinaryDataset(
        binary_data, config, end_index=train_length, is_training=True
    )
    eval_dataset = BinaryDataset(binary_data, config, start_index=train_length)

    return (
        DataLoader(
            train_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True, 
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        ),
        DataLoader(
            eval_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True, 
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        ),
    )


if __name__ == "__main__":
    print("This is a module for loading and preparing data for the GPT-2 model.")
