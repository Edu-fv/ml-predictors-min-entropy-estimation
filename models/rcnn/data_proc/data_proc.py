import os
import sys
import mmap

import numpy as np

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, utils_path)


def bytes_to_bits(bytes_arr):
    np_arr = np.frombuffer(
        bytes_arr, dtype=np.uint8
    )  # This converts bytes to numpy array
    return np.unpackbits(np_arr)


def data_generator(config, epochs=1, start=0.0, end=1.0):
    """
    Vectorized data generator for RCNN model training.
    Yields exactly batch_size samples per batch (except possibly the last batch).
    
    X = bits[i*step : i*step + seqlen]  
    y = bits[i*step + seqlen : i*step + seqlen + target_bits]
    """
    target_bits = config["target_bits"]
    seqlen = config["seqlen"]
    step = config["step"]
    batch_size = config["batch_size"]
    num_y_classes = 2 ** target_bits
    powers = 2 ** np.arange(target_bits - 1, -1, -1)
    window_size = seqlen + target_bits
    
    # Calculate bytes needed to generate exactly batch_size samples
    bits_needed = (batch_size - 1) * step + window_size
    bytes_needed = int(np.ceil(bits_needed / 8))
    
    # Byte step to advance without overlap (based on samples consumed)
    bits_consumed_per_batch = batch_size * step
    byte_step = int(np.ceil(bits_consumed_per_batch / 8))
    
    epoch_count = 0
    while epoch_count < epochs:
        with open(config["filename"], "rb") as file:
            mapped_file = mmap.mmap(
                file.fileno(), length=config["num_bytes"], access=mmap.ACCESS_READ
            )
            if mapped_file.size() < config["num_bytes"]:
                raise ValueError("num_bytes is larger than file size")

            start_pos = int(start * config["num_bytes"])
            end_pos = int(end * config["num_bytes"])

            batch_start = start_pos
            while batch_start < end_pos:
                batch_end = min(batch_start + bytes_needed, end_pos)
                mapped_data = bytes_to_bits(mapped_file[batch_start:batch_end])
                data_len = len(mapped_data)

                n = (data_len - window_size) // step + 1
                if n <= 0:
                    batch_start += byte_step
                    continue
                
                n = min(n, batch_size)
                
                x_indices = np.arange(n)[:, None] * step + np.arange(seqlen)
                X = np.zeros((n, seqlen, 2), dtype=bool)
                X[np.arange(n)[:, None], np.arange(seqlen), mapped_data[x_indices]] = True
                
                y_start = np.arange(n) * step + seqlen
                y_bit_indices = y_start[:, None] + np.arange(target_bits)
                y_class_indices = np.dot(mapped_data[y_bit_indices], powers)
                y = np.zeros((n, num_y_classes), dtype=bool)
                y[np.arange(n), y_class_indices] = True

                yield X, y
                
                batch_start += byte_step

        epoch_count += 1


if __name__ == "__main__":
    print("This is a module for loading and preparing data for the rcnn model.")
