import os
import sys

import numpy as np
import tensorflow as tf

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, utils_path)
from utils.nice_log import nice_log


def gpu_config(memory_fraction=0.5):
    num_gpus = len(tf.config.list_physical_devices("GPU"))
    if num_gpus > 0:
        nice_log(f"GPUs available: {num_gpus}")
    else:
        nice_log("No GPUs available.")


def get_config(
    model_name,
    filename,
    generator,
    seqlen,
    step,
    num_bytes,
    target_bits,
    train_ratio,
    test_ratio,
    learning_rate,
    batch_size,
    epochs,
):
    model_name = f"{model_name}_bits"
    total_elements = num_bytes

    WEIGHTS_DIR = f"./weights/{model_name}/{generator}"
    RESULTS_DIR = f"./results/{model_name}/{generator}"

    if "_mod" in filename:
        generator += filename.split("_mod")[1].split("-")[0]
    output_string_filename = f"{generator}_seqlen_{seqlen}_step_{step}_num_bytes_{num_bytes}_train_ratio_{train_ratio}_test_ratio_{test_ratio}"
    weights_filename = f"{output_string_filename}.weights.h5"
    second_model_weights_filename = (
        f"{output_string_filename}_second_model.weights.h5"
    )
    weights_path = os.path.join(WEIGHTS_DIR, weights_filename)
    second_model_weights_path = os.path.join(WEIGHTS_DIR, second_model_weights_filename)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Calculate actual number of samples from the data
    total_train_bits = int(total_elements * train_ratio * 8)
    window_size = seqlen + target_bits
    train_sequences = (total_train_bits - window_size) // step + 1 if total_train_bits > window_size else 0
    
    total_test_bits = int(total_elements * test_ratio * 8)
    test_sequences = (total_test_bits - window_size) // step + 1 if total_test_bits > window_size else 0
    
    steps_per_epoch = train_sequences // batch_size
    validation_steps = test_sequences // batch_size

    return {
        "filename": filename,
        "seqlen": seqlen,
        "step": step,
        "num_bytes": num_bytes,
        "target_bits": target_bits,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "RESULTS_DIR": RESULTS_DIR,
        "output_string_filename": output_string_filename,
        "weights_path": weights_path,
        "second_model_weights_path": second_model_weights_path,
        "total_train_samples": train_sequences,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
    }


def log_model_parameters(model):
    total_parameters = model.count_params()
    trainable_parameters = np.sum(
        [tf.keras.backend.count_params(w) for w in model.trainable_weights]
    )
    non_trainable_parameters = np.sum(
        [tf.keras.backend.count_params(w) for w in model.non_trainable_weights]
    )

    print("-" * 40)
    print("Model Architecture")
    print("-" * 40)
    nice_log(f"Total parameters: {total_parameters}")
    nice_log(f"Trainable parameters: {trainable_parameters}")
    nice_log(f"Non-trainable parameters: {non_trainable_parameters}")

    return total_parameters, trainable_parameters, non_trainable_parameters


def check_test_to_classes_ratio(train_ratio, test_ratio, config, target_bits):
    test_sequences = (test_ratio / train_ratio) * config["total_train_samples"]
    test_to_classes_ratio = test_sequences / (2**target_bits)
    if test_to_classes_ratio < 1:
        raise ValueError("Test to classes ratio must be greater than 1")


if __name__ == "__main__":
    print("This is a module for auxiliary functions for the rcnn model.")
