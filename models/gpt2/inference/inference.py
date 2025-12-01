import torch


def binary_inference(model, x, y, loss_fn, correct, total):
    logits = model(x).logits
    probs = torch.softmax(logits, dim=-1)
    predicted = torch.argmax(probs, dim=-1)
    
    # Flatten for CrossEntropyLoss
    loss = loss_fn(logits.view(-1, 2), y.view(-1))
    
    correct += (predicted == y).sum().item()
    total += y.numel()

    # Predictions are already indices (0 or 1)
    binary_predictions = predicted.cpu()
    return binary_predictions, loss, correct, total


def multitoken_inference(
    model, x, y, target_bits, loss_fn, correct, total, config, tokenizer
):
    # model(x).logits has shape [batch_size, sequence_length, 2**target_bits]
    logits = model(x).logits

    # y is already indices
    target = y.view(-1)

    # Flatten logits and target for loss calculation
    flattened_logits = logits.view(-1, 2**target_bits)
    loss = loss_fn(flattened_logits, target)

    if config["evaluate_all_bits"]:
        predicted = flattened_logits.argmax(dim=1)
        correct += (predicted == target).sum().item()
        total += target.size(0)
        binary_predictions = tokenizer.detokenize(predicted.cpu().tolist())
    else:
        # Evaluate only the last token's logits for each batch item
        last_logits = logits[:, -1, :]  # Shape: [batch_size, 2**target_bits]

        last_predicted = last_logits.argmax(dim=1)  # Predictions for the last token

        last_target = y[:, -1]  # Get the target index for the last token

        correct += (last_predicted == last_target).sum().item()
        total += x.size(0)  # Total number of examples in the batch

        binary_predictions = tokenizer.detokenize(last_predicted.cpu().tolist())

    binary_predictions = torch.tensor(
        [int(bit) for bit in binary_predictions], dtype=torch.int32
    )

    return binary_predictions, loss, correct, total


def eval_multi(predictions, target, target_bits, correct, total, config):
    # predictions: indices [B, S]
    # target: indices [B, S]
    
    if config["evaluate_all_bits"]:
        correct += (predictions == target).sum().item()
        total += target.numel()
    else:
        correct += (predictions[:, -1] == target[:, -1]).sum().item()
        total += target.size(0)
        
    return correct, total


def autoregressive_inference(
    model, x, y, target_bits, loss_fn, correct, total, config, device
):
    target = y
    x_current = x.to(device)
    predictions_sequence = []

    for step in range(target_bits):
        logits = model(x_current).logits
        probs = torch.softmax(logits, dim=-1)
        predicted_bits = torch.argmax(probs, dim=-1)

        # Remove the first element from x_current and append the new predicted element
        x_current = torch.cat([x_current[:, 1:], predicted_bits[:, -1:]], dim=1)

        # Expand dimensions of predicted_bits to make it compatible for concatenation
        predicted_bits_expanded = predicted_bits.unsqueeze(-1)

        predictions_sequence.append(predicted_bits_expanded)

    concatenated_bits = torch.cat(predictions_sequence, dim=-1)
    # we need to one-hot encode the predictions
    # Converting each sequence of bits to a decimal number. We multiply each bit by its corresponding power of 2
    decimal_predictions = torch.sum(
        concatenated_bits
        * 2
        ** torch.arange(target_bits - 1, -1, -1, device=device)
        .unsqueeze(0)
        .unsqueeze(0),
        dim=-1,
    )
    # Converting each decimal number to a one-hot encoded tensor
    predictions = torch.nn.functional.one_hot(
        decimal_predictions, num_classes=2**target_bits
    ).to(torch.float)
    
    # Since we have hard predictions (0 or 1), we can't calculate a true Cross Entropy Loss (which requires logits/probabilities).
    # However, to maintain interface consistency, we return a dummy loss or a proxy.
    # Here we calculate the accuracy as a proxy for loss (1 - accuracy) or just return 0.0 since we are in inference.
    # But to be safe and consistent with other functions returning a loss tensor:
    loss = torch.tensor(0.0, device=device)

    correct, total = eval_multi(
        decimal_predictions, target, target_bits, correct, total, config
    )

    binary_predictions = concatenated_bits.view(-1).cpu()

    return binary_predictions, loss, correct, total


if __name__ == "__main__":
    print(
        "This is a module with the implementation of the inference mechanisms for the GPT-2 model."
    )
