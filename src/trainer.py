import torch
from tqdm import tqdm


def run_pipeline(model, dataloader, tokenizer, device, mode="train", optimizer=None):
    if mode == "train":
        model.train()
    else:
        model.eval()

    predictions = []
    total_loss = 0

    # Progress bar
    pbar = tqdm(dataloader, desc=mode.capitalize())

    for batch in pbar:
        # Move data to GPU/CPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", None)

        if labels is not None:
            labels = labels.to(device)

        if mode == "train":
            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        else:
            with torch.no_grad():
                logits, _ = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

    if mode == "test":
        return predictions

    return total_loss / len(dataloader)
