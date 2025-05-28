import torch
from tqdm import tqdm


def evaluate(epoch, model, dataloader, device, criterion) -> None:
    model = model.to(device=device)
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # move data to device
            src_seq = batch["source_token"].to(device=device)
            tgt_seq = batch["target_token"].to(device=device)

            # create source mask
            src_mask = batch["source_mask"].unsqueeze(1)
            src_mask = src_mask.to(device=device, dtype=torch.bool) # (B, 1, S)
            # create target mask
            tgt_mask = batch["target_mask"].unsqueeze(1).to(dtype=torch.bool)
            csl_mask = torch.ones(tgt_seq.size(0), tgt_seq.size(1), tgt_seq.size(1))
            csl_mask = torch.tril(csl_mask).to(dtype=torch.bool)
            merged_mask = torch.logical_and(csl_mask, tgt_mask).to(device=device) # (B, S, S)

            label = batch["labels"].to(device=device) # (B, S)
            
            logits = model(src_seq, tgt_seq, src_mask, merged_mask) # (B, S, num_cls)

            flat_logits = logits.view(-1, logits.size(-1))
            flat_label = label.view(-1)
            loss = criterion(flat_logits, flat_label)

            eval_loss += loss.item()

    avg_eval_loss = eval_loss / len(dataloader)
    print(f"✅ >>> [Epoch {epoch}] eval loss: {avg_eval_loss:.3f} <<<")
