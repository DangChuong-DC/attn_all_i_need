import os

from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from dataset import CustomizedEnViIWSLT, collate_fn
from model.transformer import Transformer
from eval import evaluate


def train(
    epoch: int, 
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: str, 
    criterion: nn.Module, 
    lr_scheduler: LRScheduler
) -> None:
    model = model.to(device=device)
    model.train()
    train_loss = 0.0

    for i, batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()

        # move data to device
        src_seq = batch["source_token"].to(device=device)
        tgt_seq = batch["target_token"].to(device=device)

        # create source mask
        src_mask = batch["source_mask"].unsqueeze(1) # (B, 1, S_s)
        src_mask = src_mask.to(dtype=torch.bool, device=device)
        # create target mask
        tgt_mask = batch["target_mask"].unsqueeze(1).to(dtype=torch.bool) # (B, 1, S_t)
        csl_mask = torch.ones((tgt_seq.size(0), tgt_seq.size(1), tgt_seq.size(1))) # (B, S_t, S_t)
        csl_mask = torch.tril(csl_mask).to(dtype=torch.bool)
        merged_mask = torch.logical_and(csl_mask, tgt_mask).to(device=device)

        labels = batch["labels"].to(device=device) # (B, S_t)
        logits = model(src_seq, tgt_seq, src_mask, merged_mask) # (B, S_t, num_class)
        assert torch.isfinite(logits).all(), "NaN in logits!"

        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        loss = criterion(flat_logits, flat_labels)
        assert torch.isfinite(loss).all(), "loss became NaN!"
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()

        # if (i + 1) % 100 == 0:
        #     print(f"Iter {i} --- train loss: {loss.item()}")
    
    avg_train_loss = train_loss / len(dataloader)
    print(f"📝 > [Epoch {epoch}] train loss: {avg_train_loss:.3f}")


def main() -> None:
    # ——— Hyperparameters ———
    batch_size      = 128
    epochs          = 13
    learning_rate   = 1e-3
    max_seq_len     = 128
    # warmup_steps    = 500

    model_name = "tiny_transformer_250528"
    saving_model_dir = os.getenv("MODEL_CHECKPOINT_DIR")
    if not saving_model_dir:
        raise RuntimeError(
            "Please specify where to save model by export environmental variable `MODEL_CHECKPOINT_DIR`"
        )

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # ——— Model Config ———
    num_layer = 3
    num_attn_head = 8
    model_dim = 128
    feedforward_dim = 128*4
    dropout_rate = 0.1

    # ——— Data ———
    train_ds = CustomizedEnViIWSLT(split="train", max_seq_len=max_seq_len)
    val_ds   = CustomizedEnViIWSLT(split="val", max_seq_len=max_seq_len)


    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )

    # ——— Model ———
    de_transformer = Transformer(
        train_ds.get_vocab_size(),
        max_seq_len,
        n_layer=num_layer,
        n_head=num_attn_head,
        d_model=model_dim,
        d_ff=feedforward_dim,
        dropout_p=dropout_rate,
    )

    # ——— Optimizer & Scheduler & Criterion ———
    optimizer = torch.optim.Adam(
        de_transformer.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
    )
    total_steps = epochs * len(train_loader)
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps, 1e-9)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    for e in range(epochs):
        train(e, de_transformer, train_loader, optimizer, device, criterion, lr_scheduler)
        # saving the model after every epoch
        saving_model_path = Path(saving_model_dir) / (model_name + ".pt")
        torch.save(de_transformer.state_dict(), saving_model_path)
        
        evaluate(e, de_transformer, val_loader, device, criterion)


if __name__ == "__main__":
    main()
