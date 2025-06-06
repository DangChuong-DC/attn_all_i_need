from tqdm import tqdm

import torch
import sacrebleu
from transformers import AutoTokenizer


def post_process_sentences(sequences: list[str], end_token: str) -> list[str]:
    for i in range(len(sequences)):
        sequences[i] = sequences[i].split(end_token)[0]
    return sequences


def evaluate(
    epoch: int, 
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: str, 
    criterion: torch.nn.Module,
    tokenizer: AutoTokenizer,
) -> None:
    model = model.to(device=device)
    model.eval()
    eval_loss = 0.0

    all_hyps = []
    all_refs = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
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

            # decode token ids to token for calculating BLEU score later
            token_ids = logits.argmax(dim=-1) # (B, S)

            pred_seqs = tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            pred_seqs = post_process_sentences(pred_seqs, tokenizer.eos_token)
            # print(f"⬇⬇⬇ predicted sequences ⬇⬇⬇")
            # for s in pred_seqs:
            #     print(s)
            # print()

            gt_seqs = tokenizer.batch_decode(
                label,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            gt_seqs = post_process_sentences(gt_seqs, tokenizer.eos_token)
            # print(f"⬇⬇⬇ groundtruth sequences ⬇⬇⬇")
            # for s in gt_seqs:
            #     print(s)

            all_hyps.extend(pred_seqs)
            all_refs.extend(gt_seqs)
            # exit()
        
    bleu_score = sacrebleu.corpus_bleu(all_hyps, [all_refs])

    avg_eval_loss = eval_loss / len(dataloader)
    print(f"✅ >>> [Epoch {epoch}] eval loss: {avg_eval_loss:.3f} <<<")
    print(f"🔹 BLEU score = {bleu_score}")
