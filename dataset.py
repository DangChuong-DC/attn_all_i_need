from typing import Any, Sequence
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import Dataset

DATA_PATH = "/DATA01/dc/datasets/iwslt2015_en_vi"


class CustomizedEnViIWSLT(Dataset):
    def __init__(
        self, split: str = "train", max_seq_len: int = 128,
    ) -> None:
        super(CustomizedEnViIWSLT, self).__init__()
        self.df_data = pd.read_parquet(Path(DATA_PATH) / f"{split}" / f"{split}0000.parquet")

        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
        no_token_added = self.tokenizer.add_special_tokens({"bos_token": "<s>"})
        self.new_vocab_size = self.tokenizer.vocab_size + no_token_added

        self.max_seq_len = max_seq_len
        self.bos_token_id = self.new_vocab_size - 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df_data.iloc[index]
        data = row.translation
        tok_output = self.tokenizer(
            text=data["en"],
            text_target=data["vi"],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        output_dict = {}
        output_dict["source_token"] = tok_output["input_ids"] # (1, max_length)
        output_dict["source_mask"] = tok_output["attention_mask"] # (1, max_length)
        
        # prepare target token by shifting right one token
        target_token = tok_output["labels"].clone()
        target_token[:, 0] = self.bos_token_id
        target_token[:, 1:] = tok_output["labels"][:, :-1]
        output_dict["target_token"] = target_token
        # prepare target mask
        target_mask = torch.ones(*target_token.size())
        target_mask[target_token == self.tokenizer.pad_token_id] = 0
        target_mask = target_mask.to(dtype=tok_output["attention_mask"].dtype)
        output_dict["target_mask"] = target_mask

        labels = tok_output["labels"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        output_dict["labels"] = labels
        return output_dict
    
    def __len__(self) -> int:
        return len(self.df_data)

    def get_vocab_size(self) -> int:
        return self.new_vocab_size

    def get_max_seq_len(self) -> int:
        return self.max_seq_len
    
    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer


def collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    src_tok = torch.cat([exp["source_token"] for exp in batch], dim=0)
    tgt_tok = torch.cat([exp["target_token"] for exp in batch], dim=0)
    src_mask = torch.cat([exp["source_mask"] for exp in batch], dim=0)
    tgt_mask = torch.cat([exp["target_mask"] for exp in batch], dim=0)
    labels = torch.cat([exp["labels"] for exp in batch], dim=0)
    return {
        "source_token": src_tok,
        "target_token": tgt_tok,
        "source_mask": src_mask,
        "target_mask": tgt_mask,
        "labels": labels,
    }


if __name__ == "__main__":
    dset = CustomizedEnViIWSLT(split="train")

    test_out = dset.__getitem__(9)

    print(test_out)
    print(test_out["source_token"].dtype)
    print(test_out["source_mask"].dtype)
    print(test_out["target_token"].dtype)
    print(test_out["target_mask"].dtype)
    print(test_out["labels"].dtype)
