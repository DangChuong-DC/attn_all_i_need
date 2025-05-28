import torch
from transformers import AutoTokenizer
import pandas as pd
import math


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")

tokenizer.add_special_tokens({"bos_token": "<s>"})
# print(len(tokenizer))

# Get the vocabulary pool
# vocab = tokenizer.get_vocab()
# print(vocab)

# Get the vocabulary size
# vacab_size = tokenizer.vocab_size
# print(vacab_size)

# If using separated vocabulary pool
# is_separate_vocab = tokenizer.separate_vocabs
# print(f"If separating source & target vocab: {is_separate_vocab}")

DATA_PATH = "/DATA01/dc/datasets/iwslt2015_en_vi/train/train0000.parquet"

df_data = pd.read_parquet(DATA_PATH)

# longest_len = 0

# for i in range(len(df_data)):
#     row = df_data.iloc[i]["translation"]
#     model_input = tokenizer(
#         text=row["en"],
#         text_target=row["vi"],
#         padding=False,
#         truncation=False,
#         return_tensors="pt",
#     )
#     longest_len = max(longest_len, model_input["input_ids"].size(-1))
#     longest_len = max(longest_len, model_input["labels"].size(-1))

# print(f"Longest length: {longest_len}")

# row = df_data.iloc[13]
# data = row["translation"]
# en_txt = data["en"]
# vi_txt = data["vi"]

# model_input = tokenizer(
#     text=en_txt,
#     text_target=vi_txt,
#     padding="max_length",
#     truncation=True,
#     max_length=64,
#     return_tensors="pt",
# )

# print(model_input)

test_key_pad_mask = torch.ones((1, 9))
for i in range(6, 9):
    test_key_pad_mask[:, i] = 0

print(test_key_pad_mask)

test_key_pad_mask0 = test_key_pad_mask.unsqueeze(1).to(dtype=torch.bool)
test_attn_bias = torch.zeros((1, 9, 9), dtype=torch.float)
test_attn_bias = test_attn_bias.masked_fill_(test_key_pad_mask0.logical_not(), -math.inf)
print(test_attn_bias)


# test_pad_mask = torch.matmul(test_pad_mask.transpose(-2, -1), test_pad_mask)
# test_att_mask = torch.ones((1, 9, 9))
# test_att_mask = torch.tril(test_att_mask)

# test_key_pad_mask = test_key_pad_mask.to(dtype=torch.bool)
# test_att_mask = test_att_mask.to(dtype=torch.bool)

# print(test_key_pad_mask.shape)
# print(test_key_pad_mask)

# print(test_att_mask.size())
# print(test_att_mask)

# print()
# merged_mask = torch.logical_and(test_att_mask, test_key_pad_mask.unsqueeze(1))
# print(merged_mask)

# test_attn_mask = torch.ones((9, 9))
# test_attn_mask = torch.tril(test_attn_mask)
# test_attn_mask = test_attn_mask.to(dtype=torch.bool).unsqueeze(0)
# print(test_attn_mask.shape)
# print(test_attn_mask)

# test_merged_mask = test_attn_mask + test_pad_mask
# print(test_merged_mask.shape)
# print(test_merged_mask)
