import torch
from model.transformer import Transformer


if __name__ == "__main__":
    vocab_size = 999
    max_seq_len = 32

    transformer = Transformer(
        vocab_size,
        max_seq_len,
        n_layer=1,
        n_head=8,
        d_model=256,
        d_ff=256 * 2,
        dropout_p=0.0
    )
    src_ids = torch.randint(0, vocab_size, size=(4, 32))
    tgt_ids = torch.randint(0, vocab_size, size=(4, 32))

    test_out = transformer(src_ids, tgt_ids)
    print(test_out.size())
