import sacrebleu


def main():
    ref_t = ["xin chào bạn, bạn khoẻ không và bạn từ đầu tới"]             

    pred_t = ["chào bạn, bạn tên gì", "xin chào bạn, bạn từ đầu tới"]                                                                                                                                     

    bleu = sacrebleu.corpus_bleu(pred_t, [ref_t], force=True, tokenize=None)
    print(bleu)


if __name__ == "__main__":
    main()
