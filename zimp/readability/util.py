def get_freq_score(texts, f_token_filter):
    cnt_filter, cnt_toks = 0, 0
    for txt in texts:
        filtered_toks, toks = f_token_filter(txt)
        cnt_filter += len(filtered_toks)
        cnt_toks += len(toks)

    return cnt_filter/cnt_toks