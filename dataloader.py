

def nli_triplet_collate_fn(batch,tokenizer):
    batch_size = len(batch)
    anchors = [b["anchor"] for b in batch]
    positives = [b["positive"] for b in batch]
    negatives = [b["negative"] for b in batch]

    # Combine into one list
    all_sentences = anchors + positives + negatives
    # Tokenize in one pass
    enc_all = tokenizer(all_sentences, padding=True, truncation=True, return_tensors="pt")
    # Then split back again
    enc_anchor = {k: v[:batch_size] for k, v in enc_all.items()}
    enc_pos = {k: v[batch_size:2*batch_size] for k, v in enc_all.items()}
    enc_neg = {k: v[2*batch_size:] for k, v in enc_all.items()}


    # each is dict of of 'input_ids', 'token_type_ids', 'attention_mask' which are tensors
    return enc_anchor, enc_pos, enc_neg
