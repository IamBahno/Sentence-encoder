import torch

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

def nli_pair_class_collate_fn(batch, tokenizer):
    batch_size = len(batch)
    premise = [b["premise"] for b in batch]
    hypothesis = [b["hypothesis"] for b in batch]
    target = [b["label"] for b in batch]

    # Combine into one list
    all_sentences = premise + hypothesis
    # Tokenize in one pass
    enc_all = tokenizer(all_sentences, padding=True, truncation=True, return_tensors="pt")
    # Then split back again
    enc_premise = {k: v[:batch_size] for k, v in enc_all.items()}
    enc_hypothesis = {k: v[batch_size:] for k, v in enc_all.items()}

    # each is dict of 'input_ids', 'token_type_ids', 'attention_mask' which are tensors, target is int (0/1/2)
    return enc_premise, enc_hypothesis, torch.tensor(target, dtype=torch.long)

def mnr_collate_fn(batch, tokenizer):
    batch_size = len(batch)
    anchor = [b["premise"] for b in batch]
    anchor_candidates = [b["hypothesis"] for b in batch]

    # Combine into one list
    all_sentences = anchor + anchor_candidates
    # Tokenize in one pass
    enc_all = tokenizer(all_sentences, padding=True, truncation=True, return_tensors="pt")
    # Then split back again
    enc_anchor = {k: v[:batch_size] for k, v in enc_all.items()}
    enc_anchor_candidates = {k: v[batch_size:] for k, v in enc_all.items()}

    return enc_anchor, enc_anchor_candidates