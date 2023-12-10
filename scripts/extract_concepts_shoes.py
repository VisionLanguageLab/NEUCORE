import json
import os
import os.path as osp

import spacy
from tqdm import tqdm

DATA_DIR = "/data/dataset/Retrieval/shoes/annotations"
OUT_DIR = "/data/dataset/Retrieval/shoes/concepts_annotations"
os.makedirs(OUT_DIR, exist_ok=True)

nlp = spacy.load("en_core_web_sm")
for filename in os.listdir(DATA_DIR):
    if not filename.startswith("triplet"):
        continue
    with open(osp.join(DATA_DIR, filename), "r") as fi:
        json_data = json.load(fi)
    res = []
    for item in tqdm(json_data, desc=f"Extracting concepts from {filename}"):
        cap = item["RelativeCaption"]
        nouns = set()
        adjs = set()
        advs = set()
        verbs = set()
        for token in nlp(cap):
            if token.pos_ == "NOUN":
                nouns.add(token.lemma_)
            elif token.pos_ == "ADJ":
                adjs.add(token.lemma_)
            elif token.pos_ == "ADV":
                advs.add(token.lemma_)
            elif token.pos_ == "VERB":
                verbs.add(token.lemma_)
        item["noun"] = list(nouns)
        item["adj"] = list(adjs)
        item["adv"] = list(advs)
        item["verb"] = list(verbs)
        res.append(item)

    with open(osp.join(OUT_DIR, filename), "w") as fo:
        json.dump(res, fo)
