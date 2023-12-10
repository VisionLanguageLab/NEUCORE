import json
import os
import os.path as osp

import spacy
from tqdm import tqdm

SPLITS = ("train", "val", "test1")
DATA_DIR = "/data/dataset/Retrieval/cirr/captions"
OUT_DIR = "/data/dataset/Retrieval/cirr/concepts_captions"
os.makedirs(OUT_DIR, exist_ok=True)

nlp = spacy.load("en_core_web_sm")
for split in SPLITS:
    res = []
    filename = f"cap.rc2.{split}.json"
    with open(osp.join(DATA_DIR, filename)) as f:
        json_data = json.load(f)
    for item in tqdm(json_data, desc=f"Extracting concepts from {split}"):
        cap = item["caption"]
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
    with open(osp.join(OUT_DIR, f"cap.rc2.{split}.json"), "w") as f:
        json.dump(res, f)