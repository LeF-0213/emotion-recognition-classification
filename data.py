import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms

import config as C

_AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

def _looks_arabic(text: str) -> bool:
    return bool(_AR_RE.search(text or ""))

def translate_ar_to_en(texts, *, model_name, batch_size, max_chars, device=-1):
    from tqdm import tqdm
    from transformers import pipeline

    if isinstance(texts, (list, tuple)):
        texts = list(texts)
    else:
        raw = texts.tolist()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = -1 # CPU

    translator = pipeline("translation", model=model_name, tokenizer=model_name, device=device)

    def chunks(s: str):
        s = (s or "").strip()
        if not s:
            return []
        return [s[i : i + max_chars] for i in range(0, len(s), max_chars)]

    out = list(raw)
    idx_map = []
    flat = []
    for i, t in enumerate(raw):
        t = "" if t is None else str(t)
        if not _looks_arabic(t):
            continue
        ch = chunks(t)
        if not ch:
            continue
        idx_map.append((i, len(ch)))
        flat.extend(ch)

    if not flat:
        return out

    done = []
    for start in tqdm(range(0, len(flat), batch_size), desc="ar->en"):
        batch = flat[start : start + batch_size]
        res = translator(batch)
        done.extend([r["translation_text"].strip() for r in res])

    ptr = 0
    for i, n in idx_map:
        parts = done[ptr : ptr + n]
        ptr += n
        out[i] = " ".join(parts).strip()
    return out

def spectrogram_path(sample_id: str) -> str:
    sid = str(sample_id)
    p = os.path.join(C.SPEC_DIR, sid + ".jpg")
    if os.path.isfile(p):
        return p
    return os.path.join(C.SPEC_DIR, sid + ".jpg")

def load_and_merge_dataframe():
    texts_df = pd.read_csv(C.TEXT_CSV)
    labels_df = pd.read_csv(C.LABEL_CSV)
    df = pd.merge(labels_df, texts_df, on=C.COL_ID, how="inner")
    df[C.COL_EMOTION] = df[C.COL_EMOTION].astype(str).str.strip().str.lower()
    bad = sorted(set(df[C.COL_EMOTION]) - set(C.EMOTION_CLASSES))
    if bad:
        raise ValueError(f"Unknown labels: {bad}")
    df["label_idx"] = df[C.COL_EMOTION].map(C.LABEL2IDX)

    if C.TRANSLATE_AR_TO_EN:
        df[C.COL_TEXT] = translate_ar_to_en(
            df[C.COL_TEXT], 
            model_name=C.TRANSLATION_MODEL_NAME, 
            batch_size=C.TRANSLATION_BATCH_SIZE, 
            max_chars=C.TRANSLATION_MAX_CHARS,
        )

    df["spec_path"] = df[C.COL_ID].astype(str).map(spectrogram_path)
    df["has_spec"] = df["spec_path"].map(os.path.isfile)
    df["text_ok"] = df[C.COL_TEXT].fillna("").map(lambda s: len(str(s).strip()) > 0)
    df = df[df["has_spec"] & df["text_ok"]].reset_index(drop=True)
    return df

def stratified_splits(df: pd.DataFrame, seed: int):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label_idx"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label_idx"]
    )
    return train_df, val_df, test_df

def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.RandomResizedCrop(C.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])

    return train_tf, eval_tf

class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform, max_len):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        enc = self.tokenizer(
            str(row[C.COL_TEXT]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        image = Image.open(row["spec_path"]).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        label = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": image,
            "label": label,
        }

def class_weights_from_train(train_df: pd.DataFrame, device: torch.device):
    cnt = Counter(train_df["label_idx"].values)
    total = sum(cnt.values())
    w = torch.tensor([total / cnt[i] for i in range(C.NUM_CLASSES)], dtype=torch.float32, device=device)
    return w

def weighted_sampler(train_df: pd.DataFrame):
    cnt = Counter(train_df["label_idx"].values)
    inv = {c: 1.0 / n for c, n in cnt.items()}
    sample_w = [inv[int(l)] for l in train_df["label_idx"].values]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def make_loaders(tokenizer, train_df, val_df, test_df, device):
    train_tf, eval_tf = build_transforms()
    train_ds = EmotionDataset(train_df, tokenizer, train_tf, C.MAX_TEXT_LEN)
    val_ds = EmotionDataset(val_df, tokenizer, eval_tf, C.MAX_TEXT_LEN)
    test_ds = EmotionDataset(test_df, tokenizer, eval_tf, C.MAX_TEXT_LEN)
    sampler = weighted_sampler(train_df)
    pin = device.type in ("cuda", "mps")
    train_loader = DataLoader(
        train_ds, batch_size=C.BATCH_SIZE, sampler=sampler, num_workers=C.NUM_WORKERS, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS, pin_memory=pin
    )
    loss_weights = class_weights_from_train(train_df, device)
    return train_loader, val_loader, test_loader, loss_weights