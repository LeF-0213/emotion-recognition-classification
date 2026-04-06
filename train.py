import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import config as C

def mixup_batch(images, labels, alpha: float, device):
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam

def train_one_epoch(model, loader, optimizer, loss_weights, scaler, scheduler, device, use_mixup: bool):
    model.train()
    crit = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=C.LABEL_SMOOTHING)
    amp_enabled = device.type == "cuda"
    device_type = "cuda" if amp_enabled else "cpu"
    total_loss = total_correct = total_n = 0

    for batch in tqdm(loader, leave=False, desc="train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        labels = batch["label"].to(device)

        if use_mixup:
            image, y_a, y_b, lam = mixup_batch(image, labels, C.MIXUP_ALPHA, device)
        else:
            y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            logits = model(input_ids, attention_mask, image)
            if use_mixup and lam < 1.0:
                loss = lam * crit(logits, y_a) + (1.0 - lam) * crit(logits, y_b)
            else:
                loss = crit(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        pred = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (pred == labels).sum().item()
        total_n += labels.size(0)

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)

@torch.no_grad()
def evaluate(model, loader, loss_weights, device):
    model.eval()
    crit = nn.CrossEntropyLoss(weight=loss_weights)
    amp_enabled = device.type == "cuda"
    device_type = "cuda" if amp_enabled else "cpu"
    total_loss = total_correct = total_n = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, leave=False, desc="eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            logits = model(input_ids, attention_mask, image)
            loss = crit(logits, labels)

        pred = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (pred == labels).sum().item()
        total_n += labels.size(0)
        all_preds.extend(pred.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1), all_preds, all_labels

def optimizer_groups(model: nn.Module):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm")
    decay, nodecay = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            nodecay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": C.WEIGHT_DECAY},
        {"params": nodecay, "weight_decay": 0.0},
    ]

def train_model(model, train_loader, val_loader, loss_weights, device, tag: str, use_mixup: bool):
    opt = AdamW(optimizer_groups(model), lr=C.LR)
    steps = len(train_loader) * C.NUM_EPOCHS
    sched = OneCycleLR(opt, max_lr=C.LR, total_steps=steps, pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0
    patience_left = C.EARLY_STOP_PATIENCE
    best_path = C.OUTPUT_DIR / f"best_{tag}.pt"

    for epoch in range(1, C.NUM_EPOCHS + 1):
        tl, ta = train_one_epoch(
            model, train_loader, opt, loss_weights, scaler, sched, device, use_mixup
        )
        vl, va, _, _ = evaluate(model, val_loader, loss_weights, device)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(f"[{tag}] ep {epoch}/{C.NUM_EPOCHS} train {tl:.4f}/{ta:.4f} val {vl:.4f}/{va:.4f}")

        if va > best_val:
            best_val = va
            patience_left = C.EARLY_STOP_PATIENCE
            C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{tag}] early stop at {epoch}")
                break

    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device))

    return history, best_val, best_path

@torch.no_grad()
def ensemble_predict(models, loader, device):
    for m in models:
        m.eval()
    probs_all, y_all = [], []
    amp_enabled = device.type == "cuda"
    device_type = "cuda" if amp_enabled else "cpu"

    for batch in tqdm(loader, leave=False, desc="ensemble"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        labels = batch["label"].to(device)
        acc = None
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            for m in models:
                p = F.softmax(m(input_ids, attention_mask, image), dim=-1)
                acc = p if acc is None else acc + p

        acc = acc / len(models)
        probs_all.append(acc.cpu().numpy())
        y_all.append(labels.cpu().numpy())
        
    probs = np.concatenate(probs_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    return probs.argmax(axis=1), y
