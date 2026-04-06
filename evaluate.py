import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

import config as C

def run_test(model, test_loader, loss_weights, device):
    from train import evaluate

    vl, va, preds, labels = evaluate(model, test_loader, loss_weights, device)
    f1m = f1_score(labels, preds, average="macro")
    f1w = f1_score(labels, preds, average="weighted")
    report = classification_report(labels, preds, target_names=C.EMOTION_CLASSES, digits=4)
    
    return {
        "loss": vl,
        "accuracy": va,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "preds": preds,
        "labels": labels,
        "report": report,
    }

def metrics_from_preds(labels, preds):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
    }
