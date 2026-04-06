import argparse
import random

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer

import config as C
from data import load_and_merge_dataframe, make_loaders, stratified_splits
from evaluate import metrics_from_preds, run_test
from models import build_model
from plots import plot_confusion_matrices, plot_learning_curves, save_results_table
from train import ensemble_predict, train_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_merge_dataframe()
    train_df, val_df, test_df = stratified_splits(df, C.SEED)
    tokenizer = BertTokenizer.from_pretrained(C.TEXT_MODEL_NAME)
    train_loader, val_loader, test_loader, loss_weights = make_loaders(
        tokenizer, train_df, val_df, test_df, device
    )

    tags = [("ConcatFusion", "concat"), ("CrossModalFusion", "cross")]
    histories, names = [], []
    models = []
    test_results = {}

    for tag, key in tags:
        model = build_model(key).to(device)
        if not args.skip_train:
            hist, _, _ = train_model(
                model, train_loader, val_loader, loss_weights, device, tag, args.mixup
            )
            histories.append(hist)
            names.append(tag)
        ckpt = C.OUTPUT_DIR / f"best_{tag}.pt"
        if ckpt.is_file():
            model.load_state_dict(torch.load(ckpt, map_location=device))
        models.append(model)
        test_results[tag] = run_test(model, test_loader, loss_weights, device)
        print(test_results[tag]["report"])

    if histories:
        plot_learning_curves(histories, names, C.FIG_DIR / "learning_curves.png")

    rows = []
    for tag, r in test_results.items():
        fusion = "Concat" if "Concat" in tag else "2-token self-attn"
        rows.append(
            {
                "model": tag,
                "fusion": fusion,
                "test_accuracy": f"{r['accuracy']:.4f}",
                "f1_macro": f"{r['f1_macro']:.4f}",
                "f1_weighted": f"{r['f1_weighted']:.4f}",
            }
        )

    ens_pred, ens_y = ensemble_predict(models, test_loader, device)
    ens_m = metrics_from_preds(ens_y.tolist(), ens_pred.tolist())
    rows.append(
        {
            "model": "Ensemble",
            "fusion": "soft vote",
            "test_accuracy": f"{ens_m['accuracy']:.4f}",
            "f1_macro": f"{ens_m['f1_macro']:.4f}",
            "f1_weighted": f"{ens_m['f1_weighted']:.4f}",
        }
    )
    save_results_table(rows, C.OUTPUT_DIR / "results.csv")
    print(pd.DataFrame(rows).to_string(index=False))

    cm_dict = {k: {"labels": v["labels"], "preds": v["preds"]} for k, v in test_results.items()}
    plot_confusion_matrices(cm_dict, C.FIG_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()