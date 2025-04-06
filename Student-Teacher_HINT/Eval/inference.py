#!/usr/bin/env python3
import os
import sys
import csv
import glob
import pickle
import torch
torch.manual_seed(0)
sys.path.append('..')
import numpy as np
from tqdm import tqdm

# Minimal imports from your codebase:
from HINT.dataloader import csv_three_feature_2_dataloader
from HINT.molecule_encode import MPNN
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding
from HINT.model import Interaction, HINTModel  # or HINTModel if your checkpoint is that class

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score

DEVICE = torch.device("cpu")
BATCH_SIZE = 32

##############################################################################
# 1) Decide which test CSV to use based on checkpoint name
##############################################################################
def pick_test_csv_from_ckpt(ckpt_name):
    """
    If checkpoint starts with 'all_' or 'cos_all_', we use all_test.csv.
    If checkpoint starts with 'hint_' or 'cos_hint_', we use hint_test.csv.
    Otherwise default to all_test.csv.
    """
    ckpt_name = ckpt_name.lower()
    if ckpt_name.startswith("all_") or ckpt_name.startswith("cos_all_"):
        return os.path.join("..", "data", "all_test.csv")
    elif ckpt_name.startswith("hint_") or ckpt_name.startswith("cos_hint_"):
        return os.path.join("..", "data", "hint_test.csv")
    else:
        print(f"Warning: no test CSV found for {ckpt_name}")
        return None

##############################################################################
# 2) Build a test_loader from a CSV
##############################################################################
def make_test_loader(csv_file, batch_size=BATCH_SIZE):
    """
    Creates a test loader from your standard function.
    """
    return csv_three_feature_2_dataloader(csv_file, shuffle=False, batch_size=batch_size)

##############################################################################
# 3) Generate predictions and labels
##############################################################################
def generate_predictions(model, test_loader, device=DEVICE):
    """
    1) Set model to eval mode.
    2) For each batch in test_loader:
         - forward pass → predicted probabilities,
         - gather predictions and true labels.
    3) Return arrays of predictions and ground-truth labels.
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in test_loader:
            label_vec = label_vec.to(device)
            outputs = model.forward(smiles_lst2, icdcode_lst3, criteria_lst).view(-1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.extend(probs)
            labels.extend(label_vec.cpu().numpy())
    model.train()
    return np.array(preds), np.array(labels)

##############################################################################
# 4) Re-implement bootstrap_test
##############################################################################
def bootstrap_test(model, test_loader, device=DEVICE, sample_num=20):
    preds, labels = generate_predictions(model, test_loader, device=device)
    preds = preds.flatten()
    labels = labels.flatten()

    def compute_metrics(y_pred, y_true, threshold=0.5):
        y_bin = (y_pred >= threshold).astype(int)
        roc_auc_val = roc_auc_score(y_true, y_pred)
        f1_val      = f1_score(y_true, y_bin)
        pr_auc_val  = average_precision_score(y_true, y_pred)
        precision_val = precision_score(y_true, y_bin)
        accuracy_val  = accuracy_score(y_true, y_bin)
        return (roc_auc_val, f1_val, pr_auc_val, precision_val, accuracy_val)

    N = len(preds)
    replicate_metrics = []
    np.random.seed(0)
    for _ in range(sample_num):
        idx = np.random.choice(N, size=N, replace=True)
        y_pred_bs = preds[idx]
        y_true_bs = labels[idx]
        replicate_metrics.append(compute_metrics(y_pred_bs, y_true_bs))

    replicate_metrics = np.array(replicate_metrics)  # shape = (sample_num, 5)
    means = replicate_metrics.mean(axis=0)
    stds  = replicate_metrics.std(axis=0)

    stats = {
        "roc_auc_mean":    means[0], "roc_auc_std":    stds[0],
        "f1_mean":         means[1], "f1_std":         stds[1],
        "pr_auc_mean":     means[2], "pr_auc_std":     stds[2],
        "precision_mean":  means[3], "precision_std":  stds[3],
        "accuracy_mean":   means[4], "accuracy_std":   stds[4]
    }
    return stats, (preds, labels)

##############################################################################
# 5) Main script: loop subdirectories, load .ckpt, run inference, store stats
##############################################################################
def main():
    SUBDIRS = ["mse_all", "mse_hint", "cos_all", "cos_hint"]
    BASE_SAVE_MODEL = os.path.join("..", "save_model")
    out_csv = os.path.join("..", "results", "inference_results.csv")

    header = [
        "subdir",
        "checkpoint",
        "roc_auc_mean", "roc_auc_std",
        "f1_mean", "f1_std",
        "pr_auc_mean", "pr_auc_std",
        "precision_mean", "precision_std",
        "accuracy_mean", "accuracy_std"
    ]
    rows = []

    # Outer loop over subdirectories with a tqdm progress bar
    for subdir in tqdm(SUBDIRS, desc="Subdirectories"):
        folder = os.path.join(BASE_SAVE_MODEL, subdir)
        if not os.path.isdir(folder):
            print(f"Warning: subdir {folder} not found, skipping.")
            continue

        ckpt_files = glob.glob(os.path.join(folder, "*.ckpt"))
        ckpt_files.sort()
        if not ckpt_files:
            print(f"No .ckpt files found in {folder}, skipping.")
            continue

        # Inner loop over checkpoint files with tqdm progress bar
        for ckpt_path in tqdm(ckpt_files, desc=f"Processing {subdir}", leave=False):
            ckpt_name = os.path.basename(ckpt_path)
            print(f"\n[INFO] Now processing: {subdir} / {ckpt_name}")

            try:
                model = torch.load(ckpt_path, map_location=DEVICE)
                model = model.to(DEVICE)
            except Exception as e:
                print(f"Error loading {ckpt_name}: {e}")
                continue

            test_csv = pick_test_csv_from_ckpt(ckpt_name)
            if test_csv is None or not os.path.exists(test_csv):
                print(f"Test CSV {test_csv} not found; skipping {ckpt_name}.")
                continue

            test_loader = make_test_loader(csv_file=test_csv, batch_size=BATCH_SIZE)
            stats, (preds, labels) = bootstrap_test(model, test_loader, device=DEVICE, sample_num=20)

            # Save pickle file in the corresponding results subdirectory (under results/)
            pickle_subdir = os.path.join("..", "results", subdir)
            os.makedirs(pickle_subdir, exist_ok=True)
            pkl_name = os.path.splitext(ckpt_name)[0] + ".pkl"
            pkl_path = os.path.join(pickle_subdir, pkl_name)
            with open(pkl_path, "wb") as pf:
                pickle.dump({"preds": preds, "labels": labels}, pf)

            print("Stats for this checkpoint:")
            for k, v in stats.items():
                print(f"  {k}: {v:.4f}")

            row = [
                subdir,
                ckpt_name,
                stats["roc_auc_mean"],  stats["roc_auc_std"],
                stats["f1_mean"],       stats["f1_std"],
                stats["pr_auc_mean"],   stats["pr_auc_std"],
                stats["precision_mean"],stats["precision_std"],
                stats["accuracy_mean"], stats["accuracy_std"]
            ]
            rows.append(row)
        # Save CSV after finishing each subdirectory (overwriting previous save)
        with open(out_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
        print(f"[INFO] CSV updated with results from subdirectory {subdir}: {out_csv}")

    print(f"\nInference complete. Final results saved to {out_csv}.\n")

if __name__ == "__main__":
    main()
