"""
Step 1: Generate Calibration Dataset
=====================================
Iterates over predefined augmentation configs (N, P, A) and computes Energy.

Output:
  - {model}_{test_split}.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from scipy import stats

from datasets.load_dataset import load_dataset
from models.get_model import get_model
from models.tools import set_seed, select_device, energy_from_logits, normalize_iq

from models.augmentation import (
    get_all_configs, create_augmentor, CombinedAugmentor,
    print_statistics, print_param_table,
)


class AugDataset(Dataset):
    def __init__(self, x_np, y_np, augmentor: CombinedAugmentor = None):
        self.x = x_np
        self.y = y_np.astype(np.int64)
        self.augmentor = augmentor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx].copy()
        if self.augmentor is not None:
            x = self.augmentor(x)
        x = normalize_iq(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(int(self.y[idx]))


def evaluate(model, x_data, y_data, device, augmentor=None, T=1.0, batch_size=128):
    model.eval()
    ds = AugDataset(x_data, y_data, augmentor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    correct, total = 0, 0
    energies = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            logits = out[1] if isinstance(out, (tuple, list)) else out
            energies.append(energy_from_logits(logits, T=T).cpu().numpy())
            correct += logits.argmax(1).eq(target).sum().item()
            total += len(target)

    energies = np.concatenate(energies)
    return {
        "Accuracy": 100.0 * correct / total,
        "Avg_Energy": float(energies.mean()),
    }


def generate_dataset(model, x_data, y_data, device, configs, T=1.0, batch_size=128):
    results = []
    total = len(configs)
    print(f"\nIterating over {total} configs...")

    for i, cfg in enumerate(configs):
        if cfg.total_severity == 0:
            augmentor = None
            name = "Clean"
        else:
            augmentor = create_augmentor(cfg)
            name = cfg.name

        result = evaluate(model, x_data, y_data, device, augmentor, T, batch_size)

        results.append({
            "Config": name,
            "AWGN_sev": cfg.awgn_sev,
            "Phase_sev": cfg.phase_sev,
            "PA_sev": cfg.pa_sev,
            "Total_sev": cfg.total_severity,
            "Accuracy": result["Accuracy"],
            "Avg_Energy": result["Avg_Energy"],
        })

        print(f"  [{i+1:3d}/{total}] {name:15s}: Acc={result['Accuracy']:.2f}%, Energy={result['Avg_Energy']:.4f}")
        torch.cuda.empty_cache()

    return pd.DataFrame(results)


def get_train_splits(test_split):
    all_splits = ["data1", "data2", "data3", "data4"]
    test_list = [s.strip() for s in test_split.split(",")]
    return tuple(s for s in all_splits if s not in test_list)


def get_train_folder(test_split):
    train_splits = get_train_splits(test_split)
    nums = "".join([s.replace("data", "") for s in train_splits])
    return f"data{nums}"


def get_model_path(model_name, test_split, dataset_name="WiSig_ManyRx"):
    train_folder = get_train_folder(test_split)
    return f"weights/{train_folder}/{model_name}_{dataset_name}_best.pt"


def get_output_prefix(model_name, test_split):
    return f"{model_name}_{test_split}"


def main():
    parser = argparse.ArgumentParser(description="Step 1: Generate Calibration Dataset")

    parser.add_argument("--base_dir", type=str, default="/datasets")
    parser.add_argument("--dataset", type=str, default="WiSig_ManyRx")
    parser.add_argument("--wisig_test_split", type=str, default="data2")
    parser.add_argument("--rx", type=str, default="1-1")
    parser.add_argument("--samples_per_tx", type=int, default=200)

    parser.add_argument("--model_name", type=str, default="MSCAN")

    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--energy_T", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")

    output_prefix = get_output_prefix(args.model_name, args.wisig_test_split)
    model_path = get_model_path(args.model_name, args.wisig_test_split, args.dataset)

    train_splits = get_train_splits(args.wisig_test_split)
    test_splits = tuple(s.strip() for s in args.wisig_test_split.split(","))

    print("=" * 60)
    print("Step 1: Generate Calibration Dataset")
    print("=" * 60)
    print(f"Dataset:      {args.dataset}")
    print(f"Train splits: {','.join(train_splits)}")
    print(f"Test split:   {args.wisig_test_split}")
    print(f"Model:        {model_path}")
    print(f"Output prefix:{output_prefix}")
    print("=" * 60)

    configs = get_all_configs()
    print_statistics()
    print_param_table()

    print("\nLoading dataset...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(
        args.base_dir, args.dataset,
        val_size=0.5, random_state=args.seed,
        wisig_train_splits=train_splits,
        wisig_test_splits=test_splits,
        rx=args.rx, samples_per_tx=args.samples_per_tx
    )

    x_data = np.concatenate([x_train, x_val], axis=0)
    y_data = np.concatenate([y_train, y_val], axis=0)
    print(f"Data shape:   {x_data.shape}")
    print(f"Num classes:  {int(y_data.max()) + 1}")

    print(f"\nLoading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and not hasattr(checkpoint, 'forward'):
        num_classes = int(y_data.max()) + 1
        in_channels = x_data.shape[1]
        model = get_model(args.model_name, in_channels=in_channels,
                          num_classes=num_classes, dataset_name=args.dataset)
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model = model.to(device).eval()

    df = generate_dataset(model, x_data, y_data, device, configs,
                          args.energy_T, args.batch_size)

    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, f"{output_prefix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] CSV saved: {csv_path} ({len(df)} records)")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total configs: {len(df)}")
    print(f"Accuracy range:{df['Accuracy'].min():.2f}% ~ {df['Accuracy'].max():.2f}%")
    print(f"Energy range:  {df['Avg_Energy'].min():.4f} ~ {df['Avg_Energy'].max():.4f}")
    print("=" * 60)

    print("\nCorrelation by augmentation type (Energy vs Accuracy):")
    aug_types = [('N', 'AWGN_sev'), ('P', 'Phase_sev'), ('A', 'PA_sev')]
    for prefix, col in aug_types:
        mask = (df[col] > 0) & (df['Total_sev'] == df[col])
        sub = df[mask]
        if len(sub) > 2:
            r, _ = stats.pearsonr(sub['Avg_Energy'], sub['Accuracy'])
            status = "✓" if r < -0.5 else "✗"
            print(f"  {prefix}: r={r:+.4f} {status}")

    print("\nData preview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()