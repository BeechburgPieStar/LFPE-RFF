"""
Step 2: Train Predictor + Predict Accuracy
==========================================
1. Load calibration dataset from step 1
2. Train linear regression model (Energy -> Accuracy)
3. Predict accuracy on test set

Input:  {model}_{test_split}.csv
Output: predictor_model.joblib, prediction results
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from scipy import stats

from datasets.load_dataset import load_dataset
from models.get_model import get_model
from models.tools import set_seed, select_device, energy_from_logits, normalize_iq


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


def compute_test_metrics(model, x_test, y_test, device, T=1.0, batch_size=128):
    model.eval()

    x_normalized = np.array([normalize_iq(x) for x in x_test])
    x_normalized = np.nan_to_num(x_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    ds = TensorDataset(
        torch.from_numpy(x_normalized).float(),
        torch.from_numpy(y_test).long()
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    correct, total = 0, 0
    all_energies = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            logits = out[1] if isinstance(out, (tuple, list)) else out
            all_energies.append(energy_from_logits(logits, T=T).cpu().numpy())
            correct += logits.argmax(dim=1).eq(target).sum().item()
            total += data.size(0)

    all_energies = np.concatenate(all_energies, axis=0)
    true_acc = 100.0 * correct / max(1, total)
    avg_energy = float(np.mean(all_energies))

    return true_acc, avg_energy


def train_predictor(X, y, degree=1):
    """Train linear regression: Energy -> Accuracy"""
    X_train = X.reshape(-1, 1)

    if degree > 1:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    else:
        model = LinearRegression()

    model.fit(X_train, y)

    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y, y_pred)
    pearson_r, _ = stats.pearsonr(X, y)

    metrics = {
        "mae": mae,
        "pearson_r": pearson_r,
    }

    return model, metrics


def predict_accuracy(model, avg_energy):
    X = np.array([[avg_energy]])
    pred = model.predict(X)[0]
    return np.clip(pred, 0, 100)


def main():
    parser = argparse.ArgumentParser(description="Step 2: Train Predictor + Predict Accuracy")

    parser.add_argument("--base_dir", type=str, default="/datasets")
    parser.add_argument("--dataset", type=str, default="WiSig_ManyRx")
    parser.add_argument("--wisig_test_split", type=str, default="data1")
    parser.add_argument("--rx", type=str, default="1-1")
    parser.add_argument("--samples_per_tx", type=int, default=200)

    parser.add_argument("--model_name", type=str, default="MSCAN")

    parser.add_argument("--degree", type=int, default=1)

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
    test_splits = (args.wisig_test_split,)

    calibration_csv = os.path.join(args.output_dir, f"{output_prefix}.csv")

    print("=" * 60)
    print("Step 2: Train Predictor + Predict Accuracy")
    print("=" * 60)
    print(f"Model:        {args.model_name}")
    print(f"Train splits: {','.join(train_splits)}")
    print(f"Test split:   {args.wisig_test_split}")
    print(f"Calibration:  {calibration_csv}")
    print("=" * 60)

    print(f"\nLoading calibration data: {calibration_csv}")
    df = pd.read_csv(calibration_csv)
    print(f"Records: {len(df)}")

    X = df["Avg_Energy"].values.astype(np.float32)
    y = df["Accuracy"].values.astype(np.float32)

    print(f"\nTraining predictor (degree={args.degree})")
    predictor, metrics = train_predictor(X, y, args.degree)

    print(f"\nPredictor performance:")
    print(f"  MAE:       {metrics['mae']:.2f}%")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")

    print(f"\nLoading test data...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(
        args.base_dir, args.dataset,
        val_size=0.5, random_state=args.seed,
        wisig_train_splits=train_splits,
        wisig_test_splits=test_splits,
        rx=args.rx, samples_per_tx=args.samples_per_tx
    )
    print(f"Test set: {x_test.shape}")

    print(f"\nLoading classifier: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    num_classes = int(len(np.unique(y_train)))
    in_channels = x_train.shape[1]

    if isinstance(checkpoint, dict) and not hasattr(checkpoint, 'forward'):
        classifier = get_model(args.model_name, in_channels=in_channels,
                               num_classes=num_classes, dataset_name=args.dataset)
        classifier.load_state_dict(checkpoint)
    else:
        classifier = checkpoint

    classifier = classifier.to(device).eval()

    del x_train, y_train, x_val, y_val

    print("\nComputing test set metrics...")
    true_acc, avg_energy = compute_test_metrics(
        classifier, x_test, y_test, device,
        T=args.energy_T, batch_size=args.batch_size
    )

    pred_acc = predict_accuracy(predictor, avg_energy)
    error = abs(pred_acc - true_acc)

    print("\n" + "=" * 60)
    print("Prediction Result")
    print("=" * 60)
    print(f"  Avg Energy:         {avg_energy:.4f}")
    print(f"  True Accuracy:      {true_acc:.2f}%")
    print(f"  Predicted Accuracy: {pred_acc:.2f}%")
    print(f"  Error:              {error:.2f}%")
    print("=" * 60)

    model_save_path = os.path.join(args.output_dir, f"{output_prefix}_predictor.joblib")

    model_metadata = {
        'model': predictor,
        'degree': args.degree,
        'mae': metrics['mae'],
        'energy_range': [float(X.min()), float(X.max())],
        'n_samples': len(X),
    }

    joblib.dump(model_metadata, model_save_path)
    print(f"[OK] Predictor saved: {model_save_path}")

    energy_range = model_metadata['energy_range']
    if avg_energy < energy_range[0] or avg_energy > energy_range[1]:
        print(f"\n[WARNING] Energy={avg_energy:.4f} is outside training range {energy_range}")

    result_df = pd.DataFrame([{
        'Model': args.model_name,
        'Test_Split': args.wisig_test_split,
        'Avg_Energy': avg_energy,
        'True_Accuracy': true_acc,
        'Predicted_Accuracy': pred_acc,
        'Error': error,
    }])

    result_csv = os.path.join(args.output_dir, f"{output_prefix}_result.csv")
    result_df.to_csv(result_csv, index=False)
    print(f"[OK] Result CSV: {result_csv}")


if __name__ == "__main__":
    main()