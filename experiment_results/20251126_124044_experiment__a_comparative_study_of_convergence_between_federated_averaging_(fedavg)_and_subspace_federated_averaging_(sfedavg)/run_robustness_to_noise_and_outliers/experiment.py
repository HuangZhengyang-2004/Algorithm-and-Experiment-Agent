# Experiment file - Federated Averaging (FedAvg) vs Subspace Federated Averaging (SFedAvg)

import argparse
import json
import os
import shutil
import time
from typing import Dict, Tuple, List

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def loss_and_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 0.0) -> Tuple[float, np.ndarray]:
    p = sigmoid(X @ w)
    loss = binary_cross_entropy(y, p)
    grad = (X.T @ (p - y)) / X.shape[0]
    if l2 > 0.0:
        loss += 0.5 * l2 * float(np.dot(w, w))
        grad = grad + l2 * w
    return loss, grad


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    p = sigmoid(X @ w)
    preds = (p >= 0.5).astype(np.int64)
    return float(np.mean(preds == y))


def generate_logistic_data(n_samples: int, n_features: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a non-trivial binary classification dataset for logistic regression.
    - Features X ~ N(0, 1)
    - True weight w_true is unit-norm to avoid overly separable data
    - Labels are generated via Bernoulli(sigmoid(scale * X @ w_true))
    - No label noise injected here; corruption (noise/outliers) is applied per client later to match scenarios.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n_samples, n_features))
    w_true = rng.normal(0, 1.0, size=n_features)
    w_true_norm = np.linalg.norm(w_true) + 1e-12
    w_true = w_true / w_true_norm
    scale = 1.0  # Controls separability; 1.0 gives moderate difficulty
    logits = X @ w_true * scale
    p = sigmoid(logits)
    y = rng.binomial(1, p, size=n_samples).astype(np.int64)
    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    split = int(n * (1.0 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def partition_clients(X: np.ndarray, y: np.ndarray, num_clients: int, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    splits = np.array_split(idx, num_clients)
    client_data = []
    for s in splits:
        client_data.append((X[s], y[s]))
    return client_data


def orthonormal_subspace(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    # Draw random Gaussian d x r and orthonormalize via QR
    A = rng.normal(0, 1.0, size=(d, r))
    Q, _ = np.linalg.qr(A, mode='reduced')  # Q: d x r with orthonormal columns
    return Q


def project_vec(P: np.ndarray, v: np.ndarray) -> np.ndarray:
    # One-sided projection: Pi v = P (P^T v) without forming Pi in d x d
    return P @ (P.T @ v)


def sample_batch(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n == 0:
        raise ValueError("Client has no data.")
    if batch_size >= n:
        # Sample with replacement to ensure batch_size elements
        idx = rng.integers(0, n, size=batch_size)
    else:
        idx = rng.choice(n, size=batch_size, replace=False)
    return X[idx], y[idx]


def add_label_noise(y: np.ndarray, noise_rate: float, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly flip a proportion `noise_rate` of binary labels.
    """
    if noise_rate <= 0.0:
        return y
    flips = rng.random(y.shape[0]) < noise_rate
    y_noisy = y.copy()
    y_noisy[flips] = 1 - y_noisy[flips]
    return y_noisy


def inject_feature_outliers(X: np.ndarray, outlier_rate: float, rng: np.random.Generator, scale: float = 10.0) -> np.ndarray:
    """
    Inject heavy-tailed (Cauchy) feature outliers into ~outlier_rate of rows.
    Adds scaled Cauchy noise to selected rows to simulate extreme outliers.
    """
    if outlier_rate <= 0.0:
        return X
    n = X.shape[0]
    mask = rng.random(n) < outlier_rate
    if not np.any(mask):
        return X
    noise = rng.standard_cauchy(size=(int(mask.sum()), X.shape[1])) * scale
    X_out = X.copy()
    X_out[mask] = X_out[mask] + noise
    return X_out


def client_update_fedavg(
    X_i: np.ndarray,
    y_i: np.ndarray,
    w_global: np.ndarray,
    tau: int,
    eta: float,
    mu: float,
    batch_size: int,
    v_prev: np.ndarray,
    rng: np.random.Generator,
    l2: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard FedAvg local update without momentum, following the baseline algorithm.
    Each client performs tau steps of SGD on its local data starting from the global model.
    """
    w_local = w_global.copy()
    for _ in range(tau):
        Xb, yb = sample_batch(X_i, y_i, batch_size, rng)
        _, grad = loss_and_grad(Xb, yb, w_local, l2=l2)
        w_local = w_local - eta * grad
    delta = w_local - w_global
    # No momentum state maintained for FedAvg baseline; return the previous (unchanged) state
    return delta, v_prev


def client_update_sfedavg(
    X_i: np.ndarray,
    y_i: np.ndarray,
    w_global: np.ndarray,
    Pi: np.ndarray,
    tau: int,
    eta: float,
    mu: float,
    batch_size: int,
    v_prev: np.ndarray,
    rng: np.random.Generator,
    l2: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subspace Federated Averaging (SFedAvg) with one-sided projected momentum.
    Momentum projection at block start: v0 = Pi @ v_prev if available, else 0.
    Per-step update: v_{s+1} = mu * v_s + Pi @ g_s, w_{s+1} = w_s - eta * v_{s+1}.
    """
    w_local = w_global.copy()
    v = (Pi @ v_prev) if v_prev is not None else np.zeros_like(w_global)
    for _ in range(tau):
        Xb, yb = sample_batch(X_i, y_i, batch_size, rng)
        _, grad = loss_and_grad(Xb, yb, w_local, l2=l2)
        grad_proj = Pi @ grad
        v = mu * v + grad_proj
        w_local = w_local - eta * v
    delta = w_local - w_global
    return delta, v


def run_experiment(
    out_dir: str,
    num_rounds: int = 60,
    num_clients: int = 50,
    client_fraction: float = 0.2,
    local_steps: int = 5,
    stepsize: float = 0.05,
    momentum: float = 0.8,
    batch_size: int = 128,
    n_samples: int = 20000,
    n_features: int = 50,
    subspace_dim: int = 8,
    l2_reg: float = 0.0001,
    seed: int = 2,
    algorithm: str = "both"
) -> Dict[str, Dict[str, List[float]]]:
    # Reproducibility
    rng = np.random.default_rng(seed)

    # Validate parameters
    num_rounds = int(num_rounds)
    num_clients = int(num_clients)
    local_steps = int(local_steps)
    batch_size = int(batch_size)
    n_samples = int(n_samples)
    n_features = int(n_features)
    subspace_dim = int(subspace_dim)

    algorithm = str(algorithm).lower()
    if algorithm not in ("both", "fedavg", "sfedavg"):
        raise ValueError("algorithm must be one of 'both', 'fedavg', 'sfedavg'.")

    if not (0.0 < client_fraction <= 1.0):
        raise ValueError("client_fraction must be in (0, 1].")
    if subspace_dim <= 0:
        raise ValueError("subspace_dim must be positive.")

    # Generate dataset (binary classification with logistic regression)
    X, y = generate_logistic_data(n_samples=n_samples, n_features=n_features, seed=seed)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2, seed=seed)

    # Standardize features using training statistics and add bias term
    mu_X = X_train.mean(axis=0)
    sigma_X = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu_X) / sigma_X
    X_test = (X_test - mu_X) / sigma_X
    X_train = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)
    X_test = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)

    clients = partition_clients(X_train, y_train, num_clients=num_clients, seed=seed)

    # Scenario-specific corruption: unequal label noise per client (~30%) and ~5% heavy-tailed feature outliers
    clients_corrupted: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, (Xi, yi) in enumerate(clients):
        noise_rate_i = float(rng.uniform(0.2, 0.4))  # around 30%, heterogeneous across clients
        yi_noisy = add_label_noise(yi, noise_rate=noise_rate_i, rng=rng)
        Xi_out = inject_feature_outliers(Xi, outlier_rate=0.05, rng=rng, scale=10.0)
        clients_corrupted.append((Xi_out, yi_noisy))
    clients = clients_corrupted

    d = X_train.shape[1]
    m_per_round = max(1, int(round(client_fraction * num_clients)))
    if subspace_dim > d:
        raise ValueError(f"subspace_dim ({subspace_dim}) must be <= model dimension d ({d}).")

    # Initialize global models
    w0 = rng.normal(0, 0.1, size=d)
    w_fedavg = w0.copy()
    w_sfedavg = w0.copy()

    # Per-client momentum states
    v_prev_fedavg = [None for _ in range(num_clients)]
    v_prev_sfedavg = [None for _ in range(num_clients)]

    # Metrics
    train_loss_fedavg: List[float] = []
    test_acc_fedavg: List[float] = []
    comm_floats_fedavg: List[float] = []

    train_loss_sfedavg: List[float] = []
    test_acc_sfedavg: List[float] = []
    comm_floats_sfedavg: List[float] = []

    cum_comm_fedavg = 0.0
    cum_comm_sfedavg = 0.0

    for t in range(num_rounds):
        # Sample clients this round
        client_indices = rng.choice(num_clients, size=m_per_round, replace=False)

        # SFedAvg: sample one-sided subspace P_t and projector Pi_t = P_t P_t^T (held fixed within round)
        Pi_t = None
        if algorithm in ("both", "sfedavg"):
            P_t = orthonormal_subspace(d=d, r=subspace_dim, rng=rng)
            Pi_t = P_t @ P_t.T

        # Communication accounting (server -> clients)
        # FedAvg sends w (d floats) to each of m clients
        if algorithm in ("both", "fedavg"):
            cum_comm_fedavg += m_per_round * d
        # SFedAvg sends w (d) and Pi projector (d*d) to each of m clients (per pseudocode)
        if algorithm in ("both", "sfedavg"):
            cum_comm_sfedavg += m_per_round * (d + d * d)

        # Client updates and aggregation
        deltas_fedavg = []
        deltas_sfedavg = []

        for i in client_indices:
            Xi, yi = clients[i]

            if algorithm in ("both", "fedavg"):
                # FedAvg update
                delta_f, v_f = client_update_fedavg(
                    Xi, yi, w_fedavg, local_steps, stepsize, momentum, batch_size,
                    v_prev_fedavg[i], rng, l2=l2_reg
                )
                deltas_fedavg.append(delta_f)
                v_prev_fedavg[i] = None

            if algorithm in ("both", "sfedavg"):
                # SFedAvg update (projected gradients/momentum)
                delta_s, v_s = client_update_sfedavg(
                    Xi, yi, w_sfedavg, Pi_t, local_steps, stepsize, momentum, batch_size,
                    v_prev_sfedavg[i], rng, l2=l2_reg
                )
                deltas_sfedavg.append(delta_s)
                v_prev_sfedavg[i] = v_s

        # Communication accounting (clients -> server): deltas of size d
        if algorithm in ("both", "fedavg"):
            cum_comm_fedavg += m_per_round * d
        if algorithm in ("both", "sfedavg"):
            cum_comm_sfedavg += m_per_round * d

        # Aggregate on server
        if deltas_fedavg:
            avg_delta_f = np.mean(np.stack(deltas_fedavg, axis=0), axis=0)
            w_fedavg = w_fedavg + avg_delta_f

        if deltas_sfedavg:
            avg_delta_s = np.mean(np.stack(deltas_sfedavg, axis=0), axis=0)
            w_sfedavg = w_sfedavg + avg_delta_s

        # Evaluate after round t
        if algorithm in ("both", "fedavg"):
            train_loss_fedavg.append(loss_and_grad(X_train, y_train, w_fedavg, l2=l2_reg)[0])
            test_acc_fedavg.append(accuracy(X_test, y_test, w_fedavg))
            comm_floats_fedavg.append(cum_comm_fedavg)

        if algorithm in ("both", "sfedavg"):
            train_loss_sfedavg.append(loss_and_grad(X_train, y_train, w_sfedavg, l2=l2_reg)[0])
            test_acc_sfedavg.append(accuracy(X_test, y_test, w_sfedavg))
            comm_floats_sfedavg.append(cum_comm_sfedavg)

    # Prepare results dictionary with means and stds (stds zero for single run)
    zeros_fed = [0.0] * len(train_loss_fedavg)
    zeros_sfed = [0.0] * len(train_loss_sfedavg)

    # Scenario-specific robust metrics
    def rounds_to_threshold(acc_list: List[float], thr: float) -> int:
        for idx, a in enumerate(acc_list):
            if a >= thr:
                return idx + 1  # rounds are 1-based
        return -1

    sample_rounds = [10, 30, num_rounds]
    sample_rounds = [r for r in sample_rounds if r >= 1]

    def sample_at_rounds(vals: List[float], rounds: List[int]) -> List[float]:
        res = []
        for r in rounds:
            if r <= len(vals):
                res.append(vals[r - 1])
        return res

    robust_acc_fed = sample_at_rounds(test_acc_fedavg, sample_rounds)
    robust_acc_sfed = sample_at_rounds(test_acc_sfedavg, sample_rounds)

    rounds_to_70_fed = rounds_to_threshold(test_acc_fedavg, 0.70)
    rounds_to_70_sfed = rounds_to_threshold(test_acc_sfedavg, 0.70)

    max_spike_fed = float(np.max(np.maximum(0.0, np.diff(train_loss_fedavg)))) if len(train_loss_fedavg) > 1 else 0.0
    max_spike_sfed = float(np.max(np.maximum(0.0, np.diff(train_loss_sfedavg)))) if len(train_loss_sfedavg) > 1 else 0.0

    results = {}
    if train_loss_fedavg:
        results["train_loss_fedavg"] = {"means": train_loss_fedavg, "stds": zeros_fed}
        results["test_acc_fedavg"] = {"means": test_acc_fedavg, "stds": zeros_fed}
        results["comm_floats_fedavg"] = {"means": comm_floats_fedavg, "stds": zeros_fed}
        results["robust_accuracy_at_rounds_fedavg"] = {"means": robust_acc_fed, "stds": [0.0] * len(robust_acc_fed)}
        if rounds_to_70_fed != -1:
            results["rounds_to_70_acc_fedavg"] = {"means": [rounds_to_70_fed], "stds": [0.0]}
        results["max_loss_spike_fedavg"] = {"means": [max_spike_fed], "stds": [0.0]}
    if train_loss_sfedavg:
        results["train_loss_sfedavg"] = {"means": train_loss_sfedavg, "stds": zeros_sfed}
        results["test_acc_sfedavg"] = {"means": test_acc_sfedavg, "stds": zeros_sfed}
        results["comm_floats_sfedavg"] = {"means": comm_floats_sfedavg, "stds": zeros_sfed}
        results["robust_accuracy_at_rounds_sfedavg"] = {"means": robust_acc_sfed, "stds": [0.0] * len(robust_acc_sfed)}
        if rounds_to_70_sfed != -1:
            results["rounds_to_70_acc_sfedavg"] = {"means": [rounds_to_70_sfed], "stds": [0.0]}
        results["max_loss_spike_sfedavg"] = {"means": [max_spike_sfed], "stds": [0.0]}

    # Ensure output directories
    baseline_dir = os.path.join(out_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    # Save snapshot of this experiment script
    try:
        src_path = os.path.abspath(__file__)
        shutil.copyfile(src_path, os.path.join(out_dir, "experiment.py"))
    except Exception:
        # If __file__ is not available or copy fails, ignore snapshot silently
        pass

    # Save results
    with open(os.path.join(baseline_dir, "final_info.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare FedAvg vs SFedAvg with logistic regression.")
    parser.add_argument('--out_dir', type=str, required=True, help="Output directory to store results.")
    parser.add_argument('--rounds', type=int, default=60, help="Number of federated rounds T.")
    parser.add_argument('--clients', type=int, default=50, help="Number of clients N.")
    parser.add_argument('--client_fraction', type=float, default=0.2, help="Fraction C of clients per round.")
    parser.add_argument('--local_steps', type=int, default=5, help="Local steps tau.")
    parser.add_argument('--stepsize', type=float, default=0.05, help="Learning rate eta.")
    parser.add_argument('--momentum', type=float, default=0.8, help="Momentum mu.")
    parser.add_argument('--batch_size', type=int, default=128, help="Minibatch size B.")
    parser.add_argument('--samples', type=int, default=20000, help="Total number of samples.")
    parser.add_argument('--features', type=int, default=50, help="Number of features (model dimension d).")
    parser.add_argument('--subspace_dim', type=int, default=8, help="Subspace dimension r for SFedAvg.")
    parser.add_argument('--l2', type=float, default=0.0001, help="L2 regularization strength.")
    parser.add_argument('--seed', type=int, default=2, help="Random seed.")
    parser.add_argument('--algo', type=str, default='both', choices=['both', 'fedavg', 'sfedavg'], help="Which algorithm(s) to run.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Run experiment
    _ = run_experiment(
        out_dir=args.out_dir,
        num_rounds=args.rounds,
        num_clients=args.clients,
        client_fraction=args.client_fraction,
        local_steps=args.local_steps,
        stepsize=args.stepsize,
        momentum=args.momentum,
        batch_size=args.batch_size,
        n_samples=args.samples,
        n_features=args.features,
        subspace_dim=args.subspace_dim,
        l2_reg=args.l2,
        seed=args.seed,
        algorithm=args.algo
    )


if __name__ == "__main__":
    main()
