# Experiment file - Federated Averaging (FedAvg) vs Subspace Federated Averaging (SFedAvg)

import argparse
import json
import os
import sys
import math
from typing import Tuple, Dict, List, Optional

import numpy as np

# Try to import sklearn; if unavailable, we will fall back to a deterministic synthetic dataset.
try:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    Y = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def compute_loss_and_grad(W_vec: np.ndarray,
                          X: np.ndarray,
                          y: np.ndarray,
                          n_classes: int,
                          reg: float) -> Tuple[float, np.ndarray]:
    n, d = X.shape
    W = W_vec.reshape(d, n_classes)
    logits = X @ W
    probs = softmax(logits)
    Y = one_hot(y, n_classes)
    # Cross-entropy loss
    ce = -np.sum(Y * np.log(probs + 1e-12)) / n
    # L2 regularization
    reg_term = 0.5 * reg * np.sum(W * W)
    loss = ce + reg_term
    # Gradient
    grad_W = (X.T @ (probs - Y)) / n + reg * W
    grad_vec = grad_W.reshape(-1)
    return loss, grad_vec


def evaluate(W_vec: np.ndarray,
             X: np.ndarray,
             y: np.ndarray,
             n_classes: int,
             reg: float) -> Tuple[float, float]:
    """Return (loss, accuracy) on the provided dataset."""
    loss, _ = compute_loss_and_grad(W_vec, X, y, n_classes, reg)
    d = X.shape[1]
    W = W_vec.reshape(d, n_classes)
    logits = X @ W
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == y))
    return loss, acc


def deterministic_three_class_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Fallback dataset if sklearn is unavailable.
    Construct a deterministic 3-class, linearly-separable dataset in 2D:
    - Class 0 centered near (0, 0)
    - Class 1 centered near (5, 0)
    - Class 2 centered near (0, 5)
    """
    def make_class(cx: float, cy: float, n_per_dim: int) -> np.ndarray:
        pts = []
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                x = cx + (i - (n_per_dim // 2)) * 0.3
                y = cy + (j - (n_per_dim // 2)) * 0.3
                pts.append([x, y])
        return np.array(pts, dtype=np.float64)

    n_per_dim = 8  # 64 samples per class
    X0 = make_class(0.0, 0.0, n_per_dim)
    X1 = make_class(5.0, 0.0, n_per_dim)
    X2 = make_class(0.0, 5.0, n_per_dim)

    X = np.vstack([X0, X1, X2])
    y = np.array([0] * len(X0) + [1] * len(X1) + [2] * len(X2), dtype=np.int64)

    # Shuffle deterministically but without randomness (stable interleaving by class)
    # Build train/test split: first 80% per class for train, remaining 20% for test
    def split_by_class(Xc, yc):
        n = Xc.shape[0]
        n_train = int(0.8 * n)
        return (Xc[:n_train], yc[:n_train]), (Xc[n_train:], yc[n_train:])

    (X0_tr, y0_tr), (X0_te, y0_te) = split_by_class(X0, np.full(len(X0), 0, dtype=np.int64))
    (X1_tr, y1_tr), (X1_te, y1_te) = split_by_class(X1, np.full(len(X1), 1, dtype=np.int64))
    (X2_tr, y2_tr), (X2_te, y2_te) = split_by_class(X2, np.full(len(X2), 2, dtype=np.int64))

    X_train = np.vstack([X0_tr, X1_tr, X2_tr])
    y_train = np.concatenate([y0_tr, y1_tr, y2_tr])
    X_test = np.vstack([X0_te, X1_te, X2_te])
    y_test = np.concatenate([y0_te, y1_te, y2_te])

    # Standardize features (z-score)
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    n_classes = 3
    return X_train, y_train, X_test, y_test, n_classes


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load a standard dataset without network access.
    Prefer sklearn digits; fall back to a deterministic synthetic dataset.
    """
    if SKLEARN_AVAILABLE:
        ds = load_digits()
        X = ds.data.astype(np.float64)
        y = ds.target.astype(np.int64)
        # Scale features to [0,1]
        X = X / 16.0
        # Train/test split (deterministic)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=12345, stratify=y
        )
        # Standardize (z-score) for stable optimization
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        n_classes = int(np.max(y) + 1)
        return X_train, y_train, X_test, y_test, n_classes
    else:
        return deterministic_three_class_dataset()


def stratified_partition_clients(X: np.ndarray,
                                 y: np.ndarray,
                                 n_clients: int) -> List[Dict[str, np.ndarray]]:
    """
    Stratified round-robin partition of (X, y) into n_clients shards for IID-ish distribution.
    """
    clients = [{'X': [], 'y': []} for _ in range(n_clients)]
    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        # Deterministic ordering
        for k, i in enumerate(idx):
            clients[k % n_clients]['X'].append(X[i])
            clients[k % n_clients]['y'].append(y[i])
    # Convert to arrays
    for cl in clients:
        cl['X'] = np.array(cl['X'], dtype=np.float64)
        cl['y'] = np.array(cl['y'], dtype=np.int64)
    return clients


def partition_non_iid(X: np.ndarray,
                      y: np.ndarray,
                      n_clients: int,
                      classes_per_client: int = 2,
                      seed: Optional[int] = 12345) -> List[Dict[str, np.ndarray]]:
    """
    Non-IID label-skew partition using class shards:
    - Sort samples by label, split the dataset into 'num_shards = n_clients * classes_per_client'
      contiguous shards (each shard dominated by a single class).
    - Assign 'classes_per_client' shards to each client.
    This yields clients that predominantly observe a few classes (label skew).
    """
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    # Sort indices by label to make shards class-homogeneous
    sorted_idx = np.argsort(y, kind='mergesort')
    num_shards = max(1, n_clients * max(1, classes_per_client))
    shard_size = max(1, n // num_shards)
    shards = []
    start = 0
    while start < n:
        end = min(n, start + shard_size)
        shards.append(sorted_idx[start:end])
        start = end
    # If we have fewer shards than required (due to rounding), pad by splitting last shard
    while len(shards) < num_shards and len(shards[-1]) > 1:
        last = shards.pop()
        mid = len(last) // 2
        shards.append(last[:mid])
        shards.append(last[mid:])
    # Shuffle shards deterministically
    rng.shuffle(shards)
    # Assign shards to clients
    clients_idx = [[] for _ in range(n_clients)]
    for i in range(n_clients):
        start = i * classes_per_client
        end = min(len(shards), start + classes_per_client)
        for s in range(start, end):
            clients_idx[i].extend(shards[s].tolist())
    # In case shards < n_clients * classes_per_client, distribute remaining shards round-robin
    assigned = n_clients * classes_per_client
    for j in range(assigned, len(shards)):
        clients_idx[j % n_clients].extend(shards[j].tolist())
    # Build client datasets
    clients = []
    for idxs in clients_idx:
        idxs_arr = np.array(idxs, dtype=np.int64)
        clients.append({
            'X': X[idxs_arr],
            'y': y[idxs_arr]
        })
    return clients


def partition_label_skew_dirichlet(X: np.ndarray,
                                   y: np.ndarray,
                                   n_clients: int,
                                   alpha: float = 0.2,
                                   seed: Optional[int] = 12345) -> List[Dict[str, np.ndarray]]:
    """
    Non-IID label-skew partition using Dirichlet allocation over clients per class.
    For each class c, sample proportions p_c ~ Dirichlet(alpha * 1_{n_clients}) and
    allocate samples of class c to clients accordingly (without replacement).
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    clients_idx = [[] for _ in range(n_clients)]
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        # Sample client proportions for this class
        p = rng.dirichlet(alpha * np.ones(n_clients, dtype=np.float64))
        # Compute quota per client
        counts = np.floor(p * len(idx_c)).astype(int)
        # Assign remaining due to rounding
        remainder = len(idx_c) - np.sum(counts)
        if remainder > 0:
            # Distribute remainder to clients with largest fractional parts
            frac = p * len(idx_c) - counts
            order = np.argsort(-frac)
            for k in range(remainder):
                counts[order[k % n_clients]] += 1
        # Slice indices according to counts
        start = 0
        for i in range(n_clients):
            cnt = int(counts[i])
            if cnt > 0:
                sel = idx_c[start:start + cnt]
                clients_idx[i].extend(sel.tolist())
                start += cnt
    # Build client datasets
    clients = []
    for idxs in clients_idx:
        idxs_arr = np.array(idxs, dtype=np.int64)
        clients.append({
            'X': X[idxs_arr],
            'y': y[idxs_arr]
        })
    return clients


def build_projector(D: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build a random subspace basis P in R^{D x r} with orthonormal columns (Stiefel manifold).
    Returns P (not Pi), and clients will use projection via P @ (P^T v).
    """
    A = rng.normal(size=(D, r))
    # QR decomposition for orthonormal basis
    Q, _ = np.linalg.qr(A)
    # Ensure shape exactly D x r
    if Q.shape[1] > r:
        Q = Q[:, :r]
    elif Q.shape[1] < r:
        # Pad columns if numerical issues (unlikely)
        extra = r - Q.shape[1]
        B = rng.normal(size=(D, extra))
        Q2, _ = np.linalg.qr(B)
        Q = np.concatenate([Q, Q2[:, :extra]], axis=1)
    return Q


def project_with_P(P: np.ndarray, v: np.ndarray) -> np.ndarray:
    # One-sided projection: Pi v = P (P^T v)
    return P @ (P.T @ v)


def get_minibatch_indices(n: int, start: int, B: int) -> np.ndarray:
    """
    Deterministic cyclic minibatch indices of size B starting at 'start'.
    """
    end = start + B
    if end <= n:
        return np.arange(start, end, dtype=np.int64)
    else:
        first = np.arange(start, n, dtype=np.int64)
        rest = np.arange(0, end - n, dtype=np.int64)
        return np.concatenate([first, rest], axis=0)


def client_update(X: np.ndarray,
                  y: np.ndarray,
                  theta_vec: np.ndarray,
                  n_classes: int,
                  reg: float,
                  tau: int,
                  eta: float,
                  mu: float,
                  B: int,
                  P: Optional[np.ndarray],
                  v_prev: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform local updates on a client:
    momentum with optional one-sided projection using basis P.
    Returns (delta_theta, v_new).
    """
    d_in = X.shape[1]
    D = d_in * n_classes

    # Initialize momentum state
    if v_prev is None:
        v = np.zeros(D, dtype=np.float64)
    else:
        v = v_prev.copy()
        if P is not None:
            v = project_with_P(P, v)

    theta_local = theta_vec.copy()
    n = X.shape[0]
    start = 0

    for s in range(tau):
        idx = get_minibatch_indices(n, start, B)
        Xb = X[idx]
        yb = y[idx]
        start = (start + B) % n

        # Compute gradient from actual data
        _, grad = compute_loss_and_grad(theta_local, Xb, yb, n_classes, reg)

        # One-sided projected momentum
        if P is not None:
            grad = project_with_P(P, grad)

        v = mu * v + grad
        theta_local = theta_local - eta * v

    delta = theta_local - theta_vec
    return delta, v


def run_federated(method: str,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  n_classes: int,
                  n_clients: int = 10,
                  client_frac: float = 0.5,
                  rounds: int = 20,
                  local_steps: int = 5,
                  lr: float = 0.1,
                  mu: float = 0.9,
                  batch_size: int = 64,
                  reg: float = 1e-4,
                  subspace_frac: float = 0.25,
                  seed: int = 12345,
                  partition_strategy: str = "iid",
                  classes_per_client: int = 2,
                  dirichlet_alpha: float = 0.2) -> Dict[str, List[float]]:
    """
    Run a federated experiment for the given method ('FedAvg' or 'SFedAvg').
    Returns metric series: train_loss, test_accuracy, cumulative_comm_params.
    """
    assert method in ("FedAvg", "SFedAvg")

    # Global model parameters
    d_in = X_train.shape[1]
    D = d_in * n_classes
    theta = np.zeros(D, dtype=np.float64)

    # Partition data across clients
    if partition_strategy == "iid":
        clients = stratified_partition_clients(X_train, y_train, n_clients)
    elif partition_strategy == "noniid_shards":
        clients = partition_non_iid(X_train, y_train, n_clients, classes_per_client=classes_per_client, seed=seed)
    elif partition_strategy == "dirichlet":
        clients = partition_label_skew_dirichlet(X_train, y_train, n_clients, alpha=dirichlet_alpha, seed=seed)
    else:
        raise ValueError(f"Unknown partition_strategy='{partition_strategy}'. Use 'iid', 'noniid_shards', or 'dirichlet'.")
    N = len(clients)
    m = max(1, int(round(client_frac * N)))

    # Momentum state per client
    client_v: List[Optional[np.ndarray]] = [None for _ in range(N)]

    # Metrics
    train_losses: List[float] = []
    test_accuracies: List[float] = []
    cum_comm: List[int] = []
    comm_so_far = 0

    rng = np.random.default_rng(seed)

    # Deterministic client selection pattern (round-robin window)
    for t in range(rounds):
        # Subspace basis for SFedAvg
        P = None
        r = 0
        if method == "SFedAvg":
            r = max(1, int(round(subspace_frac * D)))
            P = build_projector(D, r, rng)

        # Select clients deterministically
        selected = [((t * m) + k) % N for k in range(m)]

        deltas = []
        for i in selected:
            Xi = clients[i]['X']
            yi = clients[i]['y']
            delta, v_new = client_update(
                Xi, yi, theta, n_classes, reg,
                tau=local_steps, eta=lr, mu=mu, B=batch_size,
                P=P, v_prev=client_v[i]
            )
            client_v[i] = v_new
            deltas.append(delta)

        if deltas:
            mean_delta = np.mean(np.stack(deltas, axis=0), axis=0)
            theta = theta + mean_delta

        # Metrics after aggregation
        tr_loss, _ = evaluate(theta, X_train, y_train, n_classes, reg)
        _, te_acc = evaluate(theta, X_test, y_test, n_classes, reg)
        train_losses.append(float(tr_loss))
        test_accuracies.append(float(te_acc))

        # Communication accounting (parameter counts)
        if method == "SFedAvg":
            comm_this_round = m * (2 * D + D * r)  # down: theta(D)+P(D*r); up: delta(D)
        else:
            comm_this_round = m * (2 * D)          # down: theta(D); up: delta(D)
        comm_so_far += comm_this_round
        cum_comm.append(int(comm_so_far))

    return {
        "train_loss": train_losses,
        "test_accuracy": test_accuracies,
        "cumulative_comm_params": cum_comm
    }


def snapshot_self(out_dir: str):
    """Save a snapshot of this script into the output directory for reproducibility."""
    try:
        src = os.path.abspath(__file__)
        dst = os.path.join(out_dir, "experiment.py")
        with open(src, "r", encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8") as fdst:
            fdst.write(fsrc.read())
    except Exception:
        # Best-effort snapshot; do not crash the experiment if snapshot fails.
        pass


def run_experiment(args) -> Dict[str, Dict[str, List[float]]]:
    """
    Execute a single experiment run for one or both algorithms using parameters in args.
    Returns a results dictionary with means/stds lists for each metric.
    """
    # Load data deterministically
    X_train, y_train, X_test, y_test, n_classes = load_dataset()

    metrics_by_method: Dict[str, Dict[str, List[float]]] = {}

    # Run selected algorithms
    if getattr(args, "algorithm", "both") in ("both", "FedAvg"):
        fedavg_metrics = run_federated(
            method="FedAvg",
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_classes=n_classes,
            n_clients=args.clients, client_frac=args.client_frac, rounds=args.rounds,
            local_steps=args.local_steps, lr=args.lr, mu=args.mu, batch_size=args.batch_size,
            reg=args.reg, subspace_frac=args.subspace_frac, seed=12345,
            partition_strategy=args.partition_strategy,
            classes_per_client=args.classes_per_client,
            dirichlet_alpha=args.dirichlet_alpha
        )
        metrics_by_method["FedAvg"] = fedavg_metrics

    if getattr(args, "algorithm", "both") in ("both", "SFedAvg"):
        sfedavg_metrics = run_federated(
            method="SFedAvg",
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_classes=n_classes,
            n_clients=args.clients, client_frac=args.client_frac, rounds=args.rounds,
            local_steps=args.local_steps, lr=args.lr, mu=args.mu, batch_size=args.batch_size,
            reg=args.reg, subspace_frac=args.subspace_frac, seed=12345,
            partition_strategy=args.partition_strategy,
            classes_per_client=args.classes_per_client,
            dirichlet_alpha=args.dirichlet_alpha
        )
        metrics_by_method["SFedAvg"] = sfedavg_metrics

    zeros = lambda arr: [0.0] * len(arr)

    results: Dict[str, Dict[str, List[float]]] = {}
    for method_name, metrics in metrics_by_method.items():
        results[f"train_loss/{method_name}"] = {
            "means": list(map(float, metrics["train_loss"])),
            "stds": zeros(metrics["train_loss"])
        }
        results[f"test_accuracy/{method_name}"] = {
            "means": list(map(float, metrics["test_accuracy"])),
            "stds": zeros(metrics["test_accuracy"])
        }
        results[f"cumulative_comm_params/{method_name}"] = {
            "means": list(map(int, metrics["cumulative_comm_params"])),
            "stds": [0] * len(metrics["cumulative_comm_params"])
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="FedAvg vs SFedAvg experiment on a standard dataset.")
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for results.')
    # Optional hyperparameters
    parser.add_argument('--rounds', type=int, default=20, help='Number of federated rounds T.')
    parser.add_argument('--clients', type=int, default=10, help='Number of total clients N.')
    parser.add_argument('--client_frac', type=float, default=0.5, help='Client fraction C per round.')
    parser.add_argument('--local_steps', type=int, default=5, help='Local steps tau per client.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate eta.')
    parser.add_argument('--mu', type=float, default=0.9, help='Momentum coefficient mu.')
    parser.add_argument('--batch_size', type=int, default=64, help='Local minibatch size B.')
    parser.add_argument('--reg', type=float, default=1e-4, help='L2 regularization strength.')
    parser.add_argument('--subspace_frac', type=float, default=0.25, help='Subspace dimension fraction r/D for SFedAvg.')
    # Partitioning and algorithm selection for scenarios
    parser.add_argument('--partition_strategy', type=str, default='iid',
                        choices=['iid', 'noniid_shards', 'dirichlet'],
                        help="Data partition strategy across clients.")
    parser.add_argument('--classes_per_client', type=int, default=2,
                        help="For noniid_shards, number of class shards per client.")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.2,
                        help="For dirichlet partition, concentration parameter alpha (smaller=more skew).")
    parser.add_argument('--algorithm', type=str, default='both',
                        choices=['both', 'FedAvg', 'SFedAvg'],
                        help="Which algorithm(s) to run.")
    # Enable batch hyperparameter tuning
    parser.add_argument('--enable_tuning', action='store_true', help='Enable batch parameter tuning')
    args = parser.parse_args()

    # Ensure root output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    if args.enable_tuning:
        # Define parameter configurations to test (noniid-label-skew scenario)
        tuning_configs = [
            {
                "lr": 0.05,
                "mu": 0.95,
                "local_steps": 10,
                "rationale": "Increase momentum to smooth client drift at the baseline local update depth without changing step size."
            },
            {
                "lr": 0.05,
                "mu": 0.8,
                "local_steps": 10,
                "rationale": "Lower momentum can reduce overshoot and oscillations when client updates are biased by label skew."
            },
            {
                "lr": 0.075,
                "mu": 0.9,
                "local_steps": 5,
                "rationale": "Slightly larger step size with fewer local steps to improve early-round progress while reducing drift per round."
            },
            {
                "lr": 0.03,
                "mu": 0.9,
                "local_steps": 10,
                "rationale": "More conservative learning rate to improve stability with the same number of local steps."
            },
            {
                "lr": 0.05,
                "mu": 0.9,
                "local_steps": 5,
                "rationale": "Cut local steps to halve client drift per synchronization while keeping the baseline step size and momentum."
            },
            {
                "lr": 0.075,
                "mu": 0.95,
                "local_steps": 5,
                "rationale": "Combine higher momentum and slightly higher LR with fewer local steps for faster but smoothed progress."
            },
            {
                "lr": 0.03,
                "mu": 0.95,
                "local_steps": 15,
                "rationale": "Stress-test larger local updates with stronger momentum and a smaller LR to see if smoothing offsets drift."
            }
        ]

        # Create tuning subdirectory
        tuning_dir = os.path.join(args.out_dir, "tuning")
        os.makedirs(tuning_dir, exist_ok=True)

        all_results: Dict[str, Dict[str, object]] = {}
        for idx, config in enumerate(tuning_configs, 1):
            config_dir = os.path.join(tuning_dir, f"config_{idx}")
            os.makedirs(config_dir, exist_ok=True)

            # Override parameters with config values (ignore non-arg fields like 'rationale')
            for param_name, param_value in config.items():
                if hasattr(args, param_name):
                    setattr(args, param_name, param_value)

            # Run experiment with this configuration
            results = run_experiment(args)

            # Save results for this configuration
            with open(os.path.join(config_dir, "final_info.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            all_results[f"config_{idx}"] = {
                "parameters": config,
                "results": results
            }

        # Save aggregated results
        with open(os.path.join(tuning_dir, "all_configs.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        # Snapshot script once per run
        snapshot_self(args.out_dir)

    else:
        # Normal single-run mode (baseline)
        baseline_dir = os.path.join(args.out_dir, "baseline")
        os.makedirs(baseline_dir, exist_ok=True)

        results = run_experiment(args)

        with open(os.path.join(baseline_dir, "final_info.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Snapshot script
        snapshot_self(args.out_dir)


if __name__ == "__main__":
    main()
