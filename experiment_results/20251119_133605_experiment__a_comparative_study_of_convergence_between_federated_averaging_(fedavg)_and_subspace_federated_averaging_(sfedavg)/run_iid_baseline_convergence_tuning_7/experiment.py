# Experiment file - Federated Averaging (FedAvg) vs Subspace Federated Averaging (SFedAvg)
# Implements SFedAvg-Golore (One-Sided Random Subspace + Momentum Projection)

import argparse
import json
import os
from typing import Dict, Tuple, List

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    # logits: (n_samples, K)
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    # y: (n_samples,)
    y_oh = np.zeros((y.shape[0], num_classes), dtype=float)
    y_oh[np.arange(y.shape[0]), y] = 1.0
    return y_oh


def compute_loss(X: np.ndarray, y: np.ndarray, W: np.ndarray) -> float:
    # Cross-entropy loss for softmax regression
    logits = X @ W  # (n, K)
    probs = softmax(logits)
    y_oh = one_hot(y, W.shape[1])
    # Avoid log(0) with small epsilon
    eps = 1e-12
    ce = -np.sum(y_oh * np.log(probs + eps)) / X.shape[0]
    return float(ce)


def compute_accuracy(X: np.ndarray, y: np.ndarray, W: np.ndarray) -> float:
    logits = X @ W
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def compute_batch_grad(Xb: np.ndarray, yb: np.ndarray, W: np.ndarray) -> np.ndarray:
    # Gradient of mean cross-entropy loss for softmax regression
    probs = softmax(Xb @ W)  # (B, K)
    yb_oh = one_hot(yb, W.shape[1])
    grad = Xb.T @ (probs - yb_oh) / Xb.shape[0]  # (d, K)
    return grad


def generate_data(n_train: int, n_test: int, d: int, K: int, noise_std: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # True weight matrix for data generation
    W_true = rng.normal(0, 1, size=(d, K))
    # Make classes separated by scaling and small random bias
    bias = rng.normal(0, 0.1, size=(K,))

    X_train = rng.normal(0, 1, size=(n_train, d))
    X_test = rng.normal(0, 1, size=(n_test, d))

    logits_train = X_train @ W_true + bias
    logits_test = X_test @ W_true + bias

    # Add Gaussian noise to logits
    logits_train += rng.normal(0, noise_std, size=logits_train.shape)
    logits_test += rng.normal(0, noise_std, size=logits_test.shape)

    y_train = np.argmax(logits_train, axis=1)
    y_test = np.argmax(logits_test, axis=1)

    # Standardize features to stabilize training
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train.astype(float), y_train.astype(int), X_test.astype(float), y_test.astype(int)


def partition_clients(n_samples: int, n_clients: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    splits = np.array_split(indices, n_clients)
    return [np.array(split, dtype=int) for split in splits]


def sample_subspace_projector(d: int, r: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    # Sample d x r with orthonormal columns via QR decomposition
    A = rng.normal(0, 1, size=(d, r))
    Q, _ = np.linalg.qr(A)
    P = Q[:, :r]  # d x r
    Pi = P @ P.T  # d x d
    return P, Pi


def client_update(
    X: np.ndarray,
    y: np.ndarray,
    client_idx: np.ndarray,
    W_global: np.ndarray,
    Pi_t: np.ndarray,
    P_t: np.ndarray,
    tau: int,
    eta: float,
    mu: float,
    B: int,
    v_prev: np.ndarray,
    rng: np.random.Generator,
    compress_uplink: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform local client update with optional one-sided momentum projection and uplink compression.

    Returns:
        delta_to_server: either full ambient delta (d x K) or compressed coefficients (r x K) if compress_uplink=True
        v_last: momentum state at end of local steps (d x K)
    """
    d, K = W_global.shape
    W_local = W_global.copy()

    # Initialize momentum
    if v_prev is None:
        v = np.zeros_like(W_local)
    else:
        # Momentum projection at block start if projector provided
        v = (Pi_t @ v_prev) if Pi_t is not None else v_prev.copy()

    # Local SGD updates
    client_X = X[client_idx]
    client_y = y[client_idx]
    n_client = client_X.shape[0]
    for _ in range(tau):
        if B >= n_client:
            # Use full client data if batch size >= client size
            Xb = client_X
            yb = client_y
        else:
            sel = rng.choice(n_client, size=B, replace=False)
            Xb = client_X[sel]
            yb = client_y[sel]
        grad = compute_batch_grad(Xb, yb, W_local)

        if Pi_t is not None:
            # One-sided projected momentum
            v = mu * v + (Pi_t @ grad)
        else:
            v = mu * v + grad

        W_local = W_local - eta * v

    delta_full = W_local - W_global  # (d, K)

    if compress_uplink and P_t is not None:
        # Send coefficients in subspace: r x K
        delta_coeffs = P_t.T @ delta_full
        return delta_coeffs, v
    else:
        return delta_full, v


def run_experiment(
    method: str,
    out_dir: str,
    rounds: int,
    n_clients: int,
    client_fraction: float,
    local_steps: int,
    lr: float,
    mu: float,
    batch_size: int,
    d: int,
    r: int,
    K: int,
    n_train: int,
    n_test: int,
    seed: int,
) -> Dict[str, Dict[str, List[float]]]:
    # Generate data
    X_train, y_train, X_test, y_test = generate_data(n_train, n_test, d, K, noise_std=0.5, seed=seed)

    # Initialize global model weights
    W_global = np.zeros((d, K), dtype=float)

    # Client partitions
    client_indices = partition_clients(X_train.shape[0], n_clients, seed=seed)
    # Momentum state per client (d x K), initially None
    v_states: Dict[int, np.ndarray] = {i: None for i in range(n_clients)}

    # Metrics over rounds
    train_losses: List[float] = []
    test_accuracies: List[float] = []
    uplink_floats: List[float] = []
    downlink_floats: List[float] = []
    total_floats: List[float] = []

    rng = np.random.default_rng(seed)

    # Determine if we use subspace and uplink compression
    use_subspace = (method.lower() == "sfedavg")
    compress_uplink = use_subspace  # model deltas sent as subspace coefficients for SFedAvg
    m_per_round = max(1, int(round(client_fraction * n_clients)))
    # Track cumulative communication to reflect efficiency progress over rounds
    cum_uplink = 0.0
    cum_downlink = 0.0
    cum_total = 0.0

    for t in range(rounds):
        P_t, Pi_t = (None, None)
        if use_subspace:
            if r <= 0 or r > d:
                raise ValueError(f"Invalid subspace dimension r={r}, must be in [1, d={d}]")
            P_t, Pi_t = sample_subspace_projector(d, r, rng)

        # Sample client subset
        subset = rng.choice(n_clients, size=m_per_round, replace=False)

        # Communication accounting (downlink)
        # Server sends theta (d*K) to each selected client
        down_theta = m_per_round * d * K
        # For SFedAvg, also send P_t (d*r) to each client
        down_proj = m_per_round * d * r if use_subspace else 0
        downlink = down_theta + down_proj

        # Local updates and uplink
        deltas_received = []
        for i in subset:
            delta_to_server, v_last = client_update(
                X_train,
                y_train,
                client_indices[i],
                W_global,
                Pi_t,
                P_t,
                tau=local_steps,
                eta=lr,
                mu=mu if use_subspace else 0.0,  # FedAvg default: no momentum
                B=batch_size,
                v_prev=v_states[i],
                rng=rng,
                compress_uplink=compress_uplink,
            )
            v_states[i] = v_last
            if use_subspace:
                # Reconstruct ambient delta from coefficients
                delta_full = P_t @ delta_to_server  # (d, K)
            else:
                delta_full = delta_to_server
            deltas_received.append(delta_full)

        # Aggregate
        if deltas_received:
            agg_delta = sum(deltas_received) / len(deltas_received)
            W_global = W_global + agg_delta

        # Communication accounting (uplink)
        uplink = m_per_round * (r * K if use_subspace else d * K)

        cum_uplink += float(uplink)
        cum_downlink += float(downlink)
        cum_total += float(uplink + downlink)

        uplink_floats.append(cum_uplink)
        downlink_floats.append(cum_downlink)
        total_floats.append(cum_total)

        # Evaluate global model
        train_losses.append(compute_loss(X_train, y_train, W_global))
        test_accuracies.append(compute_accuracy(X_test, y_test, W_global))

    # Prepare results dict with means and stds (stds set to zeros as single-run)
    zeros_std = [0.0] * rounds
    results = {
        "train_loss": {
            "means": train_losses,
            "stds": zeros_std,
        },
        "test_accuracy": {
            "means": test_accuracies,
            "stds": zeros_std,
        },
        "uplink_floats": {
            "means": uplink_floats,
            "stds": zeros_std,
        },
        "downlink_floats": {
            "means": downlink_floats,
            "stds": zeros_std,
        },
        "total_floats": {
            "means": total_floats,
            "stds": zeros_std,
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="FedAvg vs SFedAvg experiment (softmax regression)")
    parser.add_argument('--out_dir', type=str, required=False, default=os.environ.get("OUT_DIR", "run_default"), help="Output directory to save final_info.json")
    # Optional configuration
    parser.add_argument('--method', type=str, default="sfedavg", choices=["fedavg", "sfedavg"], help="Which method to run")
    parser.add_argument('--rounds', type=int, default=30, help="Number of federated rounds T")
    parser.add_argument('--clients', type=int, default=20, help="Number of clients N")
    parser.add_argument('--client_fraction', type=float, default=0.2, help="Client fraction C sampled each round")
    parser.add_argument('--local_steps', type=int, default=2, help="Local steps per client tau")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate eta")
    parser.add_argument('--mu', type=float, default=0.9, help="Momentum coefficient mu (used when method=sfedavg)")
    parser.add_argument('--batch_size', type=int, default=64, help="Minibatch size B")
    parser.add_argument('--dim', type=int, default=30, help="Feature dimension d")
    parser.add_argument('--subspace_dim', type=int, default=10, help="Subspace dimension r (for SFedAvg)")
    parser.add_argument('--classes', type=int, default=10, help="Number of classes K")
    parser.add_argument('--train_samples', type=int, default=5000, help="Number of training samples")
    parser.add_argument('--test_samples', type=int, default=2000, help="Number of test samples")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    results = run_experiment(
        method=args.method,
        out_dir=args.out_dir,
        rounds=args.rounds,
        n_clients=args.clients,
        client_fraction=args.client_fraction,
        local_steps=args.local_steps,
        lr=args.lr,
        mu=args.mu,
        batch_size=args.batch_size,
        d=args.dim,
        r=args.subspace_dim,
        K=args.classes,
        n_train=args.train_samples,
        n_test=args.test_samples,
        seed=args.seed,
    )

    # Save results in the required format
    with open(f"{args.out_dir}/final_info.json", "w") as f:
        json.dump(results, f, indent=2)

    # Optional: print a brief summary
    print(f"Saved results to: {args.out_dir}/final_info.json")
    print(f"Final train loss: {results['train_loss']['means'][-1]:.4f}, test accuracy: {results['test_accuracy']['means'][-1]:.4f}")


if __name__ == "__main__":
    main()
