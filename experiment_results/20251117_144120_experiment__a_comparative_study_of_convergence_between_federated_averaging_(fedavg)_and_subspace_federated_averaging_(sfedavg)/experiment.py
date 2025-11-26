import argparse
import json
import os
from typing import Dict, Tuple, List

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # Clip z to avoid overflow
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Binary logistic loss (cross-entropy) computed in a numerically stable way.

    Loss per sample: softplus(-z) + (1 - y) * z, where z = X @ theta.
    """
    z = X @ theta
    # softplus(x) = log(1 + exp(x)) = logaddexp(0, x)
    loss = np.logaddexp(0.0, -z) + (1.0 - y) * z
    return float(np.mean(loss))


def logistic_grad(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Gradient of binary cross-entropy loss with sigmoid link:
    grad = X.T @ (sigmoid(X @ theta) - y) / n
    """
    n = X.shape[0]
    preds = sigmoid(X @ theta)
    grad = (X.T @ (preds - y)) / float(n)
    return grad


def accuracy(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Compute classification accuracy for binary logistic regression."""
    preds = sigmoid(X @ theta) >= 0.5
    return float(np.mean(preds.astype(np.float32) == y.astype(np.float32)))


def generate_synthetic_logistic_data(
    n_clients: int,
    samples_per_client: int,
    dim: int,
    test_samples: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate federated binary classification data with a shared ground-truth model.
    Each client receives i.i.d. samples from the same distribution.

    Returns:
        X_clients: list of X arrays per client (n_i x d, float32)
        y_clients: list of y arrays per client (n_i, float32; values in {0,1})
        X_test: test features (n_test x d)
        y_test: test labels (n_test,)
        w_true: ground-truth weight vector used to generate labels
    """
    rng = np.random.default_rng(seed)
    # Ground-truth weights
    w_true = rng.normal(0, 1, size=(dim,)).astype(np.float32)
    w_true /= max(1e-6, np.linalg.norm(w_true))

    X_clients = []
    y_clients = []
    for _ in range(n_clients):
        # Features
        X = rng.normal(0, 1, size=(samples_per_client, dim)).astype(np.float32)
        # Add mild client-specific shift for heterogeneity
        shift = rng.normal(0, 0.25, size=(dim,)).astype(np.float32)
        X = X + shift  # feature shift per client

        logits = X @ w_true
        probs = sigmoid(logits)
        y = rng.uniform(size=(samples_per_client,)) < probs
        y = y.astype(np.float32)

        X_clients.append(X)
        y_clients.append(y)

    # Test set from the same distribution (no shift)
    X_test = rng.normal(0, 1, size=(test_samples, dim)).astype(np.float32)
    y_test = (rng.uniform(size=(test_samples,)) < sigmoid(X_test @ w_true)).astype(np.float32)

    return X_clients, y_clients, X_test, y_test, w_true


def sample_subspace_projector(dim: int, subspace_dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample an orthonormal basis P in R^{d x r} via QR decomposition of a random Gaussian matrix,
    then form the projector Pi = P P^T.

    Returns:
        Pi (d x d) as float32.
    """
    if subspace_dim >= dim:
        # Full space projector (identity)
        return np.eye(dim, dtype=np.float32)

    A = rng.normal(0, 1, size=(dim, subspace_dim)).astype(np.float32)
    # QR for orthonormal columns
    Q, _ = np.linalg.qr(A, mode="reduced")
    Q = Q.astype(np.float32)
    Pi = Q @ Q.T
    return Pi.astype(np.float32)


def project(vec: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """Project a vector using projector Pi. If Pi is None, return vec."""
    if Pi is None:
        return vec
    return Pi @ vec


def client_update(
    X: np.ndarray,
    y: np.ndarray,
    theta_t: np.ndarray,
    Pi_t: np.ndarray,
    v_prev: np.ndarray,
    tau: int,
    eta: float,
    mu: float,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform local client update with one-sided projected momentum.

    Args:
        X, y: local dataset
        theta_t: global weights at start of round (float32)
        Pi_t: projector matrix (d x d) or None (for FedAvg)
        v_prev: previous momentum vector (d,) or None
        tau: local steps
        eta: stepsize
        mu: momentum coefficient
        batch_size: minibatch size
        rng: random generator

    Returns:
        delta_i: theta_i_tau - theta_t
        v_tau: final momentum to store for next round
    """
    d = theta_t.shape[0]
    theta_i = theta_t.copy()
    if v_prev is None:
        v = np.zeros(d, dtype=np.float32)
    else:
        # Momentum projection at block start
        v = project(v_prev, Pi_t)

    n = X.shape[0]
    for _ in range(tau):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        Xb = X[idx]
        yb = y[idx]

        g = logistic_grad(Xb, yb, theta_i).astype(np.float32)
        g_proj = project(g, Pi_t)

        v = mu * v + g_proj
        theta_i = theta_i - eta * v

    delta_i = (theta_i - theta_t).astype(np.float32)
    return delta_i, v


def federated_run(
    algo: str,
    X_clients: List[np.ndarray],
    y_clients: List[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int,
    client_frac: float,
    local_steps: int,
    lr: float,
    momentum: float,
    batch_size: int,
    subspace_dim: int,
    seed: int,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Execute federated training for FedAvg or SFedAvg.

    Returns:
        results dict with arrays per round for metrics.
    """
    rng = np.random.default_rng(seed)
    n_clients = len(X_clients)
    d = X_clients[0].shape[1]

    # Initialize global model near zero
    theta = rng.normal(0, 0.01, size=(d,)).astype(np.float32)

    # Momentum state per client across rounds
    v_prev_dict: Dict[int, np.ndarray] = {}

    m = max(1, int(np.ceil(client_frac * n_clients)))

    train_loss_means: List[float] = []
    train_loss_stds: List[float] = []
    test_acc_means: List[float] = []
    test_acc_stds: List[float] = []
    comm_cumulative_mb: List[float] = []
    comm_cumulative = 0  # bytes

    for t in range(rounds):
        # One-sided random subspace per round for SFedAvg; identity for FedAvg
        if algo.lower() == "sfedavg":
            Pi_t = sample_subspace_projector(d, subspace_dim, rng)
        else:
            Pi_t = None  # full space (no projection)

        # Sample clients
        selected = rng.choice(n_clients, size=m, replace=False)

        # Communication accounting (server -> clients)
        theta_bytes = theta.nbytes
        Pi_bytes = 0 if Pi_t is None else Pi_t.nbytes
        server_to_clients = m * (theta_bytes + Pi_bytes)

        deltas = []
        for i in selected:
            delta_i, v_tau = client_update(
                X_clients[i],
                y_clients[i],
                theta,
                Pi_t,
                v_prev_dict.get(i, None),
                local_steps,
                lr,
                momentum,
                batch_size,
                rng,
            )
            v_prev_dict[i] = v_tau
            deltas.append(delta_i)

        # Aggregate
        if deltas:
            mean_delta = np.mean(np.stack(deltas, axis=0), axis=0).astype(np.float32)
            theta = theta + mean_delta

        # Communication accounting (clients -> server)
        delta_bytes = (theta.nbytes)  # each delta has same shape as theta
        clients_to_server = m * delta_bytes

        comm_cumulative += (server_to_clients + clients_to_server)

        # Metrics after aggregation
        # Per-client train loss for std computation
        client_losses = []
        for i in range(n_clients):
            client_losses.append(logistic_loss(X_clients[i], y_clients[i], theta))
        train_loss_means.append(float(np.mean(client_losses)))
        train_loss_stds.append(float(np.std(client_losses)))

        test_acc = accuracy(X_test, y_test, theta)
        test_acc_means.append(test_acc)
        test_acc_stds.append(0.0)  # single global accuracy, std not applicable

        comm_cumulative_mb.append(comm_cumulative / (1024.0 * 1024.0))

    results = {
        "train_loss": {
            "means": train_loss_means,
            "stds": train_loss_stds,
        },
        "test_accuracy": {
            "means": test_acc_means,
            "stds": test_acc_stds,
        },
        "communication_MB": {
            "means": comm_cumulative_mb,
            "stds": [0.0] * len(comm_cumulative_mb),
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)

    # Optional configuration arguments
    parser.add_argument('--algo', type=str, default='sfedavg', choices=['fedavg', 'sfedavg'],
                        help='Algorithm to run: fedavg or sfedavg')
    parser.add_argument('--rounds', type=int, default=20, help='Number of federated rounds')
    parser.add_argument('--clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--client_frac', type=float, default=0.5, help='Fraction of clients per round')
    parser.add_argument('--local_steps', type=int, default=5, help='Local steps per round')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (step size)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum coefficient')
    parser.add_argument('--batch_size', type=int, default=32, help='Local minibatch size')
    parser.add_argument('--dim', type=int, default=50, help='Model dimension')
    parser.add_argument('--subspace_dim', type=int, default=20, help='Subspace dimension r for SFedAvg')
    parser.add_argument('--samples_per_client', type=int, default=1000, help='Training samples per client')
    parser.add_argument('--test_samples', type=int, default=2000, help='Number of test samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Data generation/loading
    X_clients, y_clients, X_test, y_test, _ = generate_synthetic_logistic_data(
        n_clients=args.clients,
        samples_per_client=args.samples_per_client,
        dim=args.dim,
        test_samples=args.test_samples,
        seed=args.seed,
    )

    # Run federated experiment
    results = federated_run(
        algo=args.algo,
        X_clients=X_clients,
        y_clients=y_clients,
        X_test=X_test,
        y_test=y_test,
        rounds=args.rounds,
        client_frac=args.client_frac,
        local_steps=args.local_steps,
        lr=args.lr,
        momentum=args.momentum,
        batch_size=args.batch_size,
        subspace_dim=args.subspace_dim,
        seed=args.seed,
    )

    # Save results in the required format
    with open(f"{args.out_dir}/final_info.json", "w") as f:
        json.dump(results, f, indent=2)

    # Optional: print a brief summary
    print(f"Saved results to: {args.out_dir}/final_info.json")
    print(f"Final train loss: {results['train_loss']['means'][-1]:.4f}, "
          f"Test accuracy: {results['test_accuracy']['means'][-1]:.4f}, "
          f"Cumulative communication: {results['communication_MB']['means'][-1]:.2f} MB")

if __name__ == "__main__":
    main()
