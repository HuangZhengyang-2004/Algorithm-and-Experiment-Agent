# Experiment: FedAvg vs SFedAvg-Golore (One-Sided Random Subspace + Momentum Projection)
# Complete, self-contained implementation using a softmax linear classifier on real data.
# Tries to load MNIST via torchvision; falls back to synthetic multiclass data if unavailable.
# Saves metrics to {out_dir}/final_info.json in the required format.

import argparse
import json
import os
import math
import numpy as np
from typing import Dict, Tuple, List, Optional

np.set_printoptions(precision=6, suppress=True)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


class SoftmaxLinearModel:
    """
    Multiclass softmax linear classifier: scores = X @ W, shape W=(d_features, K_classes)
    """
    def __init__(self, input_dim: int, num_classes: int, dtype=np.float32, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Small random init for stability
        self.W = (rng.standard_normal((input_dim, num_classes)).astype(dtype)) * 0.01
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dtype = dtype

    def copy(self) -> "SoftmaxLinearModel":
        m = SoftmaxLinearModel(self.input_dim, self.num_classes, dtype=self.dtype)
        m.W = self.W.copy()
        return m

    def to_vec(self) -> np.ndarray:
        return self.W.reshape(-1)

    def from_vec(self, vec: np.ndarray):
        self.W = vec.reshape(self.input_dim, self.num_classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W
        return softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        # Cross-entropy loss and gradient
        n = X.shape[0]
        probs = self.predict_proba(X)
        y_oh = one_hot(y, self.num_classes)
        # Avoid log(0)
        eps = 1e-12
        loss = -np.log(np.clip((probs * y_oh).sum(axis=1), eps, None)).mean()
        # Gradient: X^T (probs - y_one_hot) / n
        grad = (X.T @ (probs - y_oh)) / float(n)
        return float(loss), grad.astype(self.dtype)


def try_load_mnist(limit_train: int = 20000, limit_test: int = 5000, seed: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Try to load MNIST via torchvision. Returns (X_train, y_train, X_test, y_test) as float32.
    Falls back to None if torchvision/torch not available or download fails.
    """
    try:
        import torch
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        data_root = os.path.join(os.getcwd(), "data_mnist")
        train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

        # Convert to numpy
        def ds_to_np(ds, limit):
            n = min(len(ds), limit) if limit is not None else len(ds)
            X = np.zeros((n, 28 * 28), dtype=np.float32)
            y = np.zeros((n,), dtype=np.int64)
            for i in range(n):
                img, label = ds[i]
                X[i] = img.numpy().reshape(-1).astype(np.float32)
                y[i] = int(label)
            return X, y

        X_train, y_train = ds_to_np(train_ds, limit_train)
        X_test, y_test = ds_to_np(test_ds, limit_test)

        # Normalize to zero mean, unit variance per feature
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-6
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        return X_train, y_train.astype(np.int64), X_test, y_test.astype(np.int64)
    except Exception:
        return None


def make_synthetic_multiclass(n_samples: int = 12000, n_features: int = 100, n_classes: int = 10, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic, learnable multiclass dataset.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    # True weights
    W_true = rng.standard_normal((n_features, n_classes)).astype(np.float32)
    logits = X @ W_true + 0.5 * rng.standard_normal((n_samples, n_classes)).astype(np.float32)
    y = logits.argmax(axis=1).astype(np.int64)

    # Train/test split
    idx = rng.permutation(n_samples)
    n_train = int(n_samples * 0.8)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Standardize features using train stats
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, y_train, X_test, y_test


def partition_clients(n_samples: int, num_clients: int, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    splits = np.array_split(idx, num_clients)
    return [split for split in splits]


def sample_subspace_projector(d: int, r: int, seed: int) -> np.ndarray:
    """
    Sample P in St(d, r) (orthonormal columns) using QR; returns P (d x r).
    Projection Pi vec implemented as P @ (P^T @ vec) to avoid dxd memory.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, r))
    # QR decomposition
    Q, _ = np.linalg.qr(A)
    P = Q[:, :r]
    return P.astype(np.float32)


def project_vec(vec: np.ndarray, P: np.ndarray) -> np.ndarray:
    # One-sided projection: Pi vec = P (P^T vec)
    return P @ (P.T @ vec)


def evaluate(model_vec: np.ndarray, input_dim: int, num_classes: int, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    m = SoftmaxLinearModel(input_dim, num_classes)
    m.from_vec(model_vec)
    train_loss, _ = m.loss_and_grad(X_train, y_train)
    test_acc = float((m.predict(X_test) == y_test).mean())
    return train_loss, test_acc


def client_update(local_vec: np.ndarray,
                  X: np.ndarray,
                  y: np.ndarray,
                  tau: int,
                  eta: float,
                  mu: float,
                  batch_size: int,
                  input_dim: int,
                  num_classes: int,
                  momentum_init: Optional[np.ndarray],
                  P: Optional[np.ndarray],
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform local updates for one client and return (updated_vec, v_last).
    If P is provided, use projected gradients/momentum. Otherwise use full gradients.
    """
    # Initialize momentum (with optional momentum projection at block start)
    d = local_vec.shape[0]
    if momentum_init is None:
        v = np.zeros(d, dtype=np.float32)
    else:
        v = momentum_init.astype(np.float32)
        if P is not None:
            v = project_vec(v, P)

    # Local loop
    model = SoftmaxLinearModel(input_dim, num_classes)
    model.from_vec(local_vec.copy())

    n = X.shape[0]
    for s in range(tau):
        if n == 0:
            break
        # Sample minibatch with replacement if needed
        if batch_size >= n:
            batch_idx = rng.integers(0, n, size=batch_size)
        else:
            batch_idx = rng.choice(n, size=batch_size, replace=False)
        Xb = X[batch_idx]
        yb = y[batch_idx]

        # Compute gradient
        loss, grad = model.loss_and_grad(Xb, yb)
        grad_vec = grad.reshape(-1).astype(np.float32)

        # Apply projection if provided (one-sided momentum projection)
        if P is not None:
            g_proj = project_vec(grad_vec, P)
            v = mu * v + g_proj
        else:
            v = mu * v + grad_vec

        # Update local parameters
        local_vec = local_vec - eta * v
        model.from_vec(local_vec)  # keep model in sync

    return local_vec, v


def run_federated(X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  num_clients: int,
                  client_fraction: float,
                  rounds: int,
                  tau: int,
                  eta: float,
                  mu: float,
                  batch_size: int,
                  subspace_dim: int,
                  seed: int,
                  algo: str,
                  input_dim: int,
                  num_classes: int) -> Tuple[List[float], List[float], List[float], np.ndarray]:
    """
    Run either 'fedavg' or 'sfedavg' and return (train_losses, test_accuracies, comm_bytes, final_model_vec).
    """
    rng = np.random.default_rng(seed)
    # Partition data among clients
    client_indices = partition_clients(X_train.shape[0], num_clients, seed=seed)
    # Initialize global model
    global_model = SoftmaxLinearModel(input_dim, num_classes, seed=seed)
    theta = global_model.to_vec().astype(np.float32)
    d_params = theta.shape[0]

    # Momentum per client across rounds (for both algos; projection only for SFedAvg)
    momentum_prev: Dict[int, np.ndarray] = {}

    train_losses: List[float] = []
    test_accuracies: List[float] = []
    comm_bytes: List[float] = []
    cum_comm = 0.0

    # Bytes per float
    bytes_per_float = 4.0  # float32

    for t in range(rounds):
        # Sample client subset
        m = max(1, int(round(client_fraction * num_clients)))
        selected = rng.choice(num_clients, size=m, replace=False).tolist()

        # For SFedAvg: sample subspace P_t once per round
        P_t = None
        if algo == "sfedavg":
            r = max(1, min(subspace_dim, d_params))
            P_t = sample_subspace_projector(d_params, r, seed=seed + t)

        # Client updates
        deltas = []
        new_momentum_prev = {}
        for i in selected:
            idx = client_indices[i]
            Xi = X_train[idx]
            yi = y_train[idx]
            # Local model starts from global
            local_vec = theta.copy()
            v_init = momentum_prev.get(i, None)
            updated_vec, v_last = client_update(
                local_vec=local_vec,
                X=Xi,
                y=yi,
                tau=tau,
                eta=eta,
                mu=mu,
                batch_size=batch_size,
                input_dim=input_dim,
                num_classes=num_classes,
                momentum_init=v_init if algo in ("fedavg", "sfedavg") else None,
                P=P_t if algo == "sfedavg" else None,
                rng=rng
            )
            delta_i = updated_vec - theta
            deltas.append(delta_i)
            new_momentum_prev[i] = v_last

        # Aggregate
        if deltas:
            mean_delta = np.mean(np.stack(deltas, axis=0), axis=0)
            theta = (theta + mean_delta).astype(np.float32)

        # Update momentum memory only for selected clients
        for i in selected:
            momentum_prev[i] = new_momentum_prev[i]

        # Evaluate
        train_loss, test_acc = evaluate(theta, input_dim, num_classes, X_train, y_train, X_test, y_test)
        train_losses.append(float(train_loss))
        test_accuracies.append(float(test_acc))

        # Communication accounting:
        # Server sends theta to each selected client, client sends delta back.
        # For SFedAvg, server additionally sends P_t (d x r) to each selected client.
        bytes_send = m * d_params * bytes_per_float
        bytes_recv = m * d_params * bytes_per_float
        if algo == "sfedavg" and P_t is not None:
            r = P_t.shape[1]
            bytes_send += m * d_params * r * bytes_per_float  # sending P_t
        round_bytes = bytes_send + bytes_recv
        cum_comm += float(round_bytes)
        comm_bytes.append(cum_comm)

    return train_losses, test_accuracies, comm_bytes, theta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated rounds T')
    parser.add_argument('--local_steps', type=int, default=2, help='Local steps tau')
    parser.add_argument('--client_fraction', type=float, default=0.2, help='Fraction C of clients per round')
    parser.add_argument('--stepsize', type=float, default=0.1, help='Learning rate eta')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum mu')
    parser.add_argument('--batch_size', type=int, default=64, help='Local minibatch size B')
    parser.add_argument('--num_clients', type=int, default=25, help='Total number of clients N')
    parser.add_argument('--subspace_dim', type=int, default=128, help='Subspace dimension r for SFedAvg')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default='mnist_or_synth', help='Dataset: tries mnist, else synth')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    data = try_load_mnist(limit_train=20000, limit_test=5000, seed=args.seed) if args.dataset.lower().startswith('mnist') else None
    if data is None:
        # Fallback to synthetic dataset with similar dimensions
        X_train, y_train, X_test, y_test = make_synthetic_multiclass(
            n_samples=12000, n_features=100, n_classes=10, seed=args.seed
        )
        input_dim = X_train.shape[1]
        num_classes = int(y_train.max()) + 1
    else:
        X_train, y_train, X_test, y_test = data
        input_dim = X_train.shape[1]
        num_classes = int(y_train.max()) + 1

    # Validate args
    num_clients = max(1, args.num_clients)
    client_fraction = min(1.0, max(0.0, args.client_fraction))
    rounds = max(1, args.rounds)
    tau = max(1, args.local_steps)
    eta = float(args.stepsize)
    mu = float(args.momentum)
    batch_size = max(1, args.batch_size)
    subspace_dim = max(1, args.subspace_dim)
    seed = int(args.seed)

    # Run FedAvg
    fedavg_train, fedavg_test, fedavg_comm, theta_fedavg = run_federated(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        num_clients=num_clients, client_fraction=client_fraction, rounds=rounds,
        tau=tau, eta=eta, mu=mu, batch_size=batch_size, subspace_dim=subspace_dim,
        seed=seed, algo="fedavg", input_dim=input_dim, num_classes=num_classes
    )

    # Run SFedAvg-Golore (one-sided projection + momentum projection)
    sfed_train, sfed_test, sfed_comm, theta_sfed = run_federated(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        num_clients=num_clients, client_fraction=client_fraction, rounds=rounds,
        tau=tau, eta=eta, mu=mu, batch_size=batch_size, subspace_dim=subspace_dim,
        seed=seed, algo="sfedavg", input_dim=input_dim, num_classes=num_classes
    )

    # Prepare results
    zeros_like = lambda arr: [0.0] * len(arr)
    results = {
        "FedAvg/train_loss": {
            "means": [float(x) for x in fedavg_train],
            "stds": zeros_like(fedavg_train)
        },
        "FedAvg/test_accuracy": {
            "means": [float(x) for x in fedavg_test],
            "stds": zeros_like(fedavg_test)
        },
        "FedAvg/communication_bytes": {
            "means": [float(x) for x in fedavg_comm],
            "stds": zeros_like(fedavg_comm)
        },
        "SFedAvg/train_loss": {
            "means": [float(x) for x in sfed_train],
            "stds": zeros_like(sfed_train)
        },
        "SFedAvg/test_accuracy": {
            "means": [float(x) for x in sfed_test],
            "stds": zeros_like(sfed_test)
        },
        "SFedAvg/communication_bytes": {
            "means": [float(x) for x in sfed_comm],
            "stds": zeros_like(sfed_comm)
        }
    }

    # Save results directly to {out_dir}/final_info.json
    with open(f"{args.out_dir}/final_info.json", "w") as f:
        json.dump(results, f, indent=2)

    # Optional: print a brief summary
    print(f"Saved results to {args.out_dir}/final_info.json")
    print(f"FedAvg final loss: {fedavg_train[-1]:.4f}, accuracy: {fedavg_test[-1]:.4f}")
    print(f"SFedAvg final loss: {sfed_train[-1]:.4f}, accuracy: {sfed_test[-1]:.4f}")


if __name__ == "__main__":
    main()
