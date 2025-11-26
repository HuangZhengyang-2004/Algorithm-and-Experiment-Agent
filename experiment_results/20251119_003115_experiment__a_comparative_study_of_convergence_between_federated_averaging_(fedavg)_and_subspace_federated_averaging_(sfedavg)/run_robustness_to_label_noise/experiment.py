# Experiment file - Federated Averaging vs Subspace Federated Averaging on sklearn digits

import argparse
import json
import os
from typing import Dict, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def softmax(logits: np.ndarray) -> np.ndarray:
    # logits: (n_samples, K)
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    Y = np.zeros((y.shape[0], K), dtype=np.float64)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def compute_loss_and_grad(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute cross-entropy loss and gradient for softmax linear classifier.
    X: (n_samples, D) augmented with bias
    Y: (n_samples, K) one-hot labels
    W: (D, K)
    Returns: loss (float), grad_W (D, K)
    """
    n = X.shape[0]
    logits = X @ W  # (n, K)
    P = softmax(logits)  # (n, K)
    # Cross-entropy loss
    eps = 1e-12
    loss = -np.sum(Y * np.log(P + eps)) / n
    # Gradient
    grad = (X.T @ (P - Y)) / n  # (D, K)
    return loss, grad


def accuracy(X: np.ndarray, y: np.ndarray, W: np.ndarray) -> float:
    logits = X @ W
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def sample_subspace(D: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample an orthonormal basis P in R^{D x r} via QR decomposition.
    """
    A = rng.normal(size=(D, r))
    # QR decomposition to get orthonormal columns
    Q, R = np.linalg.qr(A, mode='reduced')
    # Ensure deterministic sign (optional): flip columns to have positive diagonal in R
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs
    return Q  # columns orthonormal


def project_with_P(v: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Project vector v onto subspace spanned by columns of P: Pi v = P (P^T v)
    """
    return P @ (P.T @ v)


def prepare_data(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load digits dataset, normalize features, augment bias, and split into train/test.
    Returns X_train, y_train, X_test, y_test, K
    """
    data = load_digits()
    X = data.images.reshape((-1, 64)).astype(np.float64)  # 8x8 images flattened
    y = data.target.astype(int)
    K = len(np.unique(y))

    # Normalize features: zero-mean, unit-variance per feature
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Augment bias term
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    X_aug = np.hstack([X, ones])  # (n, 65)

    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y, test_size=0.2, random_state=seed, stratify=y
    )
    return X_train, y_train, X_test, y_test, K


def partition_clients(X: np.ndarray, y: np.ndarray, num_clients: int, seed: int = 42) -> List[Dict[str, np.ndarray]]:
    """
    Evenly partition the training data across clients.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    clients = []
    for split in splits:
        clients.append({
            "X": X[split],
            "y": y[split]
        })
    return clients


class FederatedRunner:
    def __init__(
        self,
        algo: str,
        num_clients: int,
        client_frac: float,
        rounds: int,
        local_steps: int,
        lr: float,
        momentum: float,
        batch_size: int,
        subspace_dim: int,
        seed: int = 42
    ):
        """
        Initialize federated experiment.
        algo: 'fedavg' or 'sfedavg'
        """
        assert algo in ("fedavg", "sfedavg"), "algo must be 'fedavg' or 'sfedavg'"
        assert 0.0 < client_frac <= 1.0, "client_frac must be in (0,1]"
        self.algo = algo
        self.num_clients = num_clients
        self.client_frac = client_frac
        self.rounds = rounds
        self.local_steps = local_steps
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.subspace_dim = subspace_dim
        self.seed = seed

        # Data
        X_train, y_train, X_test, y_test, K = prepare_data(seed=seed)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.K = K

        # Partition clients
        self.clients = partition_clients(self.X_train, self.y_train, num_clients, seed=seed)

        # Model dimensions
        D_feat = self.X_train.shape[1]  # includes bias
        self.D = D_feat * K  # flattened parameter vector length

        # Global model weights W (D_feat, K) flattened to (D,)
        rng = np.random.default_rng(seed)
        W_init = rng.normal(scale=0.01, size=(D_feat, K))
        self.w_global = W_init.reshape(-1).copy()

        # Momentum state per client (persisted across rounds)
        self.client_momentum: Dict[int, np.ndarray] = {}

        # RNG for sampling
        self.rng = rng

    def w_matrix(self, w_flat: np.ndarray) -> np.ndarray:
        return w_flat.reshape(self.X_train.shape[1], self.K)

    def evaluate_metrics(self) -> Tuple[float, float, float]:
        """
        Evaluate train loss, test loss, and test accuracy with the current global model.
        """
        W = self.w_matrix(self.w_global)
        Y_train = one_hot(self.y_train, self.K)
        train_loss, _ = compute_loss_and_grad(self.X_train, Y_train, W)
        Y_test = one_hot(self.y_test, self.K)
        test_loss, _ = compute_loss_and_grad(self.X_test, Y_test, W)
        test_acc = accuracy(self.X_test, self.y_test, W)
        return train_loss, test_loss, test_acc

    def client_update(self, client_id: int, w_start: np.ndarray, P: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform local updates on a client and return (delta, v_last).
        For SFedAvg: gradients and momentum are projected via P.
        For FedAvg: full gradients and momentum.
        """
        client = self.clients[client_id]
        Xc = client["X"]
        yc = client["y"]
        Yc = one_hot(yc, self.K)

        W_local = self.w_matrix(w_start).copy()
        w_local = W_local.reshape(-1)  # flattened for momentum operations

        # Initialize momentum
        v_prev = self.client_momentum.get(client_id, None)
        if self.algo == "sfedavg":
            if v_prev is None:
                v = np.zeros_like(w_start)
            else:
                # Project previous momentum to current subspace
                v = project_with_P(v_prev, P)
        else:
            # FedAvg: use previous momentum as-is (or zeros)
            v = np.zeros_like(w_start) if v_prev is None else v_prev.copy()

        # Local SGD with (projected) momentum
        n_samples = Xc.shape[0]
        for s in range(self.local_steps):
            # Sample a minibatch
            if n_samples >= self.batch_size:
                idx = self.rng.choice(n_samples, size=self.batch_size, replace=False)
            else:
                idx = self.rng.choice(n_samples, size=self.batch_size, replace=True)
            Xb = Xc[idx]
            Yb = Yc[idx]

            W_mat = w_local.reshape(W_local.shape)
            _, grad_W = compute_loss_and_grad(Xb, Yb, W_mat)
            grad_flat = grad_W.reshape(-1)

            if self.algo == "sfedavg":
                grad_used = project_with_P(grad_flat, P)
            else:
                grad_used = grad_flat

            # Momentum update and parameter step
            v = self.momentum * v + grad_used
            w_local = w_local - self.lr * v

        delta = w_local - w_start
        return delta, v

    def run(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Run federated training and collect metrics over rounds.
        Returns a dict with metric_name -> {"means": [...], "stds": [...]}
        """
        T = self.rounds
        N = self.num_clients
        m_per_round = max(1, int(round(self.client_frac * N)))
        D = self.D

        train_losses = []
        test_losses = []
        test_accs = []
        comm_bytes = []
        total_comm = 0

        for t in range(T):
            # Sample subspace for SFedAvg
            if self.algo == "sfedavg":
                r = min(self.subspace_dim, D)
                P_t = sample_subspace(D, r, self.rng)
            else:
                P_t = None

            # Sample clients
            selected = self.rng.choice(N, size=m_per_round, replace=False)

            # Communication accounting (float64 assumed)
            if self.algo == "sfedavg":
                bytes_server_to_clients = m_per_round * (D + D * P_t.shape[1]) * 8
            else:
                bytes_server_to_clients = m_per_round * D * 8
            bytes_clients_to_server = m_per_round * D * 8
            comm_this_round = int(bytes_server_to_clients + bytes_clients_to_server)
            total_comm += comm_this_round
            comm_bytes.append(total_comm)

            # Broadcast w_global and (optionally) subspace, collect updates
            delta_sum = np.zeros_like(self.w_global)
            for cid in selected:
                delta_i, v_last = self.client_update(cid, self.w_global, P=P_t)
                delta_sum += delta_i
                # Store momentum for the client
                self.client_momentum[cid] = v_last

            # Aggregate
            self.w_global = self.w_global + (delta_sum / m_per_round)

            # Evaluate metrics after aggregation
            tr_loss, te_loss, te_acc = self.evaluate_metrics()
            train_losses.append(tr_loss)
            test_losses.append(te_loss)
            test_accs.append(te_acc)

        # Package results with stds (zeros for single-run)
        zeros = [0.0] * T
        results = {
            "train_loss": {
                "means": train_losses,
                "stds": zeros
            },
            "test_loss": {
                "means": test_losses,
                "stds": zeros
            },
            "test_accuracy": {
                "means": test_accs,
                "stds": zeros
            },
            "communication_bytes": {
                "means": comm_bytes,
                "stds": zeros
            }
        }
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--algo', type=str, choices=['fedavg', 'sfedavg'], default='fedavg', help='Algorithm to run')
    parser.add_argument('--rounds', type=int, default=20, help='Number of federated rounds T')
    parser.add_argument('--local_steps', type=int, default=5, help='Local steps per client tau')
    parser.add_argument('--client_frac', type=float, default=0.5, help='Fraction of clients per round C')
    parser.add_argument('--num_clients', type=int, default=10, help='Total number of clients N')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate eta')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum mu')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size B')
    parser.add_argument('--r', type=int, default=128, help='Subspace dimension r for SFedAvg')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    runner = FederatedRunner(
        algo=args.algo,
        num_clients=args.num_clients,
        client_frac=args.client_frac,
        rounds=args.rounds,
        local_steps=args.local_steps,
        lr=args.lr,
        momentum=args.momentum,
        batch_size=args.batch_size,
        subspace_dim=args.r,
        seed=args.seed
    )

    results = runner.run()

    # Save results in the required format
    with open(f"{args.out_dir}/final_info.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
