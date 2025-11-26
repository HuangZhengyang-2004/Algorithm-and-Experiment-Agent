# Experiment: A Comparative Study of Convergence between FedAvg and SFedAvg
# Complete, reproducible federated learning experiment with softmax regression.

import argparse
import json
import os
import sys
import numpy as np
from typing import Dict, Tuple, List

# ----------------------------
# Utilities: softmax model
# ----------------------------

class SoftmaxModel:
    """
    Multiclass softmax regression with optional bias term.
    Parameters are stored as a weight matrix W of shape (d+1, K), where the last
    row corresponds to the bias (when include_bias=True). For federated algorithms,
    parameters are flattened to a vector and reshaped as needed.
    """
    def __init__(self, n_features: int, n_classes: int, include_bias: bool = True, seed: int = 123):
        self.n_features = n_features
        self.n_classes = n_classes
        self.include_bias = include_bias
        self.d = n_features + (1 if include_bias else 0)
        rng = np.random.default_rng(seed)
        # Small random init to avoid symmetry; helps learning
        self.W = rng.normal(loc=0.0, scale=0.01, size=(self.d, n_classes))

    @property
    def Dparam(self) -> int:
        return self.d * self.n_classes

    def copy(self) -> "SoftmaxModel":
        m = SoftmaxModel(self.n_features, self.n_classes, self.include_bias)
        m.W = self.W.copy()
        return m

    def get_params_vector(self) -> np.ndarray:
        return self.W.reshape(-1)

    def set_params_vector(self, theta: np.ndarray) -> None:
        assert theta.shape[0] == self.Dparam, "Parameter vector length mismatch."
        self.W = theta.reshape(self.d, self.n_classes)

    def add_bias(self, X: np.ndarray) -> np.ndarray:
        if not self.include_bias:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([X, ones])

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        # logits: (n, K)
        z = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = self.add_bias(X)
        logits = Xb @ self.W
        return SoftmaxModel._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        Xb = self.add_bias(X)
        logits = Xb @ self.W
        probs = SoftmaxModel._softmax(logits)
        # Cross-entropy
        n = X.shape[0]
        y_idx = (np.arange(n), y)
        # Add small epsilon for numerical stability
        eps = 1e-12
        log_likelihood = -np.log(probs[y_idx] + eps)
        return float(np.mean(log_likelihood))

    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of average cross-entropy w.r.t W.
        Returns gradient as a flattened vector matching parameter vector shape.
        """
        Xb = self.add_bias(X)  # (n, d)
        n = Xb.shape[0]
        logits = Xb @ self.W  # (n, K)
        probs = SoftmaxModel._softmax(logits)  # (n, K)
        # One-hot labels
        Y = np.zeros_like(probs)
        Y[np.arange(n), y] = 1.0
        # dL/dW = X^T (probs - Y) / n
        grad_W = (Xb.T @ (probs - Y)) / n  # (d, K)
        return grad_W.reshape(-1)


# ----------------------------
# Data generation/loading
# ----------------------------

def generate_synthetic_multiclass(n_samples: int = 4000,
                                  n_features: int = 64,
                                  n_classes: int = 10,
                                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a linearly-separable-ish synthetic multiclass dataset by sampling
    features and labels from a known softmax model, then add mild noise.
    This ensures real gradients and actual learning progress without any external downloads.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    # Ground-truth softmax model
    W_true = rng.normal(0, 1.0, size=(n_features, n_classes))
    logits = X @ W_true + rng.normal(0, 0.2, size=(n_samples, n_classes))
    probs = SoftmaxModel._softmax(logits)
    y = np.array([rng.choice(n_classes, p=probs[i]) for i in range(n_samples)], dtype=np.int64)

    # Train/test split
    idx = rng.permutation(n_samples)
    split = int(0.8 * n_samples)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Standardize features using training statistics
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test


def partition_clients(X: np.ndarray, y: np.ndarray, num_clients: int, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Evenly partition the training data across clients.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    Xs = X[idx]
    ys = y[idx]
    splits = np.array_split(np.arange(n), num_clients)
    clients = []
    for s in splits:
        clients.append((Xs[s], ys[s]))
    return clients


def add_label_noise(y: np.ndarray, noise_rate: float = 0.2, n_classes: int = 10, seed: int = None) -> np.ndarray:
    """
    Randomly flip a fraction (noise_rate) of labels to a different class.
    Ensures flipped label is not equal to the original.
    """
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n = y_noisy.shape[0]
    n_flip = int(np.round(noise_rate * n))
    if n_flip <= 0:
        return y_noisy
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    for idx in flip_idx:
        current = int(y_noisy[idx])
        candidates = [c for c in range(n_classes) if c != current]
        y_noisy[idx] = rng.choice(candidates)
    return y_noisy


# ----------------------------
# Federated algorithms
# ----------------------------

def sample_random_subspace(Dparam: int, r: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a random r-dimensional subspace in R^{Dparam} via QR decomposition
    to obtain an orthonormal basis P of shape (Dparam, r),
    and return its projector Pi = P P^T.
    """
    if r <= 0 or r > Dparam:
        raise ValueError(f"Subspace dimension r must be in [1, {Dparam}]")
    A = rng.normal(0, 1, size=(Dparam, r))
    # Orthonormal columns via QR
    Q, _ = np.linalg.qr(A, mode="reduced")  # (Dparam, r)
    P = Q[:, :r]
    Pi = P @ P.T  # (Dparam, Dparam)
    return P, Pi


def client_update(theta_global: np.ndarray,
                  Pi_t: np.ndarray,
                  client_data: Tuple[np.ndarray, np.ndarray],
                  model_template: SoftmaxModel,
                  tau: int,
                  eta: float,
                  mu: float,
                  B: int,
                  momentum_prev: np.ndarray = None,
                  rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    ClientUpdate implementing one-sided projected momentum per round.
    Returns:
      delta: local model delta (theta_local - theta_global)
      v_prev: momentum to store for next round
    """
    X_i, y_i = client_data
    n_i = X_i.shape[0]
    # Local copy of global model parameters
    theta_local = theta_global.copy()
    Dparam = theta_local.shape[0]

    # Momentum initialization with optional projection
    if momentum_prev is None:
        v = np.zeros_like(theta_local)
    else:
        v = Pi_t @ momentum_prev  # Momentum projection at block start

    # Minibatch indices
    if rng is None:
        rng = np.random.default_rng()

    for s in range(tau):
        # Sample a minibatch of size B
        batch_idx = rng.choice(n_i, size=min(B, n_i), replace=False)
        Xb = X_i[batch_idx]
        yb = y_i[batch_idx]
        # Compute gradient from actual data + current local parameters
        model_template.set_params_vector(theta_local)
        g = model_template.gradient(Xb, yb)  # (Dparam,)

        # One-sided projected momentum
        v = mu * v + Pi_t @ g
        theta_local = theta_local - eta * v

    delta = theta_local - theta_global
    v_prev = v.copy()
    return delta, v_prev


def run_federated(algorithm: str,
                  base_model: SoftmaxModel,
                  train_clients: List[Tuple[np.ndarray, np.ndarray]],
                  X_train_full: np.ndarray,
                  y_train_full: np.ndarray,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  T: int,
                  tau: int,
                  C: float,
                  eta: float,
                  mu: float,
                  B: int,
                  seed: int = 0) -> Dict[str, List[float]]:
    """
    Run federated training for either 'fedavg' or 'sfedavg'.
    Returns a dict with per-round metrics: train_loss, test_acc, comm_eff.
    """
    rng = np.random.default_rng(seed)
    N = len(train_clients)
    m = max(1, int(np.round(C * N)))
    Dparam = base_model.Dparam

    # Initialize global parameters (shared initialization across algorithms)
    init_theta = base_model.get_params_vector().copy()
    theta = init_theta.copy()

    # Per-client momentum storage
    client_momentum = [None] * N

    # Metric tracking
    train_losses: List[float] = []
    test_accs: List[float] = []
    comm_efficiency: List[float] = []

    # Fixed comm efficiency per round (dimension ratio)
    if algorithm.lower() == "sfedavg":
        # Choose r as a fixed fraction for consistency
        r = max(1, int(0.2 * Dparam))  # 20% subspace by default
        dim_ratio = r / float(Dparam)
    else:
        r = Dparam  # full dimension
        dim_ratio = 1.0

    for t in range(T):
        # Subsample clients for this round
        selected = rng.choice(N, size=m, replace=False)

        # Sample subspace and projector for SFedAvg; identity for FedAvg
        if algorithm.lower() == "sfedavg":
            _, Pi_t = sample_random_subspace(Dparam, r, rng)
        else:
            Pi_t = np.eye(Dparam, dtype=float)

        # Collect local updates
        deltas = []
        for i in selected:
            delta_i, v_prev_i = client_update(
                theta_global=theta,
                Pi_t=Pi_t,
                client_data=train_clients[i],
                model_template=base_model,
                tau=tau,
                eta=eta,
                mu=mu,
                B=B,
                momentum_prev=client_momentum[i],
                rng=rng
            )
            deltas.append(delta_i)
            client_momentum[i] = v_prev_i

        # Server aggregation
        if len(deltas) > 0:
            mean_delta = np.mean(deltas, axis=0)
            theta = theta + mean_delta

        # Evaluate metrics on full training and test sets using current global parameters
        base_model.set_params_vector(theta)
        train_loss = base_model.loss(X_train_full, y_train_full)
        preds = base_model.predict(X_test)
        test_acc = float(np.mean(preds == y_test))

        train_losses.append(train_loss)
        test_accs.append(test_acc)
        # Cumulative normalized communication (rounds * dimension ratio)
        comm_efficiency.append((t + 1) * dim_ratio)

    return {
        "train_loss": train_losses,
        "test_acc": test_accs,
        "comm_eff": comm_efficiency
    }


# ----------------------------
# Main: argument parsing and experiment execution
# ----------------------------

def run_experiment(args) -> Dict[str, Dict[str, List[float]]]:
    """
    Execute a single experiment run (FedAvg vs SFedAvg) under the provided args.
    Returns the results dict with means/stds lists for each metric.
    This function performs no file I/O; the caller is responsible for saving.
    """
    # Generate data sized for scalability scenario
    total_samples = getattr(args, "n_samples", 0)
    if total_samples is None or total_samples <= 0:
        # Ensure adequate samples per client (≈200) while keeping a floor of 4000
        total_samples = max(4000, int(getattr(args, "num_clients", 20)) * 200)
    X_train, y_train, X_test, y_test = generate_synthetic_multiclass(
        n_samples=total_samples, n_features=64, n_classes=10, seed=42
    )

    # Initialize global model template
    base_model = SoftmaxModel(
        n_features=X_train.shape[1],
        n_classes=len(np.unique(y_train)),
        include_bias=True,
        seed=args.seed
    )

    # Partition training data across clients (IID)
    clients = partition_clients(X_train, y_train, num_clients=args.num_clients, seed=args.seed)

    # Inject label noise on a subset of clients per scenario (only if explicitly enabled)
    n_classes = len(np.unique(y_train))
    rng = np.random.default_rng(args.seed)
    n_noisy = max(0, int(np.round(args.noise_clients_fraction * args.num_clients)))
    if getattr(args, "enable_noise", False) and n_noisy > 0 and args.noise_rate > 0.0:
        noisy_indices = rng.choice(args.num_clients, size=n_noisy, replace=False)
        for idx_client in noisy_indices:
            Xi, yi = clients[idx_client]
            yi_corrupt = add_label_noise(yi, noise_rate=args.noise_rate, n_classes=n_classes, seed=int(args.seed + idx_client))
            clients[idx_client] = (Xi, yi_corrupt)

    # Run selected algorithm(s)
    results = {}
    alg_choice = getattr(args, "algorithm", "both").lower()
    if alg_choice in ("fedavg", "both"):
        metrics_fedavg = run_federated(
            algorithm="fedavg",
            base_model=base_model.copy(),  # same initial weights
            train_clients=clients,
            X_train_full=X_train,
            y_train_full=y_train,
            X_test=X_test,
            y_test=y_test,
            T=args.rounds,
            tau=args.local_steps,
            C=args.client_fraction,
            eta=args.lr,
            mu=args.momentum,
            B=args.batch_size,
            seed=args.seed
        )
        zeros_fedavg = [0.0] * len(metrics_fedavg["train_loss"])
        results.update({
            "train_loss_fedavg": {"means": metrics_fedavg["train_loss"], "stds": zeros_fedavg},
            "test_acc_fedavg": {"means": metrics_fedavg["test_acc"], "stds": zeros_fedavg},
            "comm_eff_fedavg": {"means": metrics_fedavg["comm_eff"], "stds": zeros_fedavg},
        })
    if alg_choice in ("sfedavg", "both"):
        metrics_sfedavg = run_federated(
            algorithm="sfedavg",
            base_model=base_model.copy(),  # same initial weights
            train_clients=clients,
            X_train_full=X_train,
            y_train_full=y_train,
            X_test=X_test,
            y_test=y_test,
            T=args.rounds,
            tau=args.local_steps,
            C=args.client_fraction,
            eta=args.lr,
            mu=args.momentum,
            B=args.batch_size,
            seed=args.seed + 1  # different seed for subspace sampling
        )
        zeros_sfedavg = [0.0] * len(metrics_sfedavg["train_loss"])
        results.update({
            "train_loss_sfedavg": {"means": metrics_sfedavg["train_loss"], "stds": zeros_sfedavg},
            "test_acc_sfedavg": {"means": metrics_sfedavg["test_acc"], "stds": zeros_sfedavg},
            "comm_eff_sfedavg": {"means": metrics_sfedavg["comm_eff"], "stds": zeros_sfedavg},
        })
    # Derived metrics: accuracy per unit communication
    if "test_acc_fedavg" in results and "comm_eff_fedavg" in results:
        acc = results["test_acc_fedavg"]["means"]
        comm = results["comm_eff_fedavg"]["means"]
        results["acc_per_comm_fedavg"] = {
            "means": [a / c if c > 0 else 0.0 for a, c in zip(acc, comm)],
            "stds": [0.0] * len(acc)
        }
    if "test_acc_sfedavg" in results and "comm_eff_sfedavg" in results:
        acc = results["test_acc_sfedavg"]["means"]
        comm = results["comm_eff_sfedavg"]["means"]
        results["acc_per_comm_sfedavg"] = {
            "means": [a / c if c > 0 else 0.0 for a, c in zip(acc, comm)],
            "stds": [0.0] * len(acc)
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help="Output directory for results")
    # Optional hyperparameters with sensible defaults
    parser.add_argument('--rounds', type=int, default=30, help="Federated rounds T")
    parser.add_argument('--local_steps', type=int, default=5, help="Local steps tau")
    parser.add_argument('--client_fraction', type=float, default=0.5, help="Fraction of clients per round C")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate eta")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum mu")
    parser.add_argument('--batch_size', type=int, default=32, help="Minibatch size B")
    parser.add_argument('--num_clients', type=int, default=20, help="Total number of clients N")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--algorithm', type=str, default='both', choices=['fedavg', 'sfedavg', 'both'], help="Which algorithm to run")
    parser.add_argument('--noise_rate', type=float, default=0.2, help="Label noise rate for noisy clients")
    parser.add_argument('--noise_clients_fraction', type=float, default=0.3, help="Fraction of clients with label noise")
    parser.add_argument('--n_samples', type=int, default=0, help="Total number of synthetic samples; if <=0, set to max(4000, num_clients*200)")
    parser.add_argument('--enable_noise', action='store_true', help='Enable label noise injection on subset of clients')
    parser.add_argument('--enable_tuning', action='store_true', help='Enable batch parameter tuning')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Snapshot this experiment file for reproducibility
    try:
        src_path = os.path.abspath(__file__)
        dst_path = os.path.join(args.out_dir, "experiment.py")
        if os.path.exists(src_path):
            with open(src_path, "r") as src_f, open(dst_path, "w") as dst_f:
                dst_f.write(src_f.read())
    except Exception:
        # Non-fatal: continue even if snapshot fails
        pass

    if args.enable_tuning:
        # Define parameter configurations to test
        tuning_configs = [
            {
                "lr": 0.05,
                "momentum": 0.0,
                "local_steps": 5,
                "rationale": "Establish a conservative baseline without momentum to isolate the effect of projection and assess stability under non-IID skew."
            },
            {
                "lr": 0.15,
                "momentum": 0.9,
                "local_steps": 5,
                "rationale": "Moderately aggressive learning rate with strong momentum to accelerate convergence while leveraging projected momentum to stabilize updates."
            },
            {
                "lr": 0.2,
                "momentum": 0.9,
                "local_steps": 5,
                "rationale": "High learning rate to stress stability; tests whether momentum projection can tolerate larger steps without oscillation."
            },
            {
                "lr": 0.1,
                "momentum": 0.9,
                "local_steps": 10,
                "rationale": "Increase local_steps to probe client drift; projected momentum is expected to mitigate drift and preserve aggregation stability."
            },
            {
                "lr": 0.1,
                "momentum": 0.5,
                "local_steps": 10,
                "rationale": "Same local depth but reduced momentum to examine sensitivity to momentum magnitude under drift-prone settings."
            },
            {
                "lr": 0.08,
                "momentum": 0.95,
                "local_steps": 3,
                "rationale": "Strong momentum with short local blocks aims for smooth, fast convergence while limiting drift."
            },
            {
                "lr": 0.12,
                "momentum": 0.7,
                "local_steps": 7,
                "rationale": "Balanced mid-range settings to explore trade-offs between speed and stability across skewed clients."
            },
            {
                "lr": 0.05,
                "momentum": 0.9,
                "local_steps": 15,
                "rationale": "Deep local training with low learning rate tests whether projection plus momentum can handle long local updates without divergence."
            }
        ]

        # Create tuning subdirectory
        tuning_dir = os.path.join(args.out_dir, "tuning")
        os.makedirs(tuning_dir, exist_ok=True)

        # Test each configuration
        all_results = {}
        for idx, config in enumerate(tuning_configs, 1):
            config_dir = os.path.join(tuning_dir, f"config_{idx}")
            os.makedirs(config_dir, exist_ok=True)

            # Override parameters with config values (keep other args as defaults)
            lr = config.get("lr", args.lr)
            momentum = config.get("momentum", args.momentum)
            local_steps = config.get("local_steps", args.local_steps)

            # Work on a shallow copy of args to avoid cross-config interference
            class A:
                pass
            cfg_args = A()
            # Copy all relevant fields
            for name in ["out_dir", "rounds", "client_fraction", "batch_size", "num_clients", "seed"]:
                setattr(cfg_args, name, getattr(args, name))
            setattr(cfg_args, "lr", lr)
            setattr(cfg_args, "momentum", momentum)
            setattr(cfg_args, "local_steps", local_steps)

            # Run experiment with this configuration
            results = run_experiment(cfg_args)

            # Save results
            with open(os.path.join(config_dir, "final_info.json"), "w") as f:
                json.dump(results, f, indent=2)

            all_results[f"config_{idx}"] = {
                "parameters": config,
                "results": results
            }

        # Save aggregated results
        with open(os.path.join(tuning_dir, "all_configs.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"Saved tuning results under: {tuning_dir}")

    else:
        # Normal single-run mode (baseline)
        baseline_dir = os.path.join(args.out_dir, "baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        results = run_experiment(args)
        with open(os.path.join(baseline_dir, "final_info.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved baseline results to {os.path.join(baseline_dir, 'final_info.json')}")

if __name__ == "__main__":
    main()
