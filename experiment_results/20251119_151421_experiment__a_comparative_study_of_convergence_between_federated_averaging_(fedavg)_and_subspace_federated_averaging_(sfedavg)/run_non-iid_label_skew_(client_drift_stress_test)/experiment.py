# Experiment: A Comparative Study of Convergence between FedAvg and SFedAvg
# Complete, runnable implementation using real data and gradients.

import argparse
import json
import os
import numpy as np
from typing import Tuple, Dict, Optional, List

# Use scikit-learn's digits dataset for fast, local classification
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def softmax(z: np.ndarray) -> np.ndarray:
    # Numerically stable softmax
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def loss_and_grad(X: np.ndarray, y: np.ndarray, W: np.ndarray, l2: float = 0.0) -> Tuple[float, np.ndarray]:
    """
    Cross-entropy loss and gradient for multinomial logistic regression (softmax).
    X: (n_samples, n_features)
    y: (n_samples,) integer labels
    W: (n_features, n_classes)
    Returns: (loss, dW)
    """
    n = X.shape[0]
    logits = X @ W
    probs = softmax(logits)
    Y = one_hot(y, W.shape[1])
    # Cross-entropy loss
    # Add small eps to avoid log(0)
    eps = 1e-12
    ce = -np.sum(Y * np.log(probs + eps)) / n
    # L2 regularization
    reg = 0.5 * l2 * np.sum(W * W)
    loss = ce + reg
    # Gradient
    dW = (X.T @ (probs - Y)) / n + l2 * W
    return loss, dW


def pack_weights(W: np.ndarray) -> np.ndarray:
    return W.reshape(-1)


def unpack_weights(w_vec: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return w_vec.reshape(shape)


def sample_projector(d: int, r: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample one-sided random subspace P in St(d, r) and return P and Pi = P P^T.
    P: (d, r) with orthonormal columns; Pi: (d, d) projector.
    """
    assert 1 <= r <= d
    # Random Gaussian matrix then QR for orthonormal columns
    A = rng.standard_normal((d, r))
    # QR decomposition
    Q, _ = np.linalg.qr(A)
    P = Q[:, :r]
    Pi = P @ P.T
    return P, Pi


def evaluate_accuracy(X: np.ndarray, y: np.ndarray, W: np.ndarray) -> float:
    logits = X @ W
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def sample_minibatch(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    if batch_size >= len(indices):
        return indices
    return rng.choice(indices, size=batch_size, replace=False)


def client_update(
    X: np.ndarray,
    y: np.ndarray,
    theta_t: np.ndarray,
    tau: int,
    eta: float,
    mu: float,
    B: int,
    rng: np.random.Generator,
    v_prev: Optional[np.ndarray],
    Pi_t: Optional[np.ndarray],
    use_momentum: bool,
    weight_shape: Tuple[int, int],
    l2: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ClientUpdate(i; θ^t, Π_t, τ, η, μ, B):
    - Initialize local state: θ_{i,0} ← θ^t
    - Momentum Projection (MP) at block start (SFedAvg only): v_i^0 ← Π_t v_i^{prev} if available, else 0
    - For s = 0..τ-1:
        • Sample minibatch ξ_{i,s} of size B
        • Compute gradient g_{i,s} = ∇F_i(θ_{i,s}; ξ_{i,s}) from actual data
        • If use_momentum (SFedAvg): one-sided projected momentum v_{i,s+1} = μ v_{i,s} + Π_t g_{i,s}; θ_{i,s+1} = θ_{i,s} − η v_{i,s+1}
        • Else (FedAvg): plain local SGD θ_{i,s+1} = θ_{i,s} − η g_{i,s}
    - Return Δ_i^t = θ_{i,τ} − θ^t and store v_i^{prev} = v_{i,τ} (zeros for FedAvg)
    """
    d_total = np.prod(weight_shape)

    theta_i = theta_t.copy()  # local weights copy from global
    theta_vec = pack_weights(theta_i)
    theta_vec0 = theta_vec.copy()

    # Initialize momentum only if enabled
    if use_momentum:
        if v_prev is not None:
            v = Pi_t @ v_prev if Pi_t is not None else v_prev.copy()
        else:
            v = np.zeros(d_total, dtype=np.float64)
    else:
        v = None  # momentum not used

    # Local steps
    for _ in range(tau):
        batch_idx = sample_minibatch(np.arange(X.shape[0]), B, rng)
        Xb = X[batch_idx]
        yb = y[batch_idx]
        # Compute true gradient from data
        _, dW = loss_and_grad(Xb, yb, unpack_weights(theta_vec, weight_shape), l2=l2)
        g_vec = pack_weights(dW)

        if use_momentum:
            # One-sided projected momentum update if Pi_t provided
            if Pi_t is not None:
                g_vec = Pi_t @ g_vec
            v = mu * v + g_vec
            theta_vec = theta_vec - eta * v
        else:
            # Plain local SGD without projection or momentum
            theta_vec = theta_vec - eta * g_vec

    delta_vec = theta_vec - theta_vec0
    delta = unpack_weights(delta_vec, weight_shape)
    if use_momentum:
        v_prev_new = v.copy()
    else:
        v_prev_new = np.zeros(d_total, dtype=np.float64)
    return delta, v_prev_new


def federated_train(
    algo_name: str,
    T: int,
    tau: int,
    C: float,
    eta: float,
    mu: float,
    B: int,
    r: Optional[int],
    X_train_clients: List[np.ndarray],
    y_train_clients: List[np.ndarray],
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    W_init: np.ndarray,
    rng: np.random.Generator,
    l2: float = 0.0,
) -> Dict[str, List[float]]:
    """
    Federated training loop adhering to the pseudocode:
    - Inputs: T rounds, local steps τ, client fraction C, stepsize η, momentum μ, batch size B
    - Server initialization: θ^0 = W_init
    - For each round t:
        • If SFedAvg: sample one-sided subspace P_t ∈ St(d, r) and set Π_t = P_t P_t^⊤ (fixed within round)
        • Sample client subset S_t of size m ≈ C N without replacement
        • For i ∈ S_t: send (θ^t, Π_t) and receive Δ_i^t = ClientUpdate(i; θ^t, Π_t, τ, η, μ, B)
        • Aggregation: θ^{t+1} = θ^t + (1/m) ∑_{i∈S_t} Δ_i^t
    - Metrics: compute train loss and test accuracy each round from actual data and predictions
    """
    N = len(X_train_clients)
    d_total = W_init.size
    weight_shape = W_init.shape
    W = W_init.copy()
    # Per-client momentum state
    v_prev_states: Dict[int, np.ndarray] = {}

    train_losses: List[float] = []
    test_accuracies: List[float] = []
    comm_cost_MB: List[float] = []
    comm_total_MB = 0.0

    for t in range(T):
        # Subspace sampling (held fixed within round) for SFedAvg
        Pi_t = None
        r_t = None
        if algo_name.lower() == "sfedavg":
            assert r is not None and 1 <= r <= d_total
            _, Pi_t = sample_projector(d_total, r, rng)
            r_t = r
        else:
            Pi_t = None

        # Sample client subset S_t
        m = max(1, int(np.round(C * N)))
        S_t = rng.choice(np.arange(N), size=m, replace=False)

        deltas = []
        # Client updates
        for i in S_t:
            theta_t = W.copy()
            v_prev = v_prev_states.get(i, None)

            delta_i, v_new = client_update(
                X_train_clients[i],
                y_train_clients[i],
                theta_t,
                tau,
                eta,
                mu,
                B,
                rng,
                v_prev,
                Pi_t,
                algo_name.lower() == "sfedavg",
                weight_shape,
                l2=l2,
            )
            v_prev_states[i] = v_new
            deltas.append(delta_i)

        # Aggregation
        mean_delta = np.mean(np.stack(deltas, axis=0), axis=0)
        W = W + mean_delta

        # Metrics
        train_loss, _ = loss_and_grad(X_train_all, y_train_all, W, l2=l2)
        acc = evaluate_accuracy(X_test, y_test, W)
        train_losses.append(train_loss)
        test_accuracies.append(acc)

        # Communication efficiency: cumulative MB up to current round
        if algo_name.lower() == "sfedavg":
            # Server->clients: theta (d_total) + projector Pi_t (d_total * d_total)
            server_to_clients = d_total + d_total * d_total
            # Clients->server: full-dimension delta per client (Δ_i^t ∈ R^d)
            clients_to_server = m * d_total
            floats = server_to_clients + clients_to_server
        else:
            # FedAvg: Server->clients: theta (d_total)
            server_to_clients = d_total
            # Clients->server: full-dimension delta per client
            clients_to_server = m * d_total
            floats = server_to_clients + clients_to_server
        per_round_MB = floats * 4.0 / 1e6
        comm_total_MB += per_round_MB
        comm_cost_MB.append(comm_total_MB)

    return {
        "train_loss": train_losses,
        "test_accuracy": test_accuracies,
        "comm_MB": comm_cost_MB,
        "final_weights": W,  # Not saved to JSON; here for potential future use
    }


def split_clients(X: np.ndarray, y: np.ndarray, N: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    IID split of training data into N clients.
    """
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    splits = np.array_split(idx, N)
    X_clients = [X[s] for s in splits]
    y_clients = [y[s] for s in splits]
    return X_clients, y_clients


def run_experiment(args) -> Dict[str, Dict[str, List[float]]]:
    """
    Execute a single experiment run using current args values.
    Returns results dict suitable for saving to final_info.json.
    """
    # RNG
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    # Load and preprocess data
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(int)
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # Clients split (IID here; scenario-specific splits can be added externally)
    N = args.clients
    X_train_clients, y_train_clients = split_clients(X_train, y_train, N)

    # Model initialization
    W0 = np.zeros((n_features, n_classes), dtype=np.float64)

    # Hyperparameters
    T = args.rounds
    tau = args.local_steps
    C = args.client_frac
    eta = args.lr
    mu = args.momentum
    B = args.batch_size
    l2 = args.l2
    d_total = W0.size
    r = max(1, int(args.subspace_ratio * d_total))

    # Run FedAvg
    fedavg_metrics = federated_train(
        "FedAvg",
        T, tau, C, eta, mu, B, None,
        X_train_clients, y_train_clients,
        X_train, y_train,
        X_test, y_test,
        W0, rng, l2=l2
    )

    # Run SFedAvg
    sfedavg_metrics = federated_train(
        "SFedAvg",
        T, tau, C, eta, mu, B, r,
        X_train_clients, y_train_clients,
        X_train, y_train,
        X_test, y_test,
        W0, rng, l2=l2
    )

    # Compose results
    results = {
        "train_loss_fedavg": {
            "means": fedavg_metrics["train_loss"],
            "stds": [0.0] * len(fedavg_metrics["train_loss"])
        },
        "train_loss_sfedavg": {
            "means": sfedavg_metrics["train_loss"],
            "stds": [0.0] * len(sfedavg_metrics["train_loss"])
        },
        "test_acc_fedavg": {
            "means": fedavg_metrics["test_accuracy"],
            "stds": [0.0] * len(fedavg_metrics["test_accuracy"])
        },
        "test_acc_sfedavg": {
            "means": sfedavg_metrics["test_accuracy"],
            "stds": [0.0] * len(sfedavg_metrics["test_accuracy"])
        },
        "comm_MB_fedavg": {
            "means": fedavg_metrics["comm_MB"],
            "stds": [0.0] * len(fedavg_metrics["comm_MB"])
        },
        "comm_MB_sfedavg": {
            "means": sfedavg_metrics["comm_MB"],
            "stds": [0.0] * len(sfedavg_metrics["comm_MB"])
        }
    }
    return results


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--out_dir', type=str, required=True, help="Directory to save results")
    # Optional hyperparameters with reasonable defaults
    parser.add_argument('--rounds', type=int, default=20, help="Federated rounds T")
    parser.add_argument('--local_steps', type=int, default=5, help="Local steps tau")
    parser.add_argument('--client_frac', type=float, default=0.5, help="Client fraction C")
    parser.add_argument('--lr', type=float, default=0.2, help="Stepsize eta")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum mu")
    parser.add_argument('--batch_size', type=int, default=32, help="Minibatch size B")
    parser.add_argument('--clients', type=int, default=20, help="Number of clients N")
    parser.add_argument('--subspace_ratio', type=float, default=0.25, help="r/d ratio for SFedAvg")
    parser.add_argument('--l2', '--l', type=float, default=0.0, help="L2 regularization strength (alias: --l)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--enable_tuning', action='store_true', help='Enable batch parameter tuning')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    if args.enable_tuning:
        # Define parameter configurations to test
        tuning_configs = [
            {
                "subspace_ratio": 0.1,
                "lr": 0.15,
                "local_steps": 4,
                "rationale": "Smaller subspace with slightly reduced lr and fewer local steps to curb drift while leveraging projection benefits."
            },
            {
                "subspace_ratio": 0.15,
                "lr": 0.12,
                "local_steps": 5,
                "rationale": "Moderate subspace size with conservative lr at default tau to balance learning speed and stability."
            },
            {
                "subspace_ratio": 0.25,
                "lr": 0.1,
                "local_steps": 3,
                "rationale": "Default r/d with lower lr and shorter local blocks to reduce client drift yet keep adequate update magnitude."
            },
            {
                "subspace_ratio": 0.05,
                "lr": 0.2,
                "local_steps": 3,
                "rationale": "Very small subspace to strongly filter drift, compensated by default lr and fewer steps to avoid underfitting."
            },
            {
                "subspace_ratio": 0.35,
                "lr": 0.08,
                "local_steps": 6,
                "rationale": "Larger subspace approaching full space; reduce lr and allow slightly longer local blocks to test stability vs speed."
            },
            {
                "subspace_ratio": 0.2,
                "lr": 0.18,
                "local_steps": 4,
                "rationale": "Intermediate r/d with modest lr and reduced tau to probe faster convergence without excessive drift."
            },
            {
                "subspace_ratio": 0.1,
                "lr": 0.25,
                "local_steps": 2,
                "rationale": "Aggressive lr with very short local blocks and small subspace to test if quick updates can still be stable under projection."
            },
            {
                "subspace_ratio": 0.3,
                "lr": 0.12,
                "local_steps": 8,
                "rationale": "Larger r/d with conservative lr and longer local blocks to explore projection’s ability to stabilize extended local training."
            }
        ]

        # Create tuning subdirectory
        tuning_dir = os.path.join(args.out_dir, "tuning")
        os.makedirs(tuning_dir, exist_ok=True)

        all_results = {}
        for idx, config in enumerate(tuning_configs, 1):
            config_dir = os.path.join(tuning_dir, f"config_{idx}")
            os.makedirs(config_dir, exist_ok=True)

            # Override parameters with config values
            for param_name, param_value in config.items():
                if param_name in {"subspace_ratio", "lr", "local_steps"}:
                    setattr(args, param_name, param_value)

            # Run experiment for this configuration
            results = run_experiment(args)

            # Save per-config results
            with open(os.path.join(config_dir, "final_info.json"), "w") as f:
                json.dump(results, f, indent=2)

            all_results[f"config_{idx}"] = {
                "parameters": {k: v for k, v in config.items() if k in {"subspace_ratio", "lr", "local_steps"}},
                "rationale": config.get("rationale", ""),
                "results": results
            }

        # Save aggregated results
        with open(os.path.join(tuning_dir, "all_configs.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        # Snapshot script
        try:
            src_path = os.path.abspath(__file__)
            with open(src_path, "r") as sf, open(os.path.join(args.out_dir, "experiment.py"), "w") as df:
                df.write(sf.read())
        except Exception:
            pass

    else:
        # Baseline single-run mode
        baseline_dir = os.path.join(args.out_dir, "baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        results = run_experiment(args)
        with open(os.path.join(baseline_dir, "final_info.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Snapshot script
        try:
            src_path = os.path.abspath(__file__)
            with open(src_path, "r") as sf, open(os.path.join(args.out_dir, "experiment.py"), "w") as df:
                df.write(sf.read())
        except Exception:
            pass


if __name__ == "__main__":
    main()
