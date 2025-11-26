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
from sklearn.decomposition import PCA


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


def augment_features(
    X: np.ndarray,
    method: str = "random_linear",
    target_dim: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Increase feature dimensionality of X according to the chosen method.

    Methods:
    - "random_linear": Concatenate original features with random linear projections to reach target_dim.
    - "pca_pad": Concatenate original features with PCA components; if still short of target_dim, pad zeros.

    X: (n_samples, n_features)
    target_dim: desired final dimensionality (> n_features)
    """
    n_samples, n_features = X.shape
    if target_dim is None or target_dim <= n_features:
        return X  # nothing to do

    rng = np.random.default_rng(seed)
    extra_dims = target_dim - n_features

    if method == "random_linear":
        # Create random projection matrix to generate extra features
        R = rng.standard_normal((n_features, extra_dims))
        X_extra = X @ R  # (n_samples, extra_dims)
        X_aug = np.concatenate([X, X_extra], axis=1)
        return X_aug

    elif method == "pca_pad":
        # Use PCA to generate components, then pad zeros if needed
        n_components = min(extra_dims, n_features)
        pca = PCA(n_components=n_components, random_state=seed)
        X_pca = pca.fit_transform(X)  # (n_samples, n_components)
        if n_components < extra_dims:
            # Pad with zeros to reach target_dim
            pad_width = extra_dims - n_components
            X_pad = np.zeros((n_samples, pad_width), dtype=X.dtype)
            X_extra = np.concatenate([X_pca, X_pad], axis=1)
        else:
            X_extra = X_pca
        X_aug = np.concatenate([X, X_extra], axis=1)
        return X_aug

    else:
        # Unknown method; return original
        return X


def add_label_noise(y: np.ndarray, noise_rate: float, n_classes: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Randomly flip a fraction (noise_rate) of labels to a different class.
    """
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n = y_noisy.shape[0]
    m = int(np.floor(noise_rate * n))
    if m <= 0:
        return y_noisy
    idx = rng.choice(n, size=m, replace=False)
    old = y_noisy[idx]
    # Ensure new labels differ from old: add random offset in [1, n_classes-1] modulo n_classes
    offsets = rng.integers(1, n_classes, size=m)
    new_labels = (old + offsets) % n_classes
    y_noisy[idx] = new_labels
    return y_noisy


def partition_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    classes_per_client: int = 2,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Non-IID partition where each client predominantly holds samples from 'classes_per_client' classes.
    Implementation:
    - Each client is assigned a preferred class set (size classes_per_client).
    - For each class, indices are distributed to clients using a Dirichlet with larger concentration
      for clients that prefer the class, yielding labeled skew while avoiding overlaps.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    n_classes = len(classes)

    # Preferred classes per client
    preferred: List[set] = [
        set(rng.choice(classes, size=min(classes_per_client, n_classes), replace=False))
        for _ in range(n_clients)
    ]

    # Collect indices per class
    class_to_indices: Dict[int, np.ndarray] = {}
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        class_to_indices[int(c)] = idx_c

    # Initialize client index buckets
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    # Distribute each class's indices across clients
    for c in classes:
        idx_c = class_to_indices[int(c)]
        if idx_c.size == 0:
            continue
        # Larger alpha for clients that prefer class c, smaller otherwise
        alpha_vec = np.array(
            [1.0 if int(c) in preferred[i] else 0.1 for i in range(n_clients)],
            dtype=np.float64
        )
        # Dirichlet proportions and integer counts
        proportions = rng.dirichlet(alpha_vec)
        counts = np.floor(proportions * idx_c.size).astype(int)
        # Fix rounding to cover all samples
        remainder = idx_c.size - counts.sum()
        if remainder > 0:
            # Give remainders to clients with largest proportions
            order = np.argsort(proportions)[::-1]
            for j in range(remainder):
                counts[order[j % n_clients]] += 1

        # Assign sequential slices
        start = 0
        for i in range(n_clients):
            k = counts[i]
            if k > 0:
                client_indices[i].extend(idx_c[start:start + k].tolist())
                start += k

    # Shuffle client indices and form per-client datasets
    X_clients = []
    y_clients = []
    for i in range(n_clients):
        ci = np.array(client_indices[i], dtype=int)
        rng.shuffle(ci)
        X_clients.append(X[ci])
        y_clients.append(y[ci])
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

    # Optional feature augmentation to increase dimensionality
    method = getattr(args, "feature_expand_method", "none")
    feature_multiplier = float(getattr(args, "feature_multiplier", 1.0))
    target_dim = getattr(args, "target_dim", None)
    aug_seed = getattr(args, "augment_seed", args.seed)

    if method != "none":
        # Determine target dimensionality
        if target_dim is None or (isinstance(target_dim, int) and target_dim <= 0):
            if feature_multiplier > 1.0:
                target_dim = int(np.ceil(X.shape[1] * feature_multiplier))
            else:
                target_dim = X.shape[1]
        # Apply augmentation only if target_dim is larger
        if target_dim is not None and target_dim > X.shape[1]:
            X = augment_features(X, method=method, target_dim=target_dim, seed=aug_seed)
    # Update feature dimension after potential augmentation
    n_features = X.shape[1]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # Clients split (IID or Non-IID), then optional label-noise injection
    N = args.clients
    partition_mode = getattr(args, "partition", "iid").lower()
    if partition_mode == "non_iid":
        X_train_clients, y_train_clients = partition_non_iid(
            X_train, y_train, N,
            classes_per_client=getattr(args, "classes_per_client", 2),
            seed=args.seed
        )
    else:
        X_train_clients, y_train_clients = split_clients(X_train, y_train, N)

    # Inject label noise per client if requested
    noise_rate = float(getattr(args, "label_noise_rate", 0.0))
    if noise_rate > 0.0:
        for ci in range(N):
            y_train_clients[ci] = add_label_noise(
                y_train_clients[ci],
                noise_rate=noise_rate,
                n_classes=n_classes,
                seed=args.seed + ci
            )

    # Model initialization
    W0 = np.zeros((X_train.shape[1], n_classes), dtype=np.float64)

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

    # Algorithm selection
    algo_sel = getattr(args, "algorithm", "both").lower()

    results: Dict[str, Dict[str, List[float]]] = {}

    # Run FedAvg if selected
    fedavg_metrics = None
    if algo_sel in ("both", "fedavg"):
        fedavg_metrics = federated_train(
            "FedAvg",
            T, tau, C, eta, mu, B, None,
            X_train_clients, y_train_clients,
            X_train, y_train,
            X_test, y_test,
            W0, rng, l2=l2
        )
        results["train_loss_fedavg"] = {
            "means": fedavg_metrics["train_loss"],
            "stds": [0.0] * len(fedavg_metrics["train_loss"])
        }
        results["test_acc_fedavg"] = {
            "means": fedavg_metrics["test_accuracy"],
            "stds": [0.0] * len(fedavg_metrics["test_accuracy"])
        }
        results["comm_MB_fedavg"] = {
            "means": fedavg_metrics["comm_MB"],
            "stds": [0.0] * len(fedavg_metrics["comm_MB"])
        }

    # Run SFedAvg if selected
    sfedavg_metrics = None
    if algo_sel in ("both", "sfedavg"):
        sfedavg_metrics = federated_train(
            "SFedAvg",
            T, tau, C, eta, mu, B, r,
            X_train_clients, y_train_clients,
            X_train, y_train,
            X_test, y_test,
            W0, rng, l2=l2
        )
        results["train_loss_sfedavg"] = {
            "means": sfedavg_metrics["train_loss"],
            "stds": [0.0] * len(sfedavg_metrics["train_loss"])
        }
        results["test_acc_sfedavg"] = {
            "means": sfedavg_metrics["test_accuracy"],
            "stds": [0.0] * len(sfedavg_metrics["test_accuracy"])
        }
        results["comm_MB_sfedavg"] = {
            "means": sfedavg_metrics["comm_MB"],
            "stds": [0.0] * len(sfedavg_metrics["comm_MB"])
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
    parser.add_argument('--partition', type=str, default='iid', choices=['iid', 'non_iid'], help="Data partition strategy")
    parser.add_argument('--label_noise_rate', type=float, default=0.0, help="Label noise rate per client (0.0–1.0)")
    parser.add_argument('--classes_per_client', type=int, default=2, help="Dominant classes per client (used for non_iid)")
    parser.add_argument('--algorithm', type=str, default='both', choices=['both', 'fedavg', 'sfedavg'], help="Algorithm(s) to run")
    # Feature augmentation options for scalability scenario
    parser.add_argument('--feature_expand_method', type=str, default='none', choices=['none', 'random_linear', 'pca_pad'], help="Method to expand feature dimensionality")
    parser.add_argument('--feature_multiplier', type=float, default=1.0, help="Multiplier for feature dimension (e.g., 2.0 doubles features)")
    parser.add_argument('--target_dim', type=int, default=None, help="Explicit target feature dimension (overrides multiplier if set)")
    parser.add_argument('--augment_seed', type=int, default=None, help="Seed for feature augmentation randomness (defaults to --seed)")
    parser.add_argument('--enable_tuning', action='store_true', help='Enable batch parameter tuning')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    if args.enable_tuning:
        # Define parameter configurations to test (batch tuning for scalability: many clients and higher-dimensional features)
        tuning_configs = [
            {
                "subspace_ratio": 0.03,
                "feature_multiplier": 2.0,
                "client_frac": 0.1,
                "rationale": "Smaller r/d to strongly filter updates under doubled feature dimension while keeping participation modest."
            },
            {
                "subspace_ratio": 0.05,
                "feature_multiplier": 2.5,
                "client_frac": 0.1,
                "rationale": "Scenario baseline-like r/d with more aggressive feature expansion to test projection’s stability in higher d."
            },
            {
                "subspace_ratio": 0.08,
                "feature_multiplier": 2.0,
                "client_frac": 0.1,
                "rationale": "Moderate r/d to retain more signal at 2x features, probing faster convergence vs drift."
            },
            {
                "subspace_ratio": 0.05,
                "feature_multiplier": 3.0,
                "client_frac": 0.15,
                "rationale": "Larger feature expansion with slightly higher client participation to reduce gradient variance."
            },
            {
                "subspace_ratio": 0.1,
                "feature_multiplier": 2.0,
                "client_frac": 0.05,
                "rationale": "Higher r/d for more signal retention when very few clients participate per round."
            },
            {
                "subspace_ratio": 0.07,
                "feature_multiplier": 1.5,
                "client_frac": 0.1,
                "rationale": "Moderate projection with modest feature growth as a conservative stability–speed trade-off."
            },
            {
                "subspace_ratio": 0.04,
                "feature_multiplier": 3.0,
                "client_frac": 0.1,
                "rationale": "Aggressive projection (smaller r/d) for very high dimensionality to cap communication and drift."
            },
            {
                "subspace_ratio": 0.06,
                "feature_multiplier": 2.5,
                "client_frac": 0.2,
                "rationale": "Balanced r/d with higher participation to accelerate averaging and reduce stochasticity in large-N settings."
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
                if param_name in {"subspace_ratio", "feature_multiplier", "client_frac"}:
                    setattr(args, param_name, param_value)
            # Ensure feature expansion is active for scalability scenario
            if getattr(args, "feature_expand_method", "none") == "none":
                setattr(args, "feature_expand_method", "random_linear")

            # Run experiment for this configuration
            results = run_experiment(args)

            # Save per-config results
            with open(os.path.join(config_dir, "final_info.json"), "w") as f:
                json.dump(results, f, indent=2)

            all_results[f"config_{idx}"] = {
                "parameters": {k: v for k, v in config.items() if k in {"subspace_ratio", "feature_multiplier", "client_frac"}},
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
