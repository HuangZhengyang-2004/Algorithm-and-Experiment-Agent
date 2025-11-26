# Experiment file - Federated Averaging (FedAvg) vs Subspace Federated Averaging (SFedAvg)

import argparse
import json
import os
import shutil
import numpy as np
from typing import Tuple, Dict, List, Optional

# Deterministic setup and reproducibility


def softmax(logits: np.ndarray) -> np.ndarray:
    # Stable softmax
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, y: np.ndarray) -> float:
    probs = softmax(logits)
    n = y.shape[0]
    # Gather probabilities of the correct classes
    p_correct = probs[np.arange(n), y]
    # Numerical stability
    eps = 1e-12
    return float(-np.mean(np.log(p_correct + eps)))


class LinearSoftmaxModel:
    """
    Simple linear softmax classifier:
    logits = X @ W + b
    where X: [n, d], W: [d, K], b: [K]
    """
    def __init__(self, d: int, k: int):
        # Deterministic small initialization to avoid randomness while breaking symmetry
        j = np.arange(d).reshape(-1, 1).astype(np.float64)
        c = np.arange(k).reshape(1, -1).astype(np.float64)
        self.W = 0.01 * (np.sin(j + 1) * np.cos(c + 1))
        self.b = np.zeros(k, dtype=np.float64)

    def copy_params_from(self, other: "LinearSoftmaxModel"):
        self.W = other.W.copy()
        self.b = other.b.copy()

    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X))

    def gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of cross-entropy loss w.r.t W and b.
        """
        n = X.shape[0]
        K = self.b.shape[0]
        probs = self.predict_proba(X)
        y_one_hot = np.eye(K)[y]
        diff = (probs - y_one_hot)  # [n, K]
        grad_W = (X.T @ diff) / n   # [d, K]
        grad_b = np.mean(diff, axis=0)  # [K]
        return grad_W, grad_b

    def param_count(self) -> int:
        return self.W.size + self.b.size


def load_dataset(
    d_features: Optional[int] = None,
    n_classes: Optional[int] = None,
    samples_per_class: Optional[int] = None,
    force_synthetic: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a dataset.
    - If force_synthetic or any of d_features/n_classes/samples_per_class is provided, generate a synthetic dataset.
    - Else try sklearn digits; fallback to small synthetic dataset if unavailable.
    Returns: X_train, y_train, X_test, y_test
    """
    # If synthetic is requested or parameters provided, build synthetic dataset
    if force_synthetic or (d_features is not None or n_classes is not None or samples_per_class is not None):
        K = n_classes if n_classes is not None else 10
        d = d_features if d_features is not None else 200
        n_per_class = samples_per_class if samples_per_class is not None else 2000
        X_list = []
        y_list = []
        for k in range(K):
            i = np.arange(n_per_class).astype(np.float64)
            base = np.zeros((n_per_class, d), dtype=np.float64)
            for j in range(d):
                # Structured but deterministic signal per feature/class
                base[:, j] = np.sin((j + 1) * (i + 1) / 75.0) + 0.15 * np.cos((k + 1) * (j + 3))
            # Class-specific offset to ensure separability
            base += (k - (K - 1) / 2.0) * 0.3
            X_list.append(base)
            y_list.append(np.full(n_per_class, k, dtype=np.int64))
        X = np.vstack(X_list).astype(np.float64)
        y = np.concatenate(y_list).astype(np.int64)
        # Standardize features
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - X_mean) / X_std
        # Train/test split (deterministic)
        n = X.shape[0]
        n_train = int(0.8 * n)
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    # Otherwise attempt to load sklearn digits dataset
    try:
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits.data.astype(np.float64)
        y = digits.target.astype(np.int64)
        # Standardize features
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - X_mean) / X_std
        # Train/test split (deterministic)
        n = X.shape[0]
        n_train = int(0.8 * n)
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    except Exception:
        # Fallback: deterministic synthetic dataset with K classes
        K = 5
        d = 20
        n_per_class = 400
        X_list = []
        y_list = []
        for k in range(K):
            i = np.arange(n_per_class).astype(np.float64)
            base = np.zeros((n_per_class, d), dtype=np.float64)
            for j in range(d):
                base[:, j] = np.sin((j + 1) * (i + 1) / 50.0) + 0.2 * np.cos((k + 1) * (j + 1))
            # Class-specific offset to ensure separability
            base += (k - (K - 1) / 2.0) * 0.5
            X_list.append(base)
            y_list.append(np.full(n_per_class, k, dtype=np.int64))
        X = np.vstack(X_list).astype(np.float64)
        y = np.concatenate(y_list).astype(np.int64)
        # Standardize features
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - X_mean) / X_std
        # Train/test split (deterministic)
        n = X.shape[0]
        n_train = int(0.8 * n)
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def partition_data(X: np.ndarray, y: np.ndarray, num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Evenly partition dataset among clients (deterministic split).
    """
    n = X.shape[0]
    idx = np.arange(n)
    shards = np.array_split(idx, num_clients)
    parts = [(X[shard], y[shard]) for shard in shards]
    return parts


def partition_iid(X: np.ndarray, y: np.ndarray, num_clients: int, seed: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    IID random partition: shuffle indices and split evenly across clients.
    Deterministic given a seed.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    shards = np.array_split(idx, num_clients)
    parts = [(X[shard], y[shard]) for shard in shards]
    return parts


def partition_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    classes_per_client: int = 3,
    major_fraction: float = 0.6,
    seed: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Non-IID partition:
    - Assign each client a subset of `classes_per_client` classes.
    - Each client predominantly holds samples from those classes (controlled by major_fraction, e.g., ~60%), remainder sampled from other classes (excluding assigned ones where possible).
    Deterministic given a seed.
    """
    rng = np.random.default_rng(seed)
    K = int(np.max(y)) + 1
    n_total = X.shape[0]

    # Indices per class
    class_indices = [np.where(y == k)[0].tolist() for k in range(K)]
    for k in range(K):
        rng.shuffle(class_indices[k])

    # Assign classes to each client
    class_assignments = [
        rng.choice(K, size=classes_per_client, replace=False).tolist()
        for _ in range(num_clients)
    ]

    # Target samples per client (even split)
    base = n_total // num_clients
    extra = n_total % num_clients
    targets = [base + (1 if i < extra else 0) for i in range(num_clients)]

    shards: List[np.ndarray] = []
    p_major = float(np.clip(major_fraction, 0.0, 1.0))  # predominant proportion from assigned classes

    # Helper to take up to 'take' items from a class pool
    def take_from_class(cls: int, take: int) -> List[int]:
        available = len(class_indices[cls])
        if available == 0 or take <= 0:
            return []
        t = min(available, take)
        grabbed = class_indices[cls][:t]
        class_indices[cls] = class_indices[cls][t:]
        return grabbed

    # Allocate per client
    for i in range(num_clients):
        target = targets[i]
        n_major = int(np.round(p_major * target))
        per_cls_need = max(1, n_major // classes_per_client)
        selected: List[int] = []

        # Pull major portion from assigned classes
        for cls in class_assignments[i]:
            selected.extend(take_from_class(cls, per_cls_need))
        # If still short, keep pulling from assigned classes while possible
        while len(selected) < n_major:
            any_available = False
            for cls in class_assignments[i]:
                grabbed = take_from_class(cls, 1)
                if grabbed:
                    selected.extend(grabbed)
                    any_available = True
                    if len(selected) >= n_major:
                        break
            if not any_available:
                break  # no more assigned-class samples

        # Fill remainder from any other classes
        n_rem = target - len(selected)
        if n_rem > 0:
            # Build a pooled list from non-assigned classes; fallback to all classes if insufficient
            other_classes = [cls for cls in range(K) if cls not in class_assignments[i]]
            pool = [idx for cls in other_classes for idx in class_indices[cls]]
            if len(pool) < n_rem:
                pool = [idx for cls in range(K) for idx in class_indices[cls]]
            rng.shuffle(pool)
            add = pool[:n_rem]
            selected.extend(add)
            # Remove used from class pools
            used_set = set(add)
            for cls in range(K):
                if class_indices[cls]:
                    class_indices[cls] = [idx for idx in class_indices[cls] if idx not in used_set]

        shards.append(np.array(selected, dtype=np.int64))

    parts = [(X[shard], y[shard]) for shard in shards]
    return parts


def add_label_noise(y: np.ndarray, noise_rate: float = 0.2, n_classes: int = 10, seed: Optional[int] = None) -> np.ndarray:
    """
    Randomly flip a noise_rate proportion of labels to a different class (uniformly),
    independently per sample. Deterministic given a seed.
    """
    if noise_rate <= 0.0:
        return y.copy()
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n = y_noisy.size
    mask = rng.random(n) < noise_rate
    if np.any(mask):
        orig = y_noisy[mask].astype(np.int64)
        # Sample uniformly from n_classes-1 options, and shift to avoid original class
        new_raw = rng.integers(0, n_classes - 1, size=mask.sum(), dtype=np.int64)
        new_labels = new_raw + (new_raw >= orig).astype(np.int64)
        y_noisy[mask] = new_labels
    return y_noisy


def sample_stiefel(d: int, r: int, round_index: int) -> np.ndarray:
    """
    Sample a random orthonormal subspace P in St(d, r) per round using a reproducible RNG,
    as required by the pseudocode (one-sided random subspace).
    """
    if r > d:
        r = d
    rng = np.random.default_rng(12345 + int(round_index))
    A = rng.standard_normal((d, r))
    Q, _ = np.linalg.qr(A)
    return Q[:, :r]


def client_update(
    model_global: LinearSoftmaxModel,
    client_data: Tuple[np.ndarray, np.ndarray],
    tau: int,
    eta: float,
    mu: float,
    batch_size: int,
    projector: np.ndarray = None,
    use_momentum: bool = True,
    momentum_state: Optional[Dict[str, np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    ClientUpdate with optional momentum and one-sided projected momentum (MP).
    Implements:
      - v^0 = Pi_t v_prev if stored, else 0 (for W component)
      - v_{s+1} = mu v_s + Pi_t g_s (for W); bias uses standard momentum if enabled
      - theta_{s+1} = theta_s - eta v_{s+1}
    """
    X, y = client_data

    # Initialize local model from global (copy parameters)
    local_W = model_global.W.copy()
    local_b = model_global.b.copy()

    # Initialize momentum states
    if use_momentum:
        if momentum_state is not None:
            vW_prev = momentum_state.get("vW", np.zeros_like(local_W))
            vB_prev = momentum_state.get("vB", np.zeros_like(local_b))
            vW = projector @ vW_prev if projector is not None else vW_prev.copy()
            vB = vB_prev.copy()
        else:
            vW = np.zeros_like(local_W)
            vB = np.zeros_like(local_b)
    else:
        vW = None
        vB = None

    rng_local = rng if rng is not None else np.random.default_rng(0)
    n = X.shape[0]
    for s in range(tau):
        # Sample a minibatch of size B (random, reproducible via rng_local)
        bs = min(batch_size, n)
        idx = rng_local.choice(n, size=bs, replace=False)
        Xb = X[idx]
        yb = y[idx]

        # Compute gradients from real predictions/data
        logits = Xb @ local_W + local_b
        probs = softmax(logits)
        K_ = local_b.shape[0]
        y_one_hot = np.eye(K_)[yb]
        diff = probs - y_one_hot
        grad_W = (Xb.T @ diff) / Xb.shape[0]
        grad_b = np.mean(diff, axis=0)

        if use_momentum:
            grad_W_proj = projector @ grad_W if projector is not None else grad_W
            vW = mu * vW + grad_W_proj
            vB = mu * vB + grad_b
            local_W = local_W - eta * vW
            local_b = local_b - eta * vB
        else:
            # Plain SGD without momentum (FedAvg local update)
            local_W = local_W - eta * grad_W
            local_b = local_b - eta * grad_b

    # Compute deltas
    delta_W = local_W - model_global.W
    delta_b = local_b - model_global.b

    updated_momentum = {"vW": vW.copy(), "vB": vB.copy()} if use_momentum else None
    return delta_W, delta_b, updated_momentum


def evaluate(model: LinearSoftmaxModel, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    train_loss = cross_entropy_loss(model.logits(X_train), y_train)
    test_pred = model.predict(X_test)
    test_acc = float(np.mean(test_pred == y_test))
    return train_loss, test_acc


def run_experiment(
    T: int,
    tau: int,
    C: float,
    eta: float,
    mu: float,
    batch_size: int,
    num_clients: int,
    r_subspace: int,
    label_noise_rate: float = 0.0,
    algorithm: str = "both",
    partition_type: str = "iid",
    d_features: int = 200,
    n_classes: int = 10,
    samples_per_class: int = 2000,
    classes_per_client: int = 3,
    major_fraction: float = 0.6,
    force_synthetic: bool = True
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run FedAvg and SFedAvg side-by-side on the same data partitions.
    Returns results dict in required format.
    """
    # Load data
    X_train, y_train, X_test, y_test = load_dataset(
        d_features=d_features,
        n_classes=n_classes,
        samples_per_class=samples_per_class,
        force_synthetic=force_synthetic
    )
    d = X_train.shape[1]
    K = int(np.max(y_train)) + 1

    # Partition data among clients
    if partition_type == "iid":
        client_parts = partition_iid(X_train, y_train, num_clients, seed=777)
    else:
        client_parts = partition_non_iid(X_train, y_train, num_clients, classes_per_client=classes_per_client, major_fraction=major_fraction, seed=777)

    # Inject uniform label noise per client independently on training labels
    if label_noise_rate > 0.0:
        noisy_parts: List[Tuple[np.ndarray, np.ndarray]] = []
        for i, (Xi, yi) in enumerate(client_parts):
            yi_noisy = add_label_noise(yi, noise_rate=label_noise_rate, n_classes=K, seed=55555 + i)
            noisy_parts.append((Xi, yi_noisy))
        client_parts = noisy_parts

    # Initialize global models
    global_fed = LinearSoftmaxModel(d, K)
    global_sfed = LinearSoftmaxModel(d, K)

    # Per-client momentum states for each method
    mom_fed = [None for _ in range(num_clients)]
    mom_sfed = [None for _ in range(num_clients)]

    # Metrics over rounds
    train_loss_fed, test_acc_fed, comm_fed = [], [], []
    train_loss_sfed, test_acc_sfed, comm_sfed = [], [], []

    param_count = global_fed.param_count()
    # Track cumulative communication for changing metrics
    cumulative_comm_fed = 0.0
    cumulative_comm_sfed = 0.0

    for t in range(T):
        # Sample client subset S_t (random subset of clients, reproducible per round)
        m = max(1, int(np.ceil(C * num_clients)))
        rng_sel = np.random.default_rng(24680 + t)
        client_ids = rng_sel.choice(num_clients, size=m, replace=False)

        # Sample one-sided subspace for SFedAvg (deterministic per round) only if running SFedAvg
        Pi_t = None
        if algorithm in ("both", "sfedavg"):
            P_t = sample_stiefel(d, r_subspace, round_index=t)
            Pi_t = P_t @ P_t.T  # projector, held fixed within the round

        # Collect deltas
        deltas_fed_W, deltas_fed_b = [], []
        deltas_sfed_W, deltas_sfed_b = [], []

        # Client updates - FedAvg (no projection, no momentum; standard FedAvg local SGD)
        if algorithm in ("both", "fedavg"):
            for i in client_ids:
                delta_W, delta_b, _ = client_update(
                    model_global=global_fed,
                    client_data=client_parts[i],
                    tau=tau,
                    eta=eta,
                    mu=0.0,
                    batch_size=batch_size,
                    projector=None,
                    use_momentum=False,
                    momentum_state=None,
                    rng=np.random.default_rng(100000 * (t + 1) + int(i))
                )
                deltas_fed_W.append(delta_W)
                deltas_fed_b.append(delta_b)

        # Client updates - SFedAvg (one-sided projection + momentum projection)
        if algorithm in ("both", "sfedavg"):
            for i in client_ids:
                delta_W, delta_b, mom = client_update(
                    model_global=global_sfed,
                    client_data=client_parts[i],
                    tau=tau,
                    eta=eta,
                    mu=mu,
                    batch_size=batch_size,
                    projector=Pi_t,
                    use_momentum=True,
                    momentum_state=mom_sfed[i],
                    rng=np.random.default_rng(200000 * (t + 1) + int(i))
                )
                deltas_sfed_W.append(delta_W)
                deltas_sfed_b.append(delta_b)
                mom_sfed[i] = mom

        # Aggregate: average deltas
        if algorithm in ("both", "fedavg") and len(deltas_fed_W) > 0:
            avg_delta_fed_W = np.mean(deltas_fed_W, axis=0)
            avg_delta_fed_b = np.mean(deltas_fed_b, axis=0)
            global_fed.W += avg_delta_fed_W
            global_fed.b += avg_delta_fed_b

        if algorithm in ("both", "sfedavg") and len(deltas_sfed_W) > 0:
            avg_delta_sfed_W = np.mean(deltas_sfed_W, axis=0)
            avg_delta_sfed_b = np.mean(deltas_sfed_b, axis=0)
            global_sfed.W += avg_delta_sfed_W
            global_sfed.b += avg_delta_sfed_b

        # Evaluate after aggregation
        if algorithm in ("both", "fedavg"):
            tl_fed, acc_fed = evaluate(global_fed, X_train, y_train, X_test, y_test)
            train_loss_fed.append(tl_fed)
            test_acc_fed.append(acc_fed)

        if algorithm in ("both", "sfedavg"):
            tl_sfed, acc_sfed = evaluate(global_sfed, X_train, y_train, X_test, y_test)
            train_loss_sfed.append(tl_sfed)
            test_acc_sfed.append(acc_sfed)

        # Communication efficiency (floats exchanged)
        # Server sends global params to each selected client and receives deltas back.
        # SFedAvg additionally sends the projector P_t (d x r).
        if algorithm in ("both", "fedavg"):
            comm_round_fed = m * (param_count + param_count)
            cumulative_comm_fed += comm_round_fed
            comm_fed.append(float(cumulative_comm_fed))

        if algorithm in ("both", "sfedavg"):
            comm_round_sfed = m * (param_count + param_count + d * r_subspace)
            cumulative_comm_sfed += comm_round_sfed
            comm_sfed.append(float(cumulative_comm_sfed))

    # Prepare results with means and stds (single run -> stds zeros)
    zeros_fed = [0.0] * len(train_loss_fed)
    zeros_sfed = [0.0] * len(train_loss_sfed)
    results: Dict[str, Dict[str, List[float]]] = {}
    if algorithm in ("both", "fedavg"):
        results.update({
            "train_loss_FedAvg": {"means": train_loss_fed, "stds": zeros_fed},
            "test_accuracy_FedAvg": {"means": test_acc_fed, "stds": zeros_fed},
            "comm_floats_FedAvg": {"means": comm_fed, "stds": zeros_fed},
        })
    if algorithm in ("both", "sfedavg"):
        results.update({
            "train_loss_SFedAvg": {"means": train_loss_sfed, "stds": zeros_sfed},
            "test_accuracy_SFedAvg": {"means": test_acc_sfed, "stds": zeros_sfed},
            "comm_floats_SFedAvg": {"means": comm_sfed, "stds": zeros_sfed},
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--rounds', type=int, default=60, help='Number of federated rounds T')
    parser.add_argument('--local_steps', type=int, default=5, help='Local steps tau per client')
    parser.add_argument('--client_fraction', type=float, default=0.1, help='Client fraction C (0,1]')
    parser.add_argument('--stepsize', type=float, default=0.1, help='Learning rate eta')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum mu')
    parser.add_argument('--batch_size', type=int, default=32, help='Local minibatch size B')
    parser.add_argument('--num_clients', type=int, default=50, help='Total number of clients N')
    parser.add_argument('--subspace_rank', type=int, default=1, help='Subspace rank r')
    parser.add_argument('--label_noise_rate', type=float, default=0.0, help='Fraction of labels to flip per client')
    parser.add_argument('--algorithm', type=str, choices=['both', 'fedavg', 'sfedavg'], default='both', help='Which algorithm(s) to run')
    parser.add_argument('--partition', type=str, choices=['iid', 'non_iid'], default='non_iid', help='Data partition strategy')
    parser.add_argument('--classes_per_client', type=int, default=3, help='Predominant classes per client in non-IID partition')
    parser.add_argument('--major_fraction', type=float, default=0.6, help="Fraction of each client's data from its assigned classes in non-IID partition")
    parser.add_argument('--d_features', type=int, default=200, help='Feature dimensionality d for synthetic dataset')
    parser.add_argument('--n_classes', type=int, default=10, help='Number of classes K for synthetic dataset')
    parser.add_argument('--samples_per_class', type=int, default=2000, help='Samples per class for synthetic dataset')
    parser.add_argument('--force_synthetic', action='store_true', help='Force synthetic dataset even if sklearn is available')
    args = parser.parse_args()

    # Create output directory structure
    os.makedirs(args.out_dir, exist_ok=True)
    baseline_dir = os.path.join(args.out_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    # Run experiment
    results = run_experiment(
        T=args.rounds,
        tau=args.local_steps,
        C=args.client_fraction,
        eta=args.stepsize,
        mu=args.momentum,
        batch_size=args.batch_size,
        num_clients=args.num_clients,
        r_subspace=args.subspace_rank,
        label_noise_rate=args.label_noise_rate,
        algorithm=args.algorithm,
        partition_type=args.partition,
        d_features=args.d_features,
        n_classes=args.n_classes,
        samples_per_class=args.samples_per_class,
        classes_per_client=args.classes_per_client,
        major_fraction=args.major_fraction,
        force_synthetic=args.force_synthetic
    )

    # Save results in the required format
    with open(os.path.join(baseline_dir, "final_info.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Snapshot the experiment script for reproducibility
    try:
        shutil.copy(__file__, os.path.join(args.out_dir, "experiment.py"))
    except Exception:
        # If running in an environment where __file__ is not available, ignore snapshotting.
        pass


if __name__ == "__main__":
    main()
