# Experiment: FedAvg vs SFedAvg (One-Sided Random Subspace + Momentum Projection)
# Complete, deterministic federated learning experiment with real gradient computation.
# Uses multinomial logistic regression trained on a deterministic, non-random synthetic dataset
# generated from a fixed linear teacher to ensure true learning behavior without random data.

import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    # logits: (n, C)
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    # probs: (n, C), y: (n,) int labels
    n = probs.shape[0]
    # Clamp for numerical stability
    eps = 1e-12
    p = np.clip(probs[np.arange(n), y], eps, 1.0)
    return float(-np.mean(np.log(p)))


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def pack_params(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([W.ravel(), b])


def unpack_params(theta: np.ndarray, d_in: int, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    w_size = d_in * n_classes
    W = theta[:w_size].reshape(d_in, n_classes)
    b = theta[w_size:]
    return W, b


def predict_logits(X: np.ndarray, theta: np.ndarray, d_in: int, n_classes: int) -> np.ndarray:
    W, b = unpack_params(theta, d_in, n_classes)
    return X @ W + b  # (n, C)


def predict_proba(X: np.ndarray, theta: np.ndarray, d_in: int, n_classes: int) -> np.ndarray:
    return softmax(predict_logits(X, theta, d_in, n_classes))


def accuracy(X: np.ndarray, y: np.ndarray, theta: np.ndarray, d_in: int, n_classes: int) -> float:
    probs = predict_proba(X, theta, d_in, n_classes)
    y_pred = np.argmax(probs, axis=1)
    return float(np.mean(y_pred == y))


def grad_ce(X: np.ndarray, y: np.ndarray, theta: np.ndarray, d_in: int, n_classes: int, l2_reg: float) -> np.ndarray:
    # Compute gradient of mean cross-entropy wrt parameters (W, b)
    n = X.shape[0]
    W, b = unpack_params(theta, d_in, n_classes)
    logits = X @ W + b
    probs = softmax(logits)
    Y = one_hot(y, n_classes)
    dZ = (probs - Y) / n  # (n, C)
    dW = X.T @ dZ  # (d_in, C)
    db = np.sum(dZ, axis=0)  # (C,)
    # L2 regularization on W only
    if l2_reg > 0.0:
        dW = dW + l2_reg * W
    return pack_params(dW, db)


def make_features(X2: np.ndarray, d_out: int) -> np.ndarray:
    # Deterministic feature mapping from 2D to d_out dimensions (no randomness)
    x1 = X2[:, 0]
    x2 = X2[:, 1]
    base_feats = [
        x1,
        x2,
        x1 * x2,
        x1 ** 2,
        x2 ** 2,
        np.sin(x1),
        np.cos(x2),
        np.sin(x1 * x2),
        np.cos(x1 + x2),
        np.tanh(x1 - x2),
    ]
    feats = []
    i = 0
    while len(feats) < d_out:
        feats.append(base_feats[i % len(base_feats)])
        i += 1
    X = np.stack(feats, axis=1)  # (n, d_out)
    # Standardize features deterministically
    X = X - np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    X = X / std
    return X


def deterministic_teacher(d_in: int, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    # Fixed teacher parameters to generate labels (no randomness)
    W_true = np.zeros((d_in, n_classes), dtype=np.float64)
    for j in range(d_in):
        for k in range(n_classes):
            W_true[j, k] = np.sin((j + 1) * (k + 1)) / np.sqrt(d_in)
    b_true = np.array([np.cos(k + 1) for k in range(n_classes)], dtype=np.float64)
    return W_true, b_true


def generate_dataset(total_size: int, d_in: int, n_classes: int, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Create a deterministic 2D grid, map to d_in features, label via fixed linear teacher
    side = int(np.ceil(np.sqrt(total_size)))
    coords = np.linspace(-3.0, 3.0, side, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords)
    X2 = np.stack([xx.ravel(), yy.ravel()], axis=1)
    X2 = X2[:total_size, :]
    X = make_features(X2, d_in)  # (N, d_in)

    W_true, b_true = deterministic_teacher(d_in, n_classes)
    logits = X @ W_true + b_true
    y = np.argmax(logits, axis=1).astype(np.int64)

    # Deterministic split
    n_test = int(np.floor(total_size * test_ratio))
    n_train = total_size - n_test
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    return X_train, y_train, X_test, y_test


def partition_noniid_by_label(X: np.ndarray, y: np.ndarray, num_clients: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Sort by label and assign contiguous blocks to clients to induce heterogeneity
    idx = np.argsort(y, kind='mergesort')
    Xs, ys = X[idx], y[idx]
    n = Xs.shape[0]
    sizes = [(n // num_clients) + (1 if r < (n % num_clients) else 0) for r in range(num_clients)]
    X_parts, y_parts = [], []
    start = 0
    for s in sizes:
        end = start + s
        X_parts.append(Xs[start:end])
        y_parts.append(ys[start:end])
        start = end
    return X_parts, y_parts


def partition_iid(X: np.ndarray, y: np.ndarray, num_clients: int, seed: int = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Randomly shuffle and split data evenly across clients (IID)
    n = X.shape[0]
    rng = np.random.RandomState(seed if seed is not None else 0)
    perm = rng.permutation(n)
    Xs, ys = X[perm], y[perm]
    sizes = [(n // num_clients) + (1 if r < (n % num_clients) else 0) for r in range(num_clients)]
    parts_X, parts_y = [], []
    start = 0
    for s in sizes:
        end = start + s
        parts_X.append(Xs[start:end])
        parts_y.append(ys[start:end])
        start = end
    return parts_X, parts_y


def partition_non_iid(X: np.ndarray, y: np.ndarray, n_clients: int, classes_per_client: int = 2, seed: int = None, majority_fraction: float = 0.8) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extreme non-IID partition: each client predominantly holds samples from 'classes_per_client' classes.
    Deterministic assignment of class subsets; majority (~majority_fraction) drawn from assigned classes, remainder from global pool.
    """
    n = y.shape[0]
    n_classes = int(np.max(y)) + 1
    classes_per_client = max(1, min(classes_per_client, n_classes))
    # Determine per-client sizes
    sizes = [(n // n_clients) + (1 if r < (n % n_clients) else 0) for r in range(n_clients)]

    # Indices per class in deterministic order
    cls_indices = {c: np.where(y == c)[0].tolist() for c in range(n_classes)}
    # Global pool of remaining indices
    unused = list(range(n))

    parts_idx: List[List[int]] = []
    for c in range(n_clients):
        s = sizes[c]
        # Deterministic class assignment for client c
        sel_classes = [int((c + k) % n_classes) for k in range(classes_per_client)]
        target_major = int(np.floor(majority_fraction * s))
        chosen: List[int] = []

        # Take majority from selected classes in round-robin
        ci = 0
        while len(chosen) < target_major and any(len(cls_indices[sc]) > 0 for sc in sel_classes):
            sc = sel_classes[ci % len(sel_classes)]
            # Pop until we find an unused index for this class
            while len(cls_indices[sc]) > 0 and cls_indices[sc][0] not in unused:
                cls_indices[sc].pop(0)
            if len(cls_indices[sc]) > 0:
                idx_val = cls_indices[sc].pop(0)
                if idx_val in unused:
                    chosen.append(idx_val)
                    # mark as used
                    # will filter unused later for efficiency
            ci += 1
            # Break if all selected classes are empty
            if all(len(cls_indices[sc]) == 0 for sc in sel_classes):
                break

        # Fill the rest from the global pool deterministically
        # Clean unused to remove any 'chosen' picked so far
        if chosen:
            chosen_set = set(chosen)
            unused = [u for u in unused if u not in chosen_set]
        need = s - len(chosen)
        if need > 0:
            take = unused[:need]
            chosen.extend(take)
            unused = unused[need:]

        parts_idx.append(chosen)

    # Build actual splits
    X_parts = [X[idxs] for idxs in parts_idx]
    y_parts = [y[idxs] for idxs in parts_idx]
    return X_parts, y_parts


def add_label_noise(y: np.ndarray, noise_rate: float = 0.0, n_classes: int = 10, rng: np.random.RandomState = None) -> np.ndarray:
    """
    Randomly flip a proportion of labels (noise_rate) to a different class in {0,...,n_classes-1}.
    Flips are deterministic given rng.
    """
    if noise_rate <= 0.0:
        return y
    if rng is None:
        rng = np.random.RandomState(0)
    y_noisy = y.copy()
    n = y_noisy.shape[0]
    m = int(np.floor(noise_rate * n))
    if m <= 0 or n_classes <= 1:
        return y_noisy
    perm = rng.permutation(n)
    flip_idx = perm[:m]
    # Choose new classes different from original
    offsets = rng.randint(1, n_classes, size=m)
    y_noisy[flip_idx] = (y_noisy[flip_idx] + offsets) % n_classes
    return y_noisy


def orthonormal_basis(D: int, r: int, rng: np.random.RandomState) -> np.ndarray:
    # Return P \in R^{D x r} with orthonormal columns (QR from a Gaussian matrix)
    if r <= 0:
        return np.zeros((D, 0), dtype=np.float64)
    if r >= D:
        # Full basis equals identity (columns are standard basis)
        return np.eye(D, dtype=np.float64)
    A = rng.randn(D, r)
    Q, _ = np.linalg.qr(A)
    return Q[:, :r]


def apply_projection(vec: np.ndarray, P: np.ndarray) -> np.ndarray:
    # Compute Pi vec = P (P^T vec) without forming Pi explicitly
    if P.size == 0:
        return np.zeros_like(vec)
    return P @ (P.T @ vec)


def client_next_batch(Xc: np.ndarray, yc: np.ndarray, ptr: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, int]:
    n = Xc.shape[0]
    if n == 0:
        raise ValueError("Client has no data.")
    end = ptr + batch_size
    if end <= n:
        Xb = Xc[ptr:end]
        yb = yc[ptr:end]
        new_ptr = end % n
    else:
        # wrap-around deterministically
        wrap = end - n
        Xb = np.concatenate([Xc[ptr:], Xc[:wrap]], axis=0)
        yb = np.concatenate([yc[ptr:], yc[:wrap]], axis=0)
        new_ptr = wrap
    return Xb, yb, new_ptr


def evaluate_metrics(theta: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                     d_in: int, n_classes: int, l2_reg: float) -> Tuple[float, float, float]:
    # Compute train loss, test accuracy, and (optional) test loss if needed
    probs_train = predict_proba(X_train, theta, d_in, n_classes)
    train_loss = cross_entropy_loss(probs_train, y_train)
    # Add L2 penalty to train loss to reflect optimization objective
    W, _ = unpack_params(theta, d_in, n_classes)
    if l2_reg > 0.0:
        train_loss = train_loss + 0.5 * l2_reg * float(np.sum(W * W)) / X_train.shape[0]
    test_acc = accuracy(X_test, y_test, theta, d_in, n_classes)
    return train_loss, test_acc, float(train_loss)  # return train loss twice for compatibility if needed


def run_experiment(args) -> Dict[str, Dict[str, List[float]]]:
    """
    Run a single federated learning experiment given argparse.Namespace or dict of parameters.
    Returns a results dictionary suitable for saving to JSON.
    """
    # Accept dict or Namespace for args
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    # Generate data
    X_train, y_train, X_test, y_test = generate_dataset(args.dataset_size, args.input_dim, args.n_classes, test_ratio=0.2)
    # Partition among clients
    if getattr(args, "partition_strategy", "noniid_sorted") == "iid":
        X_clients, y_clients = partition_iid(X_train, y_train, args.num_clients, seed=args.seed)
    elif getattr(args, "partition_strategy", "noniid_sorted") == "noniid_classes":
        cpc = max(1, min(getattr(args, "classes_per_client", 2), args.n_classes))
        X_clients, y_clients = partition_non_iid(
            X_train, y_train, args.num_clients,
            classes_per_client=cpc, seed=args.seed,
            majority_fraction=float(getattr(args, "majority_fraction", 0.8))
        )
    else:
        # Default to previous behavior: contiguous label blocks
        X_clients, y_clients = partition_noniid_by_label(X_train, y_train, args.num_clients)

    # Optional label noise injection on a subset of clients (deterministic selection)
    noise_rate = float(getattr(args, "label_noise_rate", 0.0))
    noise_frac = float(getattr(args, "label_noise_clients_fraction", 0.0))
    if noise_rate > 0.0 and noise_frac > 0.0:
        n_clients = args.num_clients
        n_corrupt = max(1, int(np.floor(noise_frac * n_clients)))
        rng_clients = np.random.RandomState(args.seed + 12345)
        perm_clients = rng_clients.permutation(n_clients)
        corrupt_ids = set(perm_clients[:n_corrupt].tolist())
        for ci in range(n_clients):
            if ci in corrupt_ids:
                rng_i = np.random.RandomState(args.seed + ci)
                y_clients[ci] = add_label_noise(y_clients[ci], noise_rate=noise_rate, n_classes=args.n_classes, rng=rng_i)

    d_in = args.input_dim
    n_classes = args.n_classes
    D = d_in * n_classes + n_classes  # parameter vector size

    # Initialize global model
    theta = np.zeros(D, dtype=np.float64)

    # Per-client momentum state and batch pointers
    v_prev = [np.zeros(D, dtype=np.float64) for _ in range(args.num_clients)]
    ptrs = [0 for _ in range(args.num_clients)]

    # Metrics storage
    train_losses: List[float] = []
    test_accuracies: List[float] = []
    comm_cum: List[float] = []

    cumulative_comm = 0.0
    rng = np.random.RandomState(args.seed)

    m_per_round = max(1, int(np.ceil(args.client_fraction * args.num_clients)))
    r = int(max(0, min(args.subspace_dim, D)))

    for t in range(args.num_iterations):
        # Subspace for this round
        if args.algo.lower() == "sfedavg":
            P_t = orthonormal_basis(D, r, rng)
        else:
            P_t = np.zeros((D, 0), dtype=np.float64)  # No projection means identity effect for gradient; handled below

        # Deterministic client selection: rolling window
        start = (t * m_per_round) % args.num_clients
        selected = [((start + j) % args.num_clients) for j in range(m_per_round)]

        # Communication accounting (floats sent/received)
        # Server -> each client: theta (D). For SFedAvg additionally P_t (D*r).
        # Client -> server: delta (D).
        round_comm = m_per_round * (D + D)  # theta + delta
        if args.algo.lower() == "sfedavg" and r > 0 and r < D:
            round_comm += m_per_round * (D * r)  # send P_t
        cumulative_comm += float(round_comm)

        # Local updates
        deltas = []
        for i in selected:
            local_theta = theta.copy()
            if args.use_momentum_projection and args.algo.lower() == "sfedavg" and r > 0:
                v = apply_projection(v_prev[i], P_t)
            else:
                v = np.zeros_like(v_prev[i])

            # Perform tau local steps of projected momentum SGD
            for _ in range(args.local_steps):
                Xb, yb, ptrs[i] = client_next_batch(X_clients[i], y_clients[i], ptrs[i], args.batch_size)
                g = grad_ce(Xb, yb, local_theta, d_in, n_classes, args.l2_reg)

                if args.algo.lower() == "sfedavg" and r > 0:
                    g = apply_projection(g, P_t)  # one-sided projected momentum

                v = args.momentum * v + g
                local_theta = local_theta - args.learning_rate * v

            delta = local_theta - theta
            v_prev[i] = v
            deltas.append(delta)

        if deltas:
            mean_delta = np.mean(np.stack(deltas, axis=0), axis=0)
            theta = theta + mean_delta

        # Evaluate
        train_loss, test_acc, _ = evaluate_metrics(theta, X_train, y_train, X_test, y_test, d_in, n_classes, args.l2_reg)
        train_losses.append(float(train_loss))
        test_accuracies.append(float(test_acc))
        comm_cum.append(float(cumulative_comm))

    # Prepare results with required structure
    results = {
        "train_loss": {
            "means": [float(x) for x in train_losses],
            "stds": [0.0 for _ in train_losses],
        },
        "test_accuracy": {
            "means": [float(x) for x in test_accuracies],
            "stds": [0.0 for _ in test_accuracies],
        },
        "cumulative_comm_floats": {
            "means": [float(x) for x in comm_cum],
            "stds": [0.0 for _ in comm_cum],
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for experiment artifacts')

    # Algorithm-specific parameters
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate (eta)')
    parser.add_argument('--num_iterations', type=int, default=50, help='Number of federated rounds T')
    parser.add_argument('--batch_size', type=int, default=64, help='Local minibatch size B')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum coefficient mu')
    parser.add_argument('--subspace_dim', type=int, default=20, help='Subspace dimension r')
    parser.add_argument('--local_steps', type=int, default=5, help='Local SGD steps per round tau')
    parser.add_argument('--client_fraction', type=float, default=0.5, help='Client sampling fraction C')
    parser.add_argument('--algo', type=str, default='sfedavg', choices=['fedavg', 'sfedavg'],
                        help='Which algorithm to run: fedavg or sfedavg')
    parser.add_argument('--use_momentum_projection', dest='use_momentum_projection', nargs='?', const=True, type=lambda s: str(s).lower() in ('1','true','yes','y','t'),
                        help='Use momentum projection at block start (MP); pass true/false or omit for true')
    parser.add_argument('--no_use_momentum_projection', dest='use_momentum_projection', action='store_false',
                        help='Disable momentum projection at block start')
    parser.set_defaults(use_momentum_projection=True)

    # Data and model parameters
    parser.add_argument('--num_clients', type=int, default=10, help='Total number of clients N')
    parser.add_argument('--dataset_size', type=int, default=2000, help='Total dataset size (train+test)')
    parser.add_argument('--input_dim', type=int, default=10, help='Input feature dimension d')
    parser.add_argument('--n_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--l2_reg', type=float, default=0.0, help='L2 regularization strength')
    parser.add_argument('--l', type=float, dest='l2_reg', help='Alias for --l2_reg')
    # Partitioning controls
    parser.add_argument('--partition_strategy', type=str, default='noniid_classes',
                        choices=['iid', 'noniid_sorted', 'noniid_classes'],
                        help='Client data partition strategy: iid random split; noniid_sorted contiguous label blocks; noniid_classes each client assigned limited classes')
    parser.add_argument('--classes_per_client', type=int, default=2,
                        help='For noniid_classes strategy: number of dominant classes per client')
    parser.add_argument('--majority_fraction', type=float, default=0.8,
                        help='For noniid_classes: fraction of each client\'s data drawn from its assigned dominant classes')
    # Label noise controls
    parser.add_argument('--label_noise_rate', type=float, default=0.0,
                        help='Proportion of labels to flip on selected clients (e.g., 0.2 for 20%)')
    parser.add_argument('--label_noise_clients_fraction', type=float, default=0.0,
                        help='Fraction of clients to apply label noise to (e.g., 0.3 for 30%)')

    # Misc
    parser.add_argument('--seed', type=int, default=123, help='Random seed for subspace sampling')

    args = parser.parse_args()

    # Create output directories (baseline structure is required)
    os.makedirs(args.out_dir, exist_ok=True)
    baseline_dir = os.path.join(args.out_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    # Run experiment
    results = run_experiment(args)

    # Save results in required location
    with open(os.path.join(baseline_dir, "final_info.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save a snapshot of this script for reproducibility
    try:
        this_file = os.path.abspath(__file__)
        shutil.copyfile(this_file, os.path.join(args.out_dir, "experiment.py"))
    except Exception:
        # If __file__ is unavailable (interactive), skip snapshot
        pass


if __name__ == "__main__":
    main()
