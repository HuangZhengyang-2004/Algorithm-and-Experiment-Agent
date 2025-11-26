# Federated Averaging (FedAvg) vs Subspace Federated Averaging (SFedAvg)
# Complete experimental framework implementing the algorithms described in the pseudocode.

import argparse
import json
import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def softmax(logits):
    # Stable softmax over rows
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(W, X, y, num_classes):
    # Compute mean cross-entropy loss for linear-softmax classifier
    logits = X @ W  # (n, k)
    probs = softmax(logits)
    # One-hot targets
    n = X.shape[0]
    y_onehot = np.zeros((n, num_classes), dtype=np.float64)
    y_onehot[np.arange(n), y] = 1.0
    # Avoid log(0)
    eps = 1e-12
    log_probs = np.log(probs + eps)
    loss = -np.sum(y_onehot * log_probs) / n
    return loss

def gradient(W, X, y, num_classes):
    # Gradient of mean cross-entropy w.r.t. W for linear-softmax
    logits = X @ W  # (n, k)
    probs = softmax(logits)  # (n, k)
    n = X.shape[0]
    y_onehot = np.zeros((n, num_classes), dtype=np.float64)
    y_onehot[np.arange(n), y] = 1.0
    # dL/dW = X^T (probs - y_onehot) / n
    grad = (X.T @ (probs - y_onehot)) / n  # (d, k)
    return grad

def accuracy(W, X, y):
    logits = X @ W
    preds = np.argmax(logits, axis=1)
    return (preds == y).mean()

def make_random_subspace_projector(d, r, rng):
    # Draw random Gaussian matrix and compute orthonormal basis via QR
    A = rng.normal(size=(d, r))
    Q, _ = np.linalg.qr(A)  # Q: d x r with orthonormal columns
    P = Q[:, :r]
    Pi = P @ P.T  # d x d
    return P, Pi

def partition_indices_iid(n_samples, n_clients, rng):
    # Partition indices evenly across clients (iid)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    splits = np.array_split(indices, n_clients)
    return [np.array(s, dtype=np.int64) for s in splits]

def partition_indices_dirichlet(y, n_clients, alpha, rng):
    # Non-IID label skew via Dirichlet over class proportions
    classes = np.unique(y)
    client_lists = [[] for _ in range(n_clients)]
    for c in classes:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        # Sample class proportions across clients
        p = rng.dirichlet(np.full(n_clients, alpha, dtype=float))
        # Allocation counts with rounding that preserves total
        expected = p * len(c_idx)
        counts = np.floor(expected).astype(int)
        remainder = len(c_idx) - counts.sum()
        if remainder > 0:
            frac = expected - counts
            order = np.argsort(frac)[::-1]
            for j in order[:remainder]:
                counts[j] += 1
        # Assign contiguous slices
        start = 0
        for j, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            sl = c_idx[start:start+cnt]
            client_lists[j].extend(sl.tolist())
            start += cnt
    # Convert to numpy arrays (allow empty clients)
    return [np.array(sorted(lst), dtype=np.int64) if lst else np.array([], dtype=np.int64) for lst in client_lists]

def add_label_noise(y, noise_rate=0.2, n_classes=10, seed=None):
    """
    Randomly flip a proportion of labels to simulate label noise.
    Flips are uniform to a different class, applied to the provided label array.
    """
    y_noisy = y.copy()
    if noise_rate <= 0 or n_classes <= 1:
        return y_noisy
    n = y_noisy.shape[0]
    n_flip = int(np.floor(noise_rate * n))
    if n_flip <= 0:
        return y_noisy
    rng = np.random.default_rng(seed)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    orig = y_noisy[flip_idx]
    # Sample labels in [0, n_classes-2], then shift to avoid original label
    rand = rng.integers(0, n_classes - 1, size=n_flip)
    new = rand + (rand >= orig)
    y_noisy[flip_idx] = new.astype(np.int64)
    return y_noisy

def run_experiment(args):
    # Extract parameters from args for a single-run experiment
    out_dir = args.out_dir
    n_clients = args.n_clients
    client_fraction = args.client_fraction
    rounds = args.rounds
    local_steps = args.local_steps
    stepsize = args.stepsize
    momentum = args.momentum
    batch_size = args.batch_size
    subspace_rank_fraction = args.subspace_rank_fraction
    heterogeneity_alpha = args.heterogeneity_alpha
    label_noise_fraction = getattr(args, "label_noise_fraction", 0.0)
    seed = args.seed

    # Load dataset (digits ~ MNIST-like, small and local)
    ds = load_digits()
    X = ds.data.astype(np.float64)
    y = ds.target.astype(np.int64)
    num_classes = len(np.unique(y))
    d = X.shape[1]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Partition training data across clients
    rng = np.random.default_rng(seed)
    if heterogeneity_alpha is not None:
        client_indices = partition_indices_dirichlet(y_train, n_clients, heterogeneity_alpha, rng)
    else:
        client_indices = partition_indices_iid(len(X_train), n_clients, rng)

    # Prepare per-client label noise (fixed across rounds)
    y_train_effective = y_train.copy()
    if label_noise_fraction > 0.0:
        for i in range(n_clients):
            idx = client_indices[i]
            if idx.size == 0:
                continue
            y_train_effective[idx] = add_label_noise(
                y_train_effective[idx],
                noise_rate=label_noise_fraction,
                n_classes=num_classes,
                seed=seed * 1000 + i
            )

    # Model initialization (shared init for both algorithms)
    k = num_classes
    W_fedavg = np.zeros((d, k), dtype=np.float64)
    W_sfedavg = np.zeros((d, k), dtype=np.float64)

    # Initialize per-client momentum states
    v_prev_fedavg = [np.zeros((d, k), dtype=np.float64) for _ in range(n_clients)]
    v_prev_sfedavg = [np.zeros((d, k), dtype=np.float64) for _ in range(n_clients)]

    # Metrics storage
    train_loss_fedavg = []
    train_loss_sfedavg = []
    test_acc_fedavg = []
    test_acc_sfedavg = []
    comm_cost_fedavg = []
    comm_cost_sfedavg = []
    eff_update_dims_fedavg = []
    eff_update_dims_sfedavg = []
    # Track cumulative communication and effective update dimensions to ensure monotonic progress
    total_comm_cost_fedavg = 0.0
    total_comm_cost_sfedavg = 0.0
    total_eff_dims_fedavg = 0.0
    total_eff_dims_sfedavg = 0.0

    # Determine subspace rank
    r = max(1, int(subspace_rank_fraction * d))
    r = min(r, d)

    # Per-round loop
    N = n_clients
    m = max(1, int(client_fraction * N))

    for t in range(rounds):
        # One-sided subspace for this round (held fixed within round)
        round_rng = np.random.default_rng(seed + t)
        _, Pi_t = make_random_subspace_projector(d, r, round_rng)

        # Sample client subset
        subset = rng.choice(N, size=m, replace=False)

        # Per-client deltas
        deltas_fedavg = []
        deltas_sfedavg = []
        participated_clients = 0

        for i in subset:
            idx = client_indices[i]
            if idx.size == 0:
                continue
            participated_clients += 1
            Xi = X_train[idx]
            yi = y_train_effective[idx]

            # Initialize local copies from global model (required)
            W_local_fedavg = W_fedavg.copy()
            W_local_sfedavg = W_sfedavg.copy()

            # Momentum Projection (MP) for SFedAvg at block start
            v_fed = v_prev_fedavg[i]  # FedAvg uses previous momentum as-is
            v_sfed = Pi_t @ v_prev_sfedavg[i]  # project previous momentum

            # Deterministic per-client minibatch RNG to ensure fair comparison
            mb_rng = np.random.default_rng((i + 1) * 10_000 + t)

            # Local steps
            n_i = Xi.shape[0]
            for s in range(local_steps):
                # Sample minibatch indices
                batch_idx = mb_rng.integers(low=0, high=n_i, size=batch_size)
                Xb = Xi[batch_idx]
                yb = yi[batch_idx]

                # Compute true gradient from data
                grad_fed = gradient(W_local_fedavg, Xb, yb, num_classes)
                grad_sfed = gradient(W_local_sfedavg, Xb, yb, num_classes)

                # Momentum updates
                v_fed = momentum * v_fed + grad_fed
                v_sfed = momentum * v_sfed + (Pi_t @ grad_sfed)

                # Parameter updates
                W_local_fedavg = W_local_fedavg - stepsize * v_fed
                W_local_sfedavg = W_local_sfedavg - stepsize * v_sfed

            # Form deltas and store end-of-block momentum
            delta_fed = W_local_fedavg - W_fedavg
            delta_sfed = W_local_sfedavg - W_sfedavg
            deltas_fedavg.append(delta_fed)
            deltas_sfedavg.append(delta_sfed)
            v_prev_fedavg[i] = v_fed
            v_prev_sfedavg[i] = v_sfed

        # Aggregation (FedAvg style)
        mean_delta_fed = np.mean(deltas_fedavg, axis=0) if deltas_fedavg else np.zeros_like(W_fedavg)
        mean_delta_sfed = np.mean(deltas_sfedavg, axis=0) if deltas_sfedavg else np.zeros_like(W_sfedavg)
        W_fedavg = W_fedavg + mean_delta_fed
        W_sfedavg = W_sfedavg + mean_delta_sfed

        # Evaluation on full train/test sets
        tl_fed = cross_entropy_loss(W_fedavg, X_train, y_train, num_classes)
        tl_sfed = cross_entropy_loss(W_sfedavg, X_train, y_train, num_classes)
        ta_fed = accuracy(W_fedavg, X_test, y_test)
        ta_sfed = accuracy(W_sfedavg, X_test, y_test)

        train_loss_fedavg.append(float(tl_fed))
        train_loss_sfedavg.append(float(tl_sfed))
        test_acc_fedavg.append(float(ta_fed))
        test_acc_sfedavg.append(float(ta_sfed))

        # Communication accounting (cumulative) per round
        m_t = participated_clients
        # FedAvg: send global model (d*k) and receive deltas (d*k) for m_t clients -> 2*m_t*d*k (units)
        comm_fed = 2 * m_t * d * k
        # SFedAvg: send global model and projector Pi (d*r) + receive deltas (d*k) -> 2*m_t*d*k + m_t*d*r (units)
        comm_sfed = 2 * m_t * d * k + m_t * d * r
        total_comm_cost_fedavg += float(comm_fed)
        total_comm_cost_sfedavg += float(comm_sfed)
        comm_cost_fedavg.append(total_comm_cost_fedavg)
        comm_cost_sfedavg.append(total_comm_cost_sfedavg)

        # Effective updated coordinates (cumulative heuristic) per round
        total_eff_dims_fedavg += float(d * k * m_t)
        total_eff_dims_sfedavg += float(r * k * m_t)
        eff_update_dims_fedavg.append(total_eff_dims_fedavg)
        eff_update_dims_sfedavg.append(total_eff_dims_sfedavg)

    # Prepare results in required format with means/stds arrays
    zeros_fed = [0.0] * rounds
    zeros_sfed = [0.0] * rounds

    results = {
        "train_loss_fedavg": {
            "means": train_loss_fedavg,
            "stds": zeros_fed
        },
        "train_loss_sfedavg": {
            "means": train_loss_sfedavg,
            "stds": zeros_sfed
        },
        "test_acc_fedavg": {
            "means": test_acc_fedavg,
            "stds": zeros_fed
        },
        "test_acc_sfedavg": {
            "means": test_acc_sfedavg,
            "stds": zeros_sfed
        },
        "comm_cost_fedavg": {
            "means": comm_cost_fedavg,
            "stds": zeros_fed
        },
        "comm_cost_sfedavg": {
            "means": comm_cost_sfedavg,
            "stds": zeros_sfed
        },
        "effective_update_dims_fedavg": {
            "means": eff_update_dims_fedavg,
            "stds": zeros_fed
        },
        "effective_update_dims_sfedavg": {
            "means": eff_update_dims_sfedavg,
            "stds": zeros_sfed
        }
    }

    return results

def main():
    parser = argparse.ArgumentParser(description="Federated vs Subspace Federated Averaging on digits dataset")
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory to store results')
    parser.add_argument('--n_clients', type=int, default=20, help='Number of clients')
    parser.add_argument('--client_fraction', type=float, default=0.5, help='Fraction of clients per round (0,1]')
    parser.add_argument('--rounds', type=int, default=20, help='Number of federated rounds')
    parser.add_argument('--local_steps', type=int, default=2, help='Local steps per client per round (tau)')
    parser.add_argument('--stepsize', type=float, default=0.3, help='Learning rate (eta)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum coefficient (mu)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size (B)')
    parser.add_argument('--subspace_rank_fraction', type=float, default=0.25, help='Fraction of feature dim for subspace rank r/d')
    parser.add_argument('--heterogeneity_alpha', type=float, default=None, help='Dirichlet alpha for non-IID label skew; if omitted, uses IID partitioning')
    parser.add_argument('--label_noise_fraction', type=float, default=0.0, help='Fraction of labels to uniformly flip at random per client (0 to 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--enable_tuning', action='store_true', help='Enable batch parameter tuning')
    args = parser.parse_args()

    # Validate arguments
    if not (0 < args.client_fraction <= 1.0):
        raise ValueError("client_fraction must be in (0, 1].")
    if args.rounds < 1 or args.local_steps < 1 or args.n_clients < 1:
        raise ValueError("rounds, local_steps, and n_clients must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if not (0 < args.subspace_rank_fraction <= 1.0):
        raise ValueError("subspace_rank_fraction must be in (0, 1].")
    if args.heterogeneity_alpha is not None and args.heterogeneity_alpha <= 0:
        raise ValueError("heterogeneity_alpha must be > 0.")
    if not (0.0 <= args.label_noise_fraction <= 1.0):
        raise ValueError("label_noise_fraction must be in [0, 1].")

    # Create output directory and subdirectories
    os.makedirs(args.out_dir, exist_ok=True)

    if args.enable_tuning:
        # Define parameter configurations to test
        tuning_configs = [
      {
        "stepsize": 0.25,
        "momentum": 0.95,
        "subspace_rank_fraction": 0.2,
        "rationale": "Strong smoothing with tighter subspace to attenuate noisy gradients while maintaining reasonable step size."
      },
      {
        "stepsize": 0.3,
        "momentum": 0.9,
        "subspace_rank_fraction": 0.25,
        "rationale": "Balanced control (close to baseline) to measure gains primarily from label-noise-aware projection."
      },
      {
        "stepsize": 0.2,
        "momentum": 0.9,
        "subspace_rank_fraction": 0.3,
        "rationale": "Lower LR to reduce noise-induced drift; larger rank to preserve capacity and accuracy."
      },
      {
        "stepsize": 0.3,
        "momentum": 0.8,
        "subspace_rank_fraction": 0.2,
        "rationale": "Reduced momentum to limit accumulation of corrupted signals; tighter projection for robustness."
      },
      {
        "stepsize": 0.35,
        "momentum": 0.9,
        "subspace_rank_fraction": 0.15,
        "rationale": "Faster steps with strong projection to control instability from noise; tests speed vs projection strength."
      },
      {
        "stepsize": 0.25,
        "momentum": 0.85,
        "subspace_rank_fraction": 0.3,
        "rationale": "Mid LR and momentum with larger rank for capacity; aims for stable convergence with good accuracy."
      },
      {
        "stepsize": 0.3,
        "momentum": 0.95,
        "subspace_rank_fraction": 0.35,
        "rationale": "Aggressive momentum and larger rank if projection sufficiently suppresses noise; targets fastest robust convergence."
      },
      {
        "stepsize": 0.2,
        "momentum": 0.8,
        "subspace_rank_fraction": 0.25,
        "rationale": "Conservative setting to maximize robustness against label flips; baseline rank preserves generalization."
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

            # Override parameters with config values
            for param_name, param_value in config.items():
                setattr(args, param_name, param_value)

            # Run experiment with this configuration
            results = run_experiment(args)

            # Save results
            with open(os.path.join(config_dir, "final_info.json"), "w") as f:
                json.dump(results, f, indent=2)

            all_results[f"config_{idx}"] = {
                "parameters": config,
                "results": results
            }

        # Save aggregated results
        with open(os.path.join(args.out_dir, "tuning", "all_configs.json"), "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        # Normal single-run mode (baseline)
        baseline_dir = os.path.join(args.out_dir, "baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        results = run_experiment(args)
        with open(os.path.join(baseline_dir, "final_info.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Snapshot this experiment file for reproducibility
    try:
        src_path = os.path.abspath(__file__)
        with open(src_path, "r") as src_f:
            src_code = src_f.read()
        with open(os.path.join(args.out_dir, "experiment.py"), "w") as dst_f:
            dst_f.write(src_code)
    except Exception:
        # If snapshot fails, continue without raising to keep experiment robust
        pass

if __name__ == "__main__":
    main()
