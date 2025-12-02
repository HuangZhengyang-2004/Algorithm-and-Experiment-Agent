# Random Search hyperparameter tuning for experiment.run_experiment

from experiment import run_experiment
import random
import numpy as np
import json
import os
import math
import argparse


SEARCH_SPACE = {
    "learning_rate": {"type": "float", "min": 0.003, "max": 0.08, "scaling": "log"},
    "subspace_dim": {"type": "int", "min": 5, "max": 25},
    "momentum": {"type": "float", "min": 0.7, "max": 0.99, "scaling": "linear"},
    "local_steps": {"type": "int", "min": 10, "max": 30}
}


def sample_parameters(search_space, rng: random.Random):
    params = {}
    for name, spec in search_space.items():
        if spec["type"] == "float":
            if spec.get("scaling") == "log":
                log_min = math.log(spec["min"])
                log_max = math.log(spec["max"])
                value = math.exp(rng.uniform(log_min, log_max))
            else:
                value = rng.uniform(spec["min"], spec["max"])
        elif spec["type"] == "int":
            # random.Random.randint is inclusive on both ends; spec requires max+1
            value = rng.randint(spec["min"], spec["max"] + 1)
        elif spec["type"] == "categorical":
            value = rng.choice(spec["values"])
        else:
            raise ValueError(f"Unsupported parameter type for {name}: {spec['type']}")
        params[name] = value
    return params


def get_final_metrics(result_dict):
    metrics = {}
    try:
        if "test_accuracy" in result_dict and "means" in result_dict["test_accuracy"]:
            ta = result_dict["test_accuracy"]["means"]
            metrics["final_test_accuracy"] = float(ta[-1]) if len(ta) > 0 else None
        if "train_loss" in result_dict and "means" in result_dict["train_loss"]:
            tl = result_dict["train_loss"]["means"]
            metrics["final_train_loss"] = float(tl[-1]) if len(tl) > 0 else None
    except Exception:
        pass
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="tuning", help="Directory to store tuning results")
    parser.add_argument("--trials", type=int, default=12, help="Number of random search trials")
    args = parser.parse_args()

    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)

    rng = random.Random(42)

    base_params = {
        "learning_rate": 0.03,
        "num_iterations": 120,
        "batch_size": 32,
        "momentum": 0.9,
        "subspace_dim": 10,
        "local_steps": 20,
        "client_fraction": 0.1,
        "num_clients": 50,
        "dataset_size": 8000,
        "input_dim": 10,
        "n_classes": 3,
        "use_momentum_projection": True,
        "algo": "sfedavg",
        "seed": 2,
        "partition_strategy": "noniid_classes",
        "classes_per_client": 2
    }

    all_results = []
    best_idx = None
    best_score = None  # maximize test accuracy; fallback to minimize train loss

    for i in range(1, args.trials + 1):
        trial_dir = os.path.join(out_root, f"trial_{i}")
        os.makedirs(trial_dir, exist_ok=True)

        sampled = sample_parameters(SEARCH_SPACE, rng)
        trial_params = dict(base_params)
        trial_params.update(sampled)

        # Construct args object for run_experiment (it accepts dict/Namespace)
        exp_args = dict(trial_params)

        status = "success"
        result = {}
        error_msg = ""
        try:
            result = run_experiment(exp_args)
            # Save per-trial result
            with open(os.path.join(trial_dir, "final_info.json"), "w") as f:
                json.dump(result, f, indent=2)
            # Also save parameters used
            with open(os.path.join(trial_dir, "params.json"), "w") as f:
                json.dump(trial_params, f, indent=2)
        except Exception as e:
            status = "failure"
            error_msg = str(e)

        metrics = get_final_metrics(result) if status == "success" else {}

        # Select best: prefer highest final_test_accuracy; if unavailable, lowest final_train_loss
        score = None
        score_mode = None
        if status == "success":
            if metrics.get("final_test_accuracy") is not None:
                score = metrics["final_test_accuracy"]
                score_mode = "max"
            elif metrics.get("final_train_loss") is not None:
                score = -metrics["final_train_loss"]  # invert to maximize
                score_mode = "min(train_loss)"
        if status == "success" and score is not None:
            if best_score is None or score > best_score:
                best_score = score
                best_idx = i

        all_results.append({
            "trial": i,
            "parameters": trial_params,
            "metrics": metrics,
            "status": status,
            "error": error_msg
        })

    summary = {
        "best_config": {},
        "best_result": {},
        "all_results": all_results
    }

    if best_idx is not None:
        best_trial_dir = os.path.join(out_root, f"trial_{best_idx}")
        # Load best config and result back from files
        try:
            with open(os.path.join(best_trial_dir, "params.json"), "r") as f:
                summary["best_config"] = json.load(f)
        except Exception:
            # fallback to in-memory
            summary["best_config"] = next((r["parameters"] for r in all_results if r["trial"] == best_idx), {})
        try:
            with open(os.path.join(best_trial_dir, "final_info.json"), "r") as f:
                summary["best_result"] = json.load(f)
        except Exception:
            summary["best_result"] = next((r["metrics"] for r in all_results if r["trial"] == best_idx), {})

    with open(os.path.join(out_root, "tuning_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Tuning complete. Summary saved to {os.path.join(out_root, 'tuning_summary.json')}")


if __name__ == "__main__":
    main()
# Random Search hyperparameter tuning for experiment.run_experiment (robustness_to_label_noise scenario)

from experiment import run_experiment
import random
import numpy as np
import json
import os
import math
import argparse


SEARCH_SPACE = {
    "learning_rate": {"type": "float", "min": 0.003, "max": 0.08, "scaling": "log"},
    "subspace_dim": {"type": "int", "min": 5, "max": 30},
    "momentum": {"type": "float", "min": 0.6, "max": 0.95, "scaling": "linear"},
    "client_fraction": {"type": "float", "min": 0.2, "max": 0.6, "scaling": "linear"}
}


def sample_parameters(search_space, rng: random.Random):
    params = {}
    for name, spec in search_space.items():
        if spec["type"] == "float":
            if spec.get("scaling") == "log":
                log_min = math.log(spec["min"])
                log_max = math.log(spec["max"])
                value = math.exp(rng.uniform(log_min, log_max))
            else:
                value = rng.uniform(spec["min"], spec["max"])
        elif spec["type"] == "int":
            # random.Random.randint is inclusive on both ends; add 1 to upper bound
            value = rng.randint(spec["min"], spec["max"])
        elif spec["type"] == "categorical":
            value = rng.choice(spec["values"])
        else:
            raise ValueError(f"Unsupported parameter type for {name}: {spec['type']}")
        params[name] = value
    return params


def get_final_metrics(result_dict):
    metrics = {}
    try:
        if "test_accuracy" in result_dict and "means" in result_dict["test_accuracy"]:
            ta = result_dict["test_accuracy"]["means"]
            metrics["final_test_accuracy"] = float(ta[-1]) if len(ta) > 0 else None
        if "train_loss" in result_dict and "means" in result_dict["train_loss"]:
            tl = result_dict["train_loss"]["means"]
            metrics["final_train_loss"] = float(tl[-1]) if len(tl) > 0 else None
    except Exception:
        pass
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="tuning", help="Directory to store tuning results")
    parser.add_argument("--trials", type=int, default=12, help="Number of random search trials")
    args = parser.parse_args()

    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)

    rng = random.Random(42)

    # Base parameters for robustness_to_label_noise_(client-local_corruption)
    base_params = {
        "learning_rate": 0.05,
        "num_iterations": 100,
        "batch_size": 64,
        "momentum": 0.9,
        "subspace_dim": 15,
        "local_steps": 5,
        "client_fraction": 0.3,
        "num_clients": 30,
        "dataset_size": 6000,
        "input_dim": 10,
        "n_classes": 3,
        "use_momentum_projection": True,
        "algo": "sfedavg",
        "seed": 3,
        # Scenario-specific controls
        "partition_strategy": "noniid_classes",
        "classes_per_client": 2,
        "label_noise_rate": 0.2,
        "label_noise_clients_fraction": 0.3
    }

    all_results = []
    best_idx = None
    best_score = None  # maximize test accuracy; fallback to minimize train loss

    for i in range(1, args.trials + 1):
        trial_dir = os.path.join(out_root, f"trial_{i}")
        os.makedirs(trial_dir, exist_ok=True)

        sampled = sample_parameters(SEARCH_SPACE, rng)
        trial_params = dict(base_params)
        trial_params.update(sampled)

        # Construct args object for run_experiment (it accepts dict/Namespace)
        exp_args = dict(trial_params)

        status = "success"
        result = {}
        error_msg = ""
        try:
            result = run_experiment(exp_args)
            # Save per-trial result
            with open(os.path.join(trial_dir, "final_info.json"), "w") as f:
                json.dump(result, f, indent=2)
            # Also save parameters used
            with open(os.path.join(trial_dir, "params.json"), "w") as f:
                json.dump(trial_params, f, indent=2)
        except Exception as e:
            status = "failure"
            error_msg = str(e)

        metrics = get_final_metrics(result) if status == "success" else {}

        # Select best: prefer highest final_test_accuracy; if unavailable, lowest final_train_loss
        score = None
        if status == "success":
            if metrics.get("final_test_accuracy") is not None:
                score = metrics["final_test_accuracy"]
            elif metrics.get("final_train_loss") is not None:
                score = -metrics["final_train_loss"]  # invert to maximize
        if status == "success" and score is not None:
            if best_score is None or score > best_score:
                best_score = score
                best_idx = i

        all_results.append({
            "trial": i,
            "parameters": trial_params,
            "metrics": metrics,
            "status": status,
            "error": error_msg
        })

    summary = {
        "best_config": {},
        "best_result": {},
        "all_results": all_results
    }

    if best_idx is not None:
        best_trial_dir = os.path.join(out_root, f"trial_{best_idx}")
        # Load best config and result back from files
        try:
            with open(os.path.join(best_trial_dir, "params.json"), "r") as f:
                summary["best_config"] = json.load(f)
        except Exception:
            # fallback to in-memory
            summary["best_config"] = next((r["parameters"] for r in all_results if r["trial"] == best_idx), {})
        try:
            with open(os.path.join(best_trial_dir, "final_info.json"), "r") as f:
                summary["best_result"] = json.load(f)
        except Exception:
            summary["best_result"] = next((r["metrics"] for r in all_results if r["trial"] == best_idx), {})

    with open(os.path.join(out_root, "tuning_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Tuning complete. Summary saved to {os.path.join(out_root, 'tuning_summary.json')}")


if __name__ == "__main__":
    main()
