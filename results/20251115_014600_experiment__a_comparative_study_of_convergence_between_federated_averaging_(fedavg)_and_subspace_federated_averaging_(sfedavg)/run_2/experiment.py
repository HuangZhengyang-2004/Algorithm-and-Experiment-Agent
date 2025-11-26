import argparse
import json
import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def predict(X, theta, num_classes):
    W = theta[:-num_classes].reshape(X.shape[1], num_classes)
    b = theta[-num_classes:]
    z = X @ W + b
    return softmax(z)

def gradient(X, y, theta, num_classes):
    W = theta[:-num_classes].reshape(X.shape[1], num_classes)
    b = theta[-num_classes:]
    n = X.shape[0]
    y_pred = predict(X, theta, num_classes)
    grad_W = (X.T @ (y_pred - y)) / n
    grad_b = np.mean(y_pred - y, axis=0)
    return np.concatenate([grad_W.ravel(), grad_b])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--algorithm', type=str, default='SFedAvg', choices=['FedAvg', 'SFedAvg'])
    parser.add_argument('--T', type=int, default=200, help='Number of rounds')
    parser.add_argument('--tau', type=int, default=5, help='Local steps')
    parser.add_argument('--C', type=float, default=0.1, help='Client fraction')
    parser.add_argument('--eta', type=float, default=0.01, help='Step size')
    parser.add_argument('--mu', type=float, default=0.9, help='Momentum')
    parser.add_argument('--B', type=int, default=32, help='Batch size')
    parser.add_argument('--r', type=int, default=100, help='Subspace dimension for SFedAvg')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int32)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert labels to one-hot encoding
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    # Create clients
    num_clients = 100
    client_data = []
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    assignments = np.array_split(indices, num_clients)
    for i in range(num_clients):
        client_data.append((X_train[assignments[i]], y_train_onehot[assignments[i]]))
    
    # Initialize model parameters with Glorot uniform initialization
    d = X_train.shape[1] * num_classes + num_classes  # W: 784*10, b: 10 -> 7850
    np.random.seed(42)
    # Glorot uniform initialization for weights, zeros for biases
    scale = np.sqrt(6.0 / (X_train.shape[1] + num_classes))
    W = np.random.uniform(-scale, scale, (X_train.shape[1], num_classes))
    b = np.zeros(num_classes)
    theta = np.concatenate([W.ravel(), b])
    
    # For SFedAvg: dictionary to store client momentum
    client_momentum = {}
    
    # Results storage
    results = {
        "test_accuracy": {"means": [], "stds": []},
        "training_loss": {"means": [], "stds": []}
    }
    
    T = args.T
    tau = args.tau
    C = args.C
    eta = args.eta
    mu = args.mu
    B = args.B
    r = args.r
    algorithm = args.algorithm
    
    print(f"Starting training with algorithm: {algorithm}")
    for t in range(T):
        if algorithm == "SFedAvg":
            # Generate random projection matrix P_t of size d x r using uniform distribution
            random_matrix = np.random.uniform(-1.0, 1.0, (d, r))
            P_t, _ = np.linalg.qr(random_matrix)
        
        # Select clients
        m = max(1, int(C * num_clients))
        client_indices = np.random.choice(num_clients, size=m, replace=False)
        deltas = []
        for i in client_indices:
            X_i, y_i = client_data[i]
            if algorithm == "FedAvg":
                theta_local = theta.copy()
                for s in range(tau):
                    # Sample minibatch
                    if len(X_i) < B:
                        idx = np.random.choice(len(X_i), size=len(X_i), replace=False)
                    else:
                        idx = np.random.choice(len(X_i), size=B, replace=False)
                    X_batch = X_i[idx]
                    y_batch = y_i[idx]
                    g = gradient(X_batch, y_batch, theta_local, num_classes)
                    theta_local = theta_local - eta * g
                delta = theta_local - theta
                deltas.append(delta)
            else:  # SFedAvg
                if i in client_momentum:
                    v_prev = client_momentum[i]
                    # Project: v0 = P_t @ (P_t.T @ v_prev)
                    v0 = P_t @ (P_t.T @ v_prev)
                else:
                    v0 = np.zeros(d)
                theta_local = theta.copy()
                v = v0.copy()
                for s in range(tau):
                    if len(X_i) < B:
                        idx = np.random.choice(len(X_i), size=len(X_i), replace=False)
                    else:
                        idx = np.random.choice(len(X_i), size=B, replace=False)
                    X_batch = X_i[idx]
                    y_batch = y_i[idx]
                    g = gradient(X_batch, y_batch, theta_local, num_classes)
                    # Project the gradient: projected_g = P_t @ (P_t.T @ g)
                    projected_g = P_t @ (P_t.T @ g)
                    v = mu * v + projected_g
                    theta_local = theta_local - eta * v
                delta = theta_local - theta
                deltas.append(delta)
                client_momentum[i] = v  # store for next round

        # Aggregate deltas
        avg_delta = np.mean(deltas, axis=0)
        theta = theta + avg_delta

        # Evaluation
        y_pred_test = predict(X_test, theta, num_classes)
        test_acc = accuracy_score(y_test, np.argmax(y_pred_test, axis=1))
        y_pred_train = predict(X_train, theta, num_classes)
        train_loss = cross_entropy_loss(y_pred_train, y_train_onehot)

        results["test_accuracy"]["means"].append(test_acc)
        results["test_accuracy"]["stds"].append(0.0)
        results["training_loss"]["means"].append(train_loss)
        results["training_loss"]["stds"].append(0.0)
        
        if t % 10 == 0:
            print(f"Round {t}, Test Accuracy: {test_acc:.4f}, Training Loss: {train_loss:.4f}")

    # Save results
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
