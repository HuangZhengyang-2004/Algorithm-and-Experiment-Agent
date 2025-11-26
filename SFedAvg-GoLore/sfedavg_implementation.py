"""
Subspace-FedAvg (SFedAvg) with Random One-Sided Projections and Momentum Projection (MP)
Implementation based on the paper "Subspace-FedAvg with Random One-Sided Projections"

Key components:
1. Random one-sided subspace sampling from Stiefel manifold St(d,r)
2. Momentum Projection (MP) at round boundaries
3. Projected momentum updates within rounds
4. Communication-efficient federated learning with compression ratio δ = r/d
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from scipy.stats import ortho_group
import warnings
warnings.filterwarnings('ignore')


class StiefelSampler:
    """Sample random orthonormal matrices from Stiefel manifold St(d,r)"""
    
    @staticmethod
    def sample(d: int, r: int) -> np.ndarray:
        """
        Sample P ∈ St(d,r) uniformly at random
        
        Args:
            d: ambient dimension
            r: subspace dimension
            
        Returns:
            P: orthonormal matrix of shape (d, r)
        """
        # Generate random matrix and perform QR decomposition
        A = np.random.randn(d, r)
        Q, _ = np.linalg.qr(A)
        return Q[:, :r]


class SFedAvgServer:
    """
    SFedAvg Server implementation with random one-sided projections
    """
    
    def __init__(self, 
                 d: int,                    # ambient dimension
                 r: int,                    # subspace dimension  
                 learning_rate: float,      # stepsize η
                 momentum: float = 0.0,     # momentum coefficient μ
                 num_clients: int = 10,
                 client_fraction: float = 1.0):
        
        self.d = d
        self.r = r
        self.delta = r / d  # subspace ratio δ = r/d
        self.eta = learning_rate
        self.mu = momentum
        self.num_clients = num_clients
        self.client_fraction = client_fraction
        
        # Global model parameter θ^t
        self.theta = np.random.randn(d) * 0.01
        
        # Current round projector Π_t (will be sampled each round)
        self.Pi_t = None
        self.P_t = None
        
        # Training history
        self.loss_history = []
        self.grad_norm_history = []
        
    def sample_subspace(self):
        """Sample new one-sided subspace at round boundary"""
        # Sample P_t ∈ St(d,r) uniformly at random
        self.P_t = StiefelSampler.sample(self.d, self.r)
        # Form orthoprojector Π_t = P_t P_t^T
        self.Pi_t = self.P_t @ self.P_t.T
        
    def select_clients(self) -> List[int]:
        """Select subset of clients for current round"""
        m = int(self.client_fraction * self.num_clients)
        return np.random.choice(self.num_clients, size=m, replace=False).tolist()
    
    def aggregate_updates(self, client_deltas: List[np.ndarray]):
        """Aggregate client model deltas"""
        m = len(client_deltas)
        # θ^{t+1} ← θ^t + (1/m) Σ Δ_i^t
        aggregate_delta = np.mean(client_deltas, axis=0)
        self.theta = self.theta + aggregate_delta


class SFedAvgClient:
    """
    SFedAvg Client implementation with projected momentum updates
    """
    
    def __init__(self, 
                 client_id: int,
                 data: Tuple[np.ndarray, np.ndarray],  # (X, y)
                 batch_size: int = 32):
        
        self.client_id = client_id
        self.X, self.y = data
        self.batch_size = batch_size
        
        # Previous momentum state for MP (Momentum Projection)
        self.v_prev = None
        
    def sample_minibatch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample minibatch ξ_{i,s} of size B"""
        n_samples = len(self.X)
        indices = np.random.choice(n_samples, size=min(self.batch_size, n_samples), replace=False)
        return self.X[indices], self.y[indices]
    
    def compute_gradient(self, theta: np.ndarray, X_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """
        Compute stochastic gradient ∇F_i(θ; ξ) for linear regression
        Loss: F_i(θ) = (1/2n) Σ (x_j^T θ - y_j)^2
        """
        predictions = X_batch @ theta
        residuals = predictions - y_batch
        gradient = X_batch.T @ residuals / len(X_batch)
        return gradient
    
    def local_update(self, 
                     theta_t: np.ndarray, 
                     Pi_t: np.ndarray,
                     tau: int,          # local steps
                     eta: float,        # stepsize  
                     mu: float) -> np.ndarray:
        """
        Perform τ local steps with projected momentum
        
        Args:
            theta_t: server model θ^t
            Pi_t: projector Π_t for current round
            tau: number of local steps
            eta: stepsize η
            mu: momentum coefficient μ
            
        Returns:
            model_delta: Δ_i^t = θ_{i,τ} - θ^t
        """
        d = len(theta_t)
        
        # Step 1: Momentum Projection (MP) at block start
        if self.v_prev is not None:
            v = Pi_t @ self.v_prev  # v_i^0 ← Π_t v_i^{prev}
        else:
            v = np.zeros(d)  # v_i^0 ← 0
            
        # Step 2: Local iterations
        theta_local = theta_t.copy()  # θ_{i,0} ← θ^t
        
        for s in range(tau):
            # Sample minibatch and compute stochastic gradient
            X_batch, y_batch = self.sample_minibatch()
            g = self.compute_gradient(theta_local, X_batch, y_batch)
            
            # Projected momentum update: v_{i,s+1} ← μ v_{i,s} + Π_t g_{i,s}
            v = mu * v + Pi_t @ g
            
            # Parameter update: θ_{i,s+1} ← θ_{i,s} - η v_{i,s+1}
            theta_local = theta_local - eta * v
        
        # Step 3: Store momentum for next round and return delta
        self.v_prev = v.copy()  # Store v_i^{prev} ← v_{i,τ}
        
        return theta_local - theta_t  # Return Δ_i^t


class SFedAvgTrainer:
    """Main trainer class for SFedAvg algorithm"""
    
    def __init__(self,
                 d: int,
                 r: int, 
                 learning_rate: float,
                 momentum: float = 0.0,
                 local_steps: int = 5,
                 batch_size: int = 32,
                 client_fraction: float = 1.0):
        
        self.d = d
        self.r = r
        self.eta = learning_rate
        self.mu = momentum
        self.tau = local_steps
        self.batch_size = batch_size
        self.client_fraction = client_fraction
        
        # Verify stepsize compatibility (Assumption 6)
        # κ = (L η τ) / (1 - μ) ≤ 1/4
        # For simplicity, we assume L = 1 for linear regression
        L = 1.0
        kappa = (L * learning_rate * local_steps) / (1 - momentum) if momentum < 1 else float('inf')
        if kappa > 0.25:
            print(f"Warning: κ = {kappa:.4f} > 0.25. Consider reducing stepsize or local steps.")
            
        self.server = None
        self.clients = []
        
    def setup_federated_data(self, client_data: List[Tuple[np.ndarray, np.ndarray]]):
        """Setup server and clients with federated data"""
        num_clients = len(client_data)
        
        # Initialize server
        self.server = SFedAvgServer(
            d=self.d,
            r=self.r,
            learning_rate=self.eta,
            momentum=self.mu,
            num_clients=num_clients,
            client_fraction=self.client_fraction
        )
        
        # Initialize clients
        self.clients = []
        for i, data in enumerate(client_data):
            client = SFedAvgClient(
                client_id=i,
                data=data,
                batch_size=self.batch_size
            )
            self.clients.append(client)
    
    def compute_global_loss(self, theta: np.ndarray) -> float:
        """Compute global objective F(θ) = Σ p_i F_i(θ)"""
        total_loss = 0.0
        total_samples = 0
        
        for client in self.clients:
            X, y = client.X, client.y
            predictions = X @ theta
            loss = 0.5 * np.mean((predictions - y) ** 2)
            total_loss += loss * len(X)
            total_samples += len(X)
            
        return total_loss / total_samples
    
    def compute_global_gradient_norm(self, theta: np.ndarray) -> float:
        """Compute ||∇F(θ)||"""
        total_grad = np.zeros(self.d)
        total_samples = 0
        
        for client in self.clients:
            X, y = client.X, client.y
            predictions = X @ theta
            residuals = predictions - y
            grad = X.T @ residuals / len(X)
            total_grad += grad * len(X)
            total_samples += len(X)
            
        global_grad = total_grad / total_samples
        return np.linalg.norm(global_grad)
    
    def train(self, num_rounds: int, verbose: bool = True) -> Dict:
        """
        Execute SFedAvg training for T rounds
        
        Args:
            num_rounds: number of communication rounds T
            verbose: whether to print progress
            
        Returns:
            training_history: dictionary with loss and gradient norm history
        """
        if self.server is None:
            raise ValueError("Must call setup_federated_data first!")
            
        loss_history = []
        grad_norm_history = []
        
        for t in range(num_rounds):
            # Step 1: Sample one-sided subspace Π_t = P_t P_t^T at round boundary
            self.server.sample_subspace()
            
            # Step 2: Select participating clients S_t
            selected_clients = self.server.select_clients()
            
            # Step 3: Parallel client updates
            client_deltas = []
            for i in selected_clients:
                delta_i = self.clients[i].local_update(
                    theta_t=self.server.theta,
                    Pi_t=self.server.Pi_t,
                    tau=self.tau,
                    eta=self.eta,
                    mu=self.mu
                )
                client_deltas.append(delta_i)
            
            # Step 4: Server aggregation
            self.server.aggregate_updates(client_deltas)
            
            # Step 5: Compute metrics
            current_loss = self.compute_global_loss(self.server.theta)
            current_grad_norm = self.compute_global_gradient_norm(self.server.theta)
            
            loss_history.append(current_loss)
            grad_norm_history.append(current_grad_norm)
            
            if verbose and t % 10 == 0:
                print(f"Round {t:3d}: Loss = {current_loss:.6f}, ||∇F|| = {current_grad_norm:.6f}")
        
        return {
            'loss_history': loss_history,
            'grad_norm_history': grad_norm_history,
            'final_theta': self.server.theta.copy(),
            'delta': self.r / self.d  # compression ratio
        }


def generate_federated_regression_data(num_clients: int = 5,
                                     samples_per_client: int = 200,
                                     d: int = 20,
                                     noise_std: float = 0.1,
                                     heterogeneity: float = 0.5) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Generate federated linear regression data with client heterogeneity
    
    Args:
        num_clients: number of clients
        samples_per_client: samples per client
        d: feature dimension
        noise_std: noise level σ
        heterogeneity: client heterogeneity level ζ
        
    Returns:
        client_data: list of (X_i, y_i) for each client
        true_theta: ground truth parameter
    """
    # Generate ground truth parameter
    true_theta = np.random.randn(d)
    
    client_data = []
    
    for i in range(num_clients):
        # Generate client-specific data with heterogeneity
        X_i = np.random.randn(samples_per_client, d)
        
        # Add client heterogeneity by shifting the effective parameter
        client_shift = np.random.randn(d) * heterogeneity
        effective_theta = true_theta + client_shift
        
        # Generate labels with noise
        y_i = X_i @ effective_theta + np.random.randn(samples_per_client) * noise_std
        
        client_data.append((X_i, y_i))
    
    return client_data, true_theta


def test_algorithm_correctness():
    """Test the correctness of SFedAvg implementation"""
    
    print("=" * 60)
    print("Testing SFedAvg Algorithm Correctness")
    print("=" * 60)
    
    # Test parameters
    d = 10          # ambient dimension
    r = 4           # subspace dimension 
    num_clients = 3
    samples_per_client = 100
    
    print(f"Setup: d={d}, r={r}, δ={r/d:.2f}, {num_clients} clients")
    
    # Test 1: Stiefel manifold sampling
    print("\n1. Testing Stiefel manifold sampling...")
    P = StiefelSampler.sample(d, r)
    print(f"   P shape: {P.shape}")
    
    # Verify orthonormality: P^T P = I_r
    orthogonality_error = np.linalg.norm(P.T @ P - np.eye(r))
    print(f"   Orthonormality error ||P^T P - I||: {orthogonality_error:.2e}")
    assert orthogonality_error < 1e-10, "P should be orthonormal!"
    
    # Test projector properties
    Pi = P @ P.T
    print(f"   Π shape: {Pi.shape}")
    
    # Verify projector properties: Π^2 = Π, Π^T = Π
    projection_error = np.linalg.norm(Pi @ Pi - Pi)
    symmetry_error = np.linalg.norm(Pi.T - Pi)
    print(f"   Projection error ||Π^2 - Π||: {projection_error:.2e}")
    print(f"   Symmetry error ||Π^T - Π||: {symmetry_error:.2e}")
    assert projection_error < 1e-10, "Π should be a projector!"
    assert symmetry_error < 1e-10, "Π should be symmetric!"
    
    # Test expected properties E[Π] = δI (approximately)
    num_samples = 1000
    Pi_sum = np.zeros((d, d))
    trace_sum = 0.0
    
    for _ in range(num_samples):
        P_sample = StiefelSampler.sample(d, r)
        Pi_sample = P_sample @ P_sample.T
        Pi_sum += Pi_sample
        trace_sum += np.trace(Pi_sample)
    
    E_Pi = Pi_sum / num_samples
    E_trace = trace_sum / num_samples
    
    print(f"   E[tr(Π)] ≈ {E_trace:.2f} (should be ≈ {r})")
    print(f"   ||E[Π] - δI|| ≈ {np.linalg.norm(E_Pi - (r/d) * np.eye(d)):.4f}")
    
    # Test 2: Generate federated data
    print("\n2. Testing federated data generation...")
    client_data, true_theta = generate_federated_regression_data(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        d=d,
        noise_std=0.1,
        heterogeneity=0.2
    )
    
    print(f"   Generated {len(client_data)} clients")
    print(f"   True parameter norm: {np.linalg.norm(true_theta):.4f}")
    
    for i, (X_i, y_i) in enumerate(client_data):
        print(f"   Client {i}: X shape {X_i.shape}, y shape {y_i.shape}")
    
    # Test 3: SFedAvg training (short run)
    print("\n3. Testing SFedAvg training...")
    
    trainer = SFedAvgTrainer(
        d=d,
        r=r,
        learning_rate=0.01,
        momentum=0.5,
        local_steps=3,
        batch_size=20,
        client_fraction=1.0
    )
    
    trainer.setup_federated_data(client_data)
    
    # Initial metrics
    initial_loss = trainer.compute_global_loss(trainer.server.theta)
    initial_grad_norm = trainer.compute_global_gradient_norm(trainer.server.theta)
    print(f"   Initial loss: {initial_loss:.6f}")
    print(f"   Initial ||∇F||: {initial_grad_norm:.6f}")
    
    # Short training run
    history = trainer.train(num_rounds=20, verbose=False)
    
    final_loss = history['loss_history'][-1]
    final_grad_norm = history['grad_norm_history'][-1]
    print(f"   Final loss: {final_loss:.6f}")
    print(f"   Final ||∇F||: {final_grad_norm:.6f}")
    
    # Verify loss decreases
    assert final_loss < initial_loss, "Loss should decrease during training!"
    print(f"   ✓ Loss decreased by {((initial_loss - final_loss) / initial_loss * 100):.2f}%")
    
    # Test 4: Compare with standard FedAvg (no subspace)
    print("\n4. Comparing with standard FedAvg...")
    
    # Standard FedAvg (r = d, so δ = 1)
    trainer_standard = SFedAvgTrainer(
        d=d,
        r=d,  # full space
        learning_rate=0.01,
        momentum=0.5,
        local_steps=3,
        batch_size=20,
        client_fraction=1.0
    )
    
    trainer_standard.setup_federated_data(client_data)
    history_standard = trainer_standard.train(num_rounds=20, verbose=False)
    
    subspace_final_loss = final_loss
    standard_final_loss = history_standard['loss_history'][-1]
    
    print(f"   Subspace FedAvg final loss: {subspace_final_loss:.6f}")
    print(f"   Standard FedAvg final loss: {standard_final_loss:.6f}")
    print(f"   Communication reduction: {(1 - r/d) * 100:.1f}%")
    
    # Test 5: Algorithm components
    print("\n5. Testing algorithm components...")
    
    # Test client update
    client = trainer.clients[0]
    theta_test = np.random.randn(d) * 0.1
    Pi_test = StiefelSampler.sample(d, r) @ StiefelSampler.sample(d, r).T
    
    delta = client.local_update(theta_test, Pi_test, tau=5, eta=0.01, mu=0.5)
    print(f"   Client update delta norm: {np.linalg.norm(delta):.6f}")
    
    # Test momentum projection
    v_prev = np.random.randn(d)
    v_projected = Pi_test @ v_prev
    projection_reduction = 1 - np.linalg.norm(v_projected) / np.linalg.norm(v_prev)
    print(f"   Momentum projection reduction: {projection_reduction * 100:.2f}%")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! SFedAvg implementation appears correct.")
    print("=" * 60)
    
    return history, history_standard


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run correctness tests
    test_algorithm_correctness()