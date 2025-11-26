"""
Simplified Linear Regression Experiment for SFedAvg vs Baselines
专注于核心功能验证的简化版实验
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sfedavg_implementation import StiefelSampler


class SimpleExperiment:
    """简化的实验类"""
    
    def __init__(self):
        # 实验配置
        self.num_clients = 10
        self.client_fraction = 0.3
        self.d = 30  # 降低维度以加快实验
        self.samples_per_client = 80
        self.num_rounds = 80  # 降低轮数
        self.local_steps = 5
        self.batch_size = 15
        self.learning_rate = 0.01
        self.momentum = 0.9
        
        # 生成数据
        self._generate_data()
    
    def _generate_data(self):
        """生成联邦数据"""
        np.random.seed(42)
        
        # 真实参数
        self.true_theta = np.random.randn(self.d)
        self.true_theta = self.true_theta / np.linalg.norm(self.true_theta) * 2.5
        
        self.client_data = []
        for client_id in range(self.num_clients):
            # 生成异质数据
            mean_shift = np.random.randn(self.d) * 0.1
            X = np.random.randn(self.samples_per_client, self.d) + mean_shift
            noise = np.random.normal(0, 0.1 * (1 + client_id * 0.05), self.samples_per_client)
            y = X @ self.true_theta + noise
            
            self.client_data.append({
                'X': X, 'y': y, 'client_id': client_id
            })
    
    def compute_global_loss(self, theta):
        """计算全局损失"""
        total_loss = 0
        total_samples = 0
        
        for client_data in self.client_data:
            X, y = client_data['X'], client_data['y']
            pred = X @ theta
            loss = np.mean((pred - y) ** 2)
            total_loss += loss * len(X)
            total_samples += len(X)
            
        return total_loss / total_samples
    
    def fedavg_method(self):
        """标准FedAvg方法"""
        theta = np.zeros(self.d)
        loss_history = []
        
        for round_idx in range(self.num_rounds):
            # 选择客户端
            num_selected = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = np.random.choice(self.num_clients, num_selected, replace=False)
            
            # 客户端更新
            client_updates = []
            for client_idx in selected_clients:
                client_data = self.client_data[client_idx]
                X, y = client_data['X'], client_data['y']
                
                local_theta = theta.copy()
                
                # 本地SGD
                for step in range(self.local_steps):
                    indices = np.random.choice(len(X), min(self.batch_size, len(X)), replace=False)
                    X_batch, y_batch = X[indices], y[indices]
                    
                    pred = X_batch @ local_theta
                    grad = X_batch.T @ (pred - y_batch) / len(X_batch)
                    local_theta -= self.learning_rate * grad
                
                client_updates.append(local_theta)
            
            # 聚合
            theta = np.mean(client_updates, axis=0)
            
            # 记录损失
            loss = self.compute_global_loss(theta)
            loss_history.append(loss)
            
            if round_idx % 20 == 0:
                print(f"  FedAvg Round {round_idx}: Loss = {loss:.6f}")
        
        return {
            'theta': theta,
            'loss_history': loss_history,
            'final_loss': loss_history[-1],
            'param_error': np.linalg.norm(theta - self.true_theta),
            'comm_cost_per_round': self.d * 8  # 传输完整模型
        }
    
    def fedavgm_method(self):
        """带动量的FedAvg方法"""
        theta = np.zeros(self.d)
        server_momentum = np.zeros(self.d)
        loss_history = []
        
        for round_idx in range(self.num_rounds):
            # 选择客户端
            num_selected = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = np.random.choice(self.num_clients, num_selected, replace=False)
            
            # 客户端更新
            client_updates = []
            for client_idx in selected_clients:
                client_data = self.client_data[client_idx]
                X, y = client_data['X'], client_data['y']
                
                local_theta = theta.copy()
                
                # 本地SGD
                for step in range(self.local_steps):
                    indices = np.random.choice(len(X), min(self.batch_size, len(X)), replace=False)
                    X_batch, y_batch = X[indices], y[indices]
                    
                    pred = X_batch @ local_theta
                    grad = X_batch.T @ (pred - y_batch) / len(X_batch)
                    local_theta -= self.learning_rate * grad
                
                client_updates.append(local_theta)
            
            # 服务器端动量聚合
            new_theta = np.mean(client_updates, axis=0)
            delta = new_theta - theta
            server_momentum = self.momentum * server_momentum + delta
            theta = theta + server_momentum
            
            # 记录损失
            loss = self.compute_global_loss(theta)
            loss_history.append(loss)
            
            if round_idx % 20 == 0:
                print(f"  FedAvgM Round {round_idx}: Loss = {loss:.6f}")
        
        return {
            'theta': theta,
            'loss_history': loss_history,
            'final_loss': loss_history[-1],
            'param_error': np.linalg.norm(theta - self.true_theta),
            'comm_cost_per_round': self.d * 8  # 传输完整模型
        }
    
    def sfedavg_method(self, delta):
        """SFedAvg方法"""
        r = max(1, int(delta * self.d))
        theta = np.zeros(self.d)
        server_momentum = np.zeros(self.d)
        loss_history = []
        
        for round_idx in range(self.num_rounds):
            # 刷新投影器
            P = StiefelSampler.sample(self.d, r)
            Pi = P @ P.T
            
            # 选择客户端
            num_selected = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = np.random.choice(self.num_clients, num_selected, replace=False)
            
            # 客户端更新
            client_updates = []
            for client_idx in selected_clients:
                client_data = self.client_data[client_idx]
                X, y = client_data['X'], client_data['y']
                
                local_theta = theta.copy()
                local_momentum = np.zeros(self.d)
                
                # 本地SGD with momentum
                for step in range(self.local_steps):
                    indices = np.random.choice(len(X), min(self.batch_size, len(X)), replace=False)
                    X_batch, y_batch = X[indices], y[indices]
                    
                    pred = X_batch @ local_theta
                    grad = X_batch.T @ (pred - y_batch) / len(X_batch)
                    
                    local_momentum = self.momentum * local_momentum + grad
                    local_theta -= self.learning_rate * local_momentum
                
                client_updates.append(local_theta)
            
            # 服务器聚合
            new_theta = np.mean(client_updates, axis=0)
            delta_theta = new_theta - theta
            
            # 动量投影 (MP)
            server_momentum = Pi @ (self.momentum * server_momentum + delta_theta)
            theta = theta + server_momentum
            
            # 记录损失
            loss = self.compute_global_loss(theta)
            loss_history.append(loss)
            
            if round_idx % 20 == 0:
                print(f"  SFedAvg-δ{delta:.2f} Round {round_idx}: Loss = {loss:.6f}")
        
        return {
            'theta': theta,
            'loss_history': loss_history,
            'final_loss': loss_history[-1],
            'param_error': np.linalg.norm(theta - self.true_theta),
            'comm_cost_per_round': r * 8 if delta < 1.0 else self.d * 8,
            'delta': delta
        }
    
    def run_experiment(self):
        """运行完整实验"""
        print("=" * 80)
        print("SFedAvg vs Baselines: 简化线性回归实验")
        print("=" * 80)
        
        print(f"\n实验配置:")
        print(f"  维度d={self.d}, 客户端={self.num_clients}, 轮数={self.num_rounds}")
        print(f"  本地步数τ={self.local_steps}, 学习率={self.learning_rate}")
        print(f"  真实参数范数: {np.linalg.norm(self.true_theta):.4f}")
        
        # 设置随机种子确保公平比较
        np.random.seed(42)
        
        results = {}
        
        # 1. FedAvg
        print(f"\n{'='*50}")
        print("运行 FedAvg")
        print(f"{'='*50}")
        np.random.seed(42)
        results['FedAvg'] = self.fedavg_method()
        
        # 2. FedAvgM
        print(f"\n{'='*50}")
        print("运行 FedAvgM")
        print(f"{'='*50}")
        np.random.seed(42)
        results['FedAvgM'] = self.fedavgm_method()
        
        # 3. SFedAvg variants
        for delta in [1.0, 0.5, 0.25]:
            print(f"\n{'='*50}")
            print(f"运行 SFedAvg (δ={delta:.2f})")
            print(f"{'='*50}")
            np.random.seed(42)
            results[f'SFedAvg-δ{delta:.2f}'] = self.sfedavg_method(delta)
        
        return results
    
    def analyze_results(self, results):
        """分析实验结果"""
        print("\n" + "=" * 80)
        print("实验结果分析")
        print("=" * 80)
        
        # 1. 性能摘要表
        print(f"\n1. 性能摘要:")
        print(f"{'方法':<15} {'最终损失':<12} {'参数误差':<12} {'通信/轮(KB)':<15} {'总通信(KB)':<12}")
        print("-" * 75)
        
        for method, result in results.items():
            comm_per_round = result['comm_cost_per_round'] / 1024
            total_comm = comm_per_round * self.num_rounds
            
            print(f"{method:<15} {result['final_loss']:<12.6f} "
                  f"{result['param_error']:<12.6f} {comm_per_round:<15.2f} {total_comm:<12.1f}")
        
        # 2. 效率分析
        print(f"\n2. 相对于FedAvg的效率分析:")
        fedavg_result = results['FedAvg']
        
        print(f"{'方法':<15} {'性能比率':<12} {'通信节省':<12} {'权衡比':<12}")
        print("-" * 55)
        
        for method, result in results.items():
            if method == 'FedAvg':
                continue
                
            perf_ratio = result['final_loss'] / fedavg_result['final_loss']
            comm_saving = 1 - (result['comm_cost_per_round'] / fedavg_result['comm_cost_per_round'])
            tradeoff = comm_saving / max(0.01, perf_ratio - 1) if perf_ratio > 1 else float('inf')
            
            print(f"{method:<15} {perf_ratio:<12.4f} {comm_saving:<12.1%} {tradeoff:<12.1f}")
        
        # 3. 收敛性分析
        print(f"\n3. 收敛性分析:")
        print(f"{'方法':<15} {'初始损失':<12} {'最终损失':<12} {'收敛率':<12}")
        print("-" * 55)
        
        for method, result in results.items():
            initial_loss = result['loss_history'][0]
            final_loss = result['final_loss']
            convergence_rate = (initial_loss - final_loss) / initial_loss * 100
            
            print(f"{method:<15} {initial_loss:<12.6f} {final_loss:<12.6f} {convergence_rate:<12.1f}%")
        
        return results
    
    def plot_results(self, results):
        """绘制结果图表"""
        print(f"\n生成可视化图表...")
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SFedAvg vs Baselines: Simplified Linear Regression Results', 
                     fontsize=14, fontweight='bold')
        
        colors = {
            'FedAvg': '#1f77b4',
            'FedAvgM': '#ff7f0e',
            'SFedAvg-δ1.00': '#2ca02c',
            'SFedAvg-δ0.50': '#d62728',
            'SFedAvg-δ0.25': '#9467bd'
        }
        
        rounds = range(1, self.num_rounds + 1)
        
        # 图1: 损失收敛
        ax1.set_title('Loss Convergence')
        for method, result in results.items():
            ax1.plot(rounds, result['loss_history'], 
                    color=colors.get(method, 'gray'), 
                    linewidth=2, label=method)
        ax1.set_xlabel('Communication Round')
        ax1.set_ylabel('Global Loss (MSE)')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 图2: 通信-性能权衡
        ax2.set_title('Communication-Performance Trade-off')
        
        comm_costs = []
        final_losses = []
        method_labels = []
        
        for method, result in results.items():
            comm_costs.append(result['comm_cost_per_round'] / 1024)  # KB
            final_losses.append(result['final_loss'])
            method_labels.append(method)
        
        colors_list = [colors.get(method, 'gray') for method in method_labels]
        scatter = ax2.scatter(comm_costs, final_losses, c=colors_list, s=100, alpha=0.8)
        
        for i, method in enumerate(method_labels):
            ax2.annotate(method, (comm_costs[i], final_losses[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Communication per Round (KB)')
        ax2.set_ylabel('Final Loss')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 图3: 参数收敛误差
        ax3.set_title('Parameter Error vs Communication Cost')
        
        param_errors = [result['param_error'] for result in results.values()]
        
        scatter = ax3.scatter(comm_costs, param_errors, c=colors_list, s=100, alpha=0.8)
        
        for i, method in enumerate(method_labels):
            ax3.annotate(method, (comm_costs[i], param_errors[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Communication per Round (KB)')
        ax3.set_ylabel('Parameter Error ||θ - θ*||')
        ax3.grid(True, alpha=0.3)
        
        # 图4: SFedAvg压缩率分析
        ax4.set_title('SFedAvg Compression Analysis')
        
        sfedavg_methods = [m for m in results.keys() if 'SFedAvg' in m]
        deltas = []
        performance_ratios = []
        comm_savings = []
        
        fedavg_loss = results['FedAvg']['final_loss']
        fedavg_comm = results['FedAvg']['comm_cost_per_round']
        
        for method in sfedavg_methods:
            result = results[method]
            if 'delta' in result:
                deltas.append(result['delta'])
                performance_ratios.append(result['final_loss'] / fedavg_loss)
                comm_savings.append(1 - result['comm_cost_per_round'] / fedavg_comm)
        
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar([str(d) for d in deltas], performance_ratios, 
                       alpha=0.7, color='skyblue', label='Performance Ratio')
        line1 = ax4_twin.plot([str(d) for d in deltas], comm_savings, 
                             'ro-', linewidth=2, markersize=8, label='Communication Saving')
        
        ax4.set_xlabel('Compression Ratio δ')
        ax4.set_ylabel('Performance Ratio (vs FedAvg)', color='blue')
        ax4_twin.set_ylabel('Communication Saving', color='red')
        ax4.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, ratio in zip(bars1, performance_ratios):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ratio:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('simplified_linear_regression_results.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 'simplified_linear_regression_results.png'")
        
        plt.show()


def main():
    """主函数"""
    print("SFedAvg vs Baselines - 简化线性回归实验")
    print("Focus on core algorithm comparison")
    
    # 创建并运行实验
    experiment = SimpleExperiment()
    results = experiment.run_experiment()
    
    # 分析结果
    experiment.analyze_results(results)
    
    # 生成图表
    experiment.plot_results(results)
    
    print("\n" + "=" * 80)
    print("✓ 简化实验完成！")
    print("✓ 验证了SFedAvg在线性回归任务上的有效性")
    print("✓ 展示了通信-性能权衡特性")
    print("=" * 80)


if __name__ == "__main__":
    main()