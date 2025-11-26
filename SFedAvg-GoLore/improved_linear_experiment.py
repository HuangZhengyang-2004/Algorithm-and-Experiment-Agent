"""
Improved Linear Regression Experiment with Hyperparameter Search
改进版线性回归实验，包含超参数搜索
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import product
from sfedavg_implementation import StiefelSampler


class ImprovedExperiment:
    """改进的实验类，包含超参数搜索"""
    
    def __init__(self):
        # 实验配置
        self.num_clients = 10
        self.client_fraction = 0.3
        self.d = 30
        self.samples_per_client = 80
        self.num_rounds = 60  # 稍微减少以加快超参数搜索
        self.local_steps = 5
        self.batch_size = 15
        
        # 超参数搜索空间
        self.lr_candidates = [0.005, 0.01, 0.02, 0.05]
        self.momentum_candidates = [0.0, 0.6, 0.9]
        
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
    
    def fedavg_method(self, learning_rate, momentum=0.0):
        """标准FedAvg方法"""
        theta = np.zeros(self.d)
        server_momentum = np.zeros(self.d) if momentum > 0 else None
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
                    local_theta -= learning_rate * grad
                
                client_updates.append(local_theta)
            
            # 聚合
            new_theta = np.mean(client_updates, axis=0)
            
            # 服务器端动量（如果使用）
            if server_momentum is not None:
                delta = new_theta - theta
                server_momentum = momentum * server_momentum + delta
                theta = theta + server_momentum
            else:
                theta = new_theta
            
            # 记录损失
            loss = self.compute_global_loss(theta)
            loss_history.append(loss)
        
        return {
            'theta': theta,
            'loss_history': loss_history,
            'final_loss': loss_history[-1],
            'param_error': np.linalg.norm(theta - self.true_theta),
            'comm_cost_per_round': self.d * 8
        }
    
    def sfedavg_method(self, delta, learning_rate, momentum):
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
                    
                    local_momentum = momentum * local_momentum + grad
                    local_theta -= learning_rate * local_momentum
                
                client_updates.append(local_theta)
            
            # 服务器聚合
            new_theta = np.mean(client_updates, axis=0)
            delta_theta = new_theta - theta
            
            # 动量投影 (MP)
            server_momentum = Pi @ (momentum * server_momentum + delta_theta)
            theta = theta + server_momentum
            
            # 记录损失
            loss = self.compute_global_loss(theta)
            loss_history.append(loss)
        
        return {
            'theta': theta,
            'loss_history': loss_history,
            'final_loss': loss_history[-1],
            'param_error': np.linalg.norm(theta - self.true_theta),
            'comm_cost_per_round': r * 8 if delta < 1.0 else self.d * 8,
            'delta': delta
        }
    
    def hyperparameter_search(self, method_name, method_func, **kwargs):
        """超参数搜索"""
        print(f"\n🔍 对 {method_name} 进行超参数搜索...")
        
        best_loss = float('inf')
        best_params = None
        best_result = None
        
        search_count = 0
        total_searches = len(self.lr_candidates) * len(self.momentum_candidates)
        
        for lr, mom in product(self.lr_candidates, self.momentum_candidates):
            search_count += 1
            
            # 设置随机种子确保一致性
            np.random.seed(42)
            
            try:
                if 'delta' in kwargs:
                    result = method_func(kwargs['delta'], lr, mom)
                else:
                    result = method_func(lr, mom)
                
                if result['final_loss'] < best_loss:
                    best_loss = result['final_loss']
                    best_params = {'learning_rate': lr, 'momentum': mom}
                    best_result = result
                    best_result['best_params'] = best_params
                
                print(f"  [{search_count}/{total_searches}] lr={lr:.3f}, mom={mom:.1f} → loss={result['final_loss']:.6f}")
                
            except Exception as e:
                print(f"  [{search_count}/{total_searches}] lr={lr:.3f}, mom={mom:.1f} → ERROR: {e}")
                continue
        
        print(f"  ✅ 最佳参数: lr={best_params['learning_rate']:.3f}, mom={best_params['momentum']:.1f}")
        print(f"  ✅ 最佳损失: {best_loss:.6f}")
        
        return best_result, best_params
    
    def run_experiment_with_hyperparameter_search(self):
        """运行包含超参数搜索的完整实验"""
        print("=" * 80)
        print("SFedAvg vs Baselines: 包含超参数搜索的实验")
        print("=" * 80)
        
        print(f"\n实验配置:")
        print(f"  维度d={self.d}, 客户端={self.num_clients}, 轮数={self.num_rounds}")
        print(f"  本地步数τ={self.local_steps}, 批次大小={self.batch_size}")
        print(f"  学习率候选: {self.lr_candidates}")
        print(f"  动量候选: {self.momentum_candidates}")
        print(f"  真实参数范数: {np.linalg.norm(self.true_theta):.4f}")
        
        results = {}
        best_params = {}
        
        # 1. FedAvg超参数搜索
        print(f"\n{'='*60}")
        print("1. FedAvg 超参数搜索")
        print(f"{'='*60}")
        
        fedavg_result, fedavg_params = self.hyperparameter_search(
            "FedAvg", self.fedavg_method
        )
        results['FedAvg'] = fedavg_result
        best_params['FedAvg'] = fedavg_params
        
        # 2. FedAvgM超参数搜索 (排除momentum=0的情况)
        print(f"\n{'='*60}")
        print("2. FedAvgM 超参数搜索")
        print(f"{'='*60}")
        
        # 临时修改momentum候选，排除0
        original_momentum = self.momentum_candidates
        self.momentum_candidates = [m for m in self.momentum_candidates if m > 0]
        
        fedavgm_result, fedavgm_params = self.hyperparameter_search(
            "FedAvgM", self.fedavg_method
        )
        results['FedAvgM'] = fedavgm_result
        best_params['FedAvgM'] = fedavgm_params
        
        # 恢复原始momentum候选
        self.momentum_candidates = original_momentum
        
        # 3. SFedAvg不同δ值的超参数搜索
        for delta in [1.0, 0.5, 0.25]:
            method_name = f'SFedAvg-δ{delta:.2f}'
            print(f"\n{'='*60}")
            print(f"3.{int(delta*4)}. {method_name} 超参数搜索")
            print(f"{'='*60}")
            
            sfedavg_result, sfedavg_params = self.hyperparameter_search(
                method_name, self.sfedavg_method, delta=delta
            )
            results[method_name] = sfedavg_result
            best_params[method_name] = sfedavg_params
        
        return results, best_params
    
    def analyze_results_with_hyperparams(self, results, best_params):
        """分析包含超参数的实验结果"""
        print("\n" + "=" * 100)
        print("实验结果分析 (包含最优超参数)")
        print("=" * 100)
        
        # 1. 最优超参数表
        print(f"\n1. 各方法的最优超参数:")
        print(f"{'方法':<15} {'学习率':<10} {'动量':<8} {'最终损失':<12} {'参数误差':<12}")
        print("-" * 65)
        
        for method, result in results.items():
            params = best_params[method]
            print(f"{method:<15} {params['learning_rate']:<10.3f} {params['momentum']:<8.1f} "
                  f"{result['final_loss']:<12.6f} {result['param_error']:<12.6f}")
        
        # 2. 性能摘要表
        print(f"\n2. 性能摘要 (使用最优超参数):")
        print(f"{'方法':<15} {'最终损失':<12} {'参数误差':<12} {'通信/轮(KB)':<15} {'总通信(KB)':<12}")
        print("-" * 75)
        
        for method, result in results.items():
            comm_per_round = result['comm_cost_per_round'] / 1024
            total_comm = comm_per_round * self.num_rounds
            
            print(f"{method:<15} {result['final_loss']:<12.6f} "
                  f"{result['param_error']:<12.6f} {comm_per_round:<15.2f} {total_comm:<12.1f}")
        
        # 3. 效率分析
        print(f"\n3. 相对于FedAvg的效率分析:")
        fedavg_result = results['FedAvg']
        
        print(f"{'方法':<15} {'性能比率':<12} {'通信节省':<12} {'权衡效率':<12}")
        print("-" * 55)
        
        for method, result in results.items():
            if method == 'FedAvg':
                continue
                
            perf_ratio = result['final_loss'] / fedavg_result['final_loss']
            comm_saving = 1 - (result['comm_cost_per_round'] / fedavg_result['comm_cost_per_round'])
            tradeoff = comm_saving / max(0.01, abs(perf_ratio - 1)) if perf_ratio != 1.0 else float('inf')
            
            print(f"{method:<15} {perf_ratio:<12.4f} {comm_saving:<12.1%} {tradeoff:<12.1f}")
        
        # 4. 超参数敏感性分析
        print(f"\n4. 超参数敏感性分析:")
        lr_usage = {}
        mom_usage = {}
        
        for method, params in best_params.items():
            lr = params['learning_rate']
            mom = params['momentum']
            
            lr_usage[lr] = lr_usage.get(lr, 0) + 1
            mom_usage[mom] = mom_usage.get(mom, 0) + 1
        
        print(f"  最常用学习率:")
        for lr, count in sorted(lr_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"    lr={lr:.3f}: {count} 个方法")
        
        print(f"  最常用动量:")
        for mom, count in sorted(mom_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"    momentum={mom:.1f}: {count} 个方法")
        
        return results
    
    def plot_results_with_hyperparams(self, results, best_params):
        """绘制包含超参数信息的结果图表"""
        print(f"\n生成可视化图表...")
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SFedAvg vs Baselines: Results with Hyperparameter Search', 
                     fontsize=16, fontweight='bold')
        
        colors = {
            'FedAvg': '#1f77b4',
            'FedAvgM': '#ff7f0e',
            'SFedAvg-δ1.00': '#2ca02c',
            'SFedAvg-δ0.50': '#d62728',
            'SFedAvg-δ0.25': '#9467bd'
        }
        
        rounds = range(1, self.num_rounds + 1)
        
        # 图1: 损失收敛 (使用最优超参数)
        ax1.set_title('Loss Convergence (Optimal Hyperparameters)')
        for method, result in results.items():
            params = best_params[method]
            label = f"{method} (lr={params['learning_rate']:.3f}, μ={params['momentum']:.1f})"
            ax1.plot(rounds, result['loss_history'], 
                    color=colors.get(method, 'gray'), 
                    linewidth=2, label=label)
        ax1.set_xlabel('Communication Round')
        ax1.set_ylabel('Global Loss (MSE)')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
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
        scatter = ax2.scatter(comm_costs, final_losses, c=colors_list, s=120, alpha=0.8)
        
        for i, method in enumerate(method_labels):
            ax2.annotate(method, (comm_costs[i], final_losses[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax2.set_xlabel('Communication per Round (KB)')
        ax2.set_ylabel('Final Loss (MSE)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 图3: 超参数分布
        ax3.set_title('Optimal Hyperparameter Distribution')
        
        lrs = [best_params[method]['learning_rate'] for method in results.keys()]
        moms = [best_params[method]['momentum'] for method in results.keys()]
        method_names = list(results.keys())
        
        colors_hp = [colors.get(method, 'gray') for method in method_names]
        scatter = ax3.scatter(lrs, moms, c=colors_hp, s=150, alpha=0.8)
        
        for i, method in enumerate(method_names):
            ax3.annotate(method, (lrs[i], moms[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Optimal Learning Rate')
        ax3.set_ylabel('Optimal Momentum')
        ax3.grid(True, alpha=0.3)
        
        # 图4: 性能改进vs通信节省
        ax4.set_title('Performance Improvement vs Communication Saving')
        
        fedavg_loss = results['FedAvg']['final_loss']
        fedavg_comm = results['FedAvg']['comm_cost_per_round']
        
        improvements = []
        comm_savings = []
        sfedavg_methods = []
        
        for method, result in results.items():
            if 'SFedAvg' in method:
                improvement = (fedavg_loss - result['final_loss']) / fedavg_loss * 100
                comm_saving = (1 - result['comm_cost_per_round'] / fedavg_comm) * 100
                improvements.append(improvement)
                comm_savings.append(comm_saving)
                sfedavg_methods.append(method)
        
        colors_sfed = [colors.get(method, 'gray') for method in sfedavg_methods]
        bars = ax4.scatter(comm_savings, improvements, c=colors_sfed, s=150, alpha=0.8)
        
        for i, method in enumerate(sfedavg_methods):
            ax4.annotate(method, (comm_savings[i], improvements[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # 添加参考线
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No improvement')
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='No communication saving')
        
        ax4.set_xlabel('Communication Saving (%)')
        ax4.set_ylabel('Performance Improvement (%)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('improved_linear_regression_results.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 'improved_linear_regression_results.png'")
        
        plt.show()


def main():
    """主函数"""
    print("SFedAvg vs Baselines - 改进版线性回归实验")
    print("包含超参数搜索的公平对比")
    
    start_time = time.time()
    
    # 创建并运行实验
    experiment = ImprovedExperiment()
    results, best_params = experiment.run_experiment_with_hyperparameter_search()
    
    # 分析结果
    experiment.analyze_results_with_hyperparams(results, best_params)
    
    # 生成图表
    experiment.plot_results_with_hyperparams(results, best_params)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✓ 改进实验完成！")
    print(f"✓ 总耗时: {total_time:.1f}秒")
    print("✓ 通过超参数搜索确保了公平对比")
    print("✓ 发现了各算法的最优配置")
    print("=" * 80)


if __name__ == "__main__":
    main()