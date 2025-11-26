"""
SFedAvg vs Baselines Experiment on Linear Regression Task
基于实验方案的完整对比实验实现
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sfedavg_implementation import StiefelSampler
import copy


@dataclass
class ExperimentConfig:
    """实验配置类"""
    num_clients: int = 10
    client_fraction: float = 0.3  # C=0.3, 每轮约3个客户端参与
    d: int = 50  # 特征维度
    samples_per_client: int = 100
    num_rounds: int = 100
    local_steps_options: List[int] = None  # [1, 5]
    batch_size: int = 10
    learning_rate: float = 0.01
    momentum: float = 0.9
    delta_options: List[float] = None  # [1.0, 0.5, 0.25] for SFedAvg
    noise_std: float = 0.1
    heterogeneity: float = 0.3  # 非IID程度
    
    def __post_init__(self):
        if self.local_steps_options is None:
            self.local_steps_options = [1, 5]
        if self.delta_options is None:
            self.delta_options = [1.0, 0.5, 0.25]


class LinearRegressionData:
    """线性回归数据生成器"""
    
    @staticmethod
    def generate_federated_data(config: ExperimentConfig) -> Tuple[List, np.ndarray]:
        """生成联邦学习的线性回归数据"""
        
        # 真实参数
        true_theta = np.random.randn(config.d)
        true_theta = true_theta / np.linalg.norm(true_theta) * 3.0  # 归一化
        
        client_data = []
        
        for client_id in range(config.num_clients):
            # 为每个客户端生成不同的数据分布（非IID）
            np.random.seed(42 + client_id)  # 确保可重复性
            
            # 生成特征矩阵，每个客户端有略微不同的协方差结构
            base_cov = np.eye(config.d)
            client_cov = base_cov + config.heterogeneity * np.random.randn(config.d, config.d) * 0.1
            client_cov = (client_cov + client_cov.T) / 2  # 保证对称
            client_cov = client_cov + np.eye(config.d) * 0.1  # 保证正定
            
            # 生成样本
            X = np.random.multivariate_normal(
                mean=np.zeros(config.d), 
                cov=client_cov, 
                size=config.samples_per_client
            )
            
            # 生成标签，每个客户端有不同的噪声水平
            client_noise = config.noise_std * (1 + client_id * 0.1)  # 不同噪声水平
            y = X @ true_theta + np.random.normal(0, client_noise, config.samples_per_client)
            
            client_data.append({'X': X, 'y': y, 'client_id': client_id})
        
        return client_data, true_theta


class FedAvgBaseline:
    """标准FedAvg基线实现"""
    
    def __init__(self, d: int, learning_rate: float = 0.01, use_momentum: bool = False, momentum: float = 0.9):
        self.d = d
        self.learning_rate = learning_rate
        self.use_momentum = use_momentum
        self.momentum = momentum
        
        # 全局模型
        self.theta = np.zeros(d)
        
        # 服务器端动量（FedAvgM）
        if use_momentum:
            self.server_momentum = np.zeros(d)
        
        # 客户端状态
        self.client_states = {}
        
    def get_memory_usage(self) -> int:
        """计算内存使用量（字节）"""
        memory = self.theta.nbytes  # 模型参数
        
        if hasattr(self, 'server_momentum'):
            memory += self.server_momentum.nbytes  # 服务器动量
            
        # 客户端状态内存
        for state in self.client_states.values():
            if 'momentum' in state and state['momentum'] is not None:
                memory += state['momentum'].nbytes
                
        return memory
        
    def client_update(self, client_data: Dict, local_steps: int, batch_size: int) -> Dict:
        """客户端本地更新"""
        client_id = client_data['client_id']
        X, y = client_data['X'], client_data['y']
        n_samples = len(X)
        
        # 初始化客户端状态
        if client_id not in self.client_states:
            self.client_states[client_id] = {
                'momentum': np.zeros(self.d) if self.use_momentum else None
            }
            
        local_theta = self.theta.copy()
        local_momentum = self.client_states[client_id]['momentum']
        
        for step in range(local_steps):
            # 随机采样批次
            indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
            X_batch, y_batch = X[indices], y[indices]
            
            # 计算梯度
            pred = X_batch @ local_theta
            grad = X_batch.T @ (pred - y_batch) / len(X_batch)
            
            # 更新参数
            if self.use_momentum and local_momentum is not None:
                local_momentum = self.momentum * local_momentum + grad
                local_theta -= self.learning_rate * local_momentum
            else:
                local_theta -= self.learning_rate * grad
        
        # 更新客户端状态
        if self.use_momentum:
            self.client_states[client_id]['momentum'] = local_momentum.copy()
            
        return {
            'theta': local_theta,
            'client_id': client_id
        }
    
    def aggregate(self, client_updates: List[Dict]):
        """服务器聚合"""
        if not client_updates:
            return
            
        # 简单平均聚合
        new_theta = np.zeros(self.d)
        for update in client_updates:
            new_theta += update['theta']
        new_theta /= len(client_updates)
        
        # 服务器端动量更新（FedAvgM）
        if hasattr(self, 'server_momentum'):
            delta = new_theta - self.theta
            self.server_momentum = self.momentum * self.server_momentum + delta
            self.theta = self.theta + self.server_momentum
        else:
            self.theta = new_theta
            
    def get_communication_cost(self) -> int:
        """计算通信成本（字节）"""
        return self.theta.nbytes  # 每轮传输完整模型


class LocalSGD:
    """Local-SGD基线实现"""
    
    def __init__(self, d: int, learning_rate: float = 0.01, momentum: float = 0.9):
        self.d = d
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # 全局模型
        self.theta = np.zeros(d)
        
        # 客户端状态
        self.client_states = {}
        
    def get_memory_usage(self) -> int:
        """计算内存使用量（字节）"""
        memory = self.theta.nbytes  # 模型参数
        
        # 客户端状态内存
        for state in self.client_states.values():
            if 'theta' in state:
                memory += state['theta'].nbytes
            if 'momentum' in state:
                memory += state['momentum'].nbytes
                
        return memory
        
    def client_update(self, client_data: Dict, local_steps: int, batch_size: int) -> Dict:
        """客户端本地更新"""
        client_id = client_data['client_id']
        X, y = client_data['X'], client_data['y']
        n_samples = len(X)
        
        # 初始化客户端状态
        if client_id not in self.client_states:
            self.client_states[client_id] = {
                'theta': self.theta.copy(),
                'momentum': np.zeros(self.d)
            }
            
        local_theta = self.client_states[client_id]['theta']
        local_momentum = self.client_states[client_id]['momentum']
        
        for step in range(local_steps):
            # 随机采样批次
            indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
            X_batch, y_batch = X[indices], y[indices]
            
            # 计算梯度
            pred = X_batch @ local_theta
            grad = X_batch.T @ (pred - y_batch) / len(X_batch)
            
            # 动量更新
            local_momentum = self.momentum * local_momentum + grad
            local_theta -= self.learning_rate * local_momentum
        
        # 更新客户端状态
        self.client_states[client_id]['theta'] = local_theta.copy()
        self.client_states[client_id]['momentum'] = local_momentum.copy()
            
        return {
            'theta': local_theta,
            'client_id': client_id
        }
    
    def aggregate(self, client_updates: List[Dict]):
        """服务器聚合"""
        if not client_updates:
            return
            
        # 简单平均聚合
        new_theta = np.zeros(self.d)
        for update in client_updates:
            new_theta += update['theta']
        new_theta /= len(client_updates)
        
        self.theta = new_theta
        
        # 更新所有客户端的全局模型副本
        for client_id in self.client_states:
            self.client_states[client_id]['theta'] = self.theta.copy()
            
    def get_communication_cost(self) -> int:
        """计算通信成本（字节）"""
        return self.theta.nbytes  # 每轮传输完整模型


class SFedAvgMethod:
    """SFedAvg方法实现"""
    
    def __init__(self, d: int, delta: float, learning_rate: float = 0.01, momentum: float = 0.9):
        self.d = d
        self.r = max(1, int(delta * d))  # 子空间维度
        self.delta = delta
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # 全局模型
        self.theta = np.zeros(d)
        
        # 当前子空间投影器
        self.P = None  # 当前投影矩阵
        self.Pi = None  # 当前投影器
        
        # 服务器动量
        self.server_momentum = np.zeros(d)
        
        # 客户端状态
        self.client_states = {}
        
        # 刷新投影器
        self._refresh_projector()
        
    def _refresh_projector(self):
        """刷新Stiefel投影器"""
        self.P = StiefelSampler.sample(self.d, self.r)
        self.Pi = self.P @ self.P.T
        
    def get_memory_usage(self) -> int:
        """计算内存使用量（字节）"""
        memory = self.theta.nbytes  # 模型参数
        memory += self.server_momentum.nbytes  # 服务器动量
        memory += self.P.nbytes  # 投影矩阵
        
        # 客户端状态内存（压缩后的动量）
        for state in self.client_states.values():
            if 'momentum_compressed' in state and state['momentum_compressed'] is not None:
                memory += state['momentum_compressed'].nbytes  # r维而非d维
                
        return memory
        
    def client_update(self, client_data: Dict, local_steps: int, batch_size: int) -> Dict:
        """客户端本地更新"""
        client_id = client_data['client_id']
        X, y = client_data['X'], client_data['y']
        n_samples = len(X)
        
        # 初始化客户端状态
        if client_id not in self.client_states:
            self.client_states[client_id] = {
                'momentum_compressed': np.zeros(self.r)  # 压缩动量
            }
            
        local_theta = self.theta.copy()
        
        # 恢复客户端动量（从压缩空间）
        momentum_compressed = self.client_states[client_id]['momentum_compressed']
        local_momentum = self.P @ momentum_compressed  # 解压缩
        
        for step in range(local_steps):
            # 随机采样批次
            indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
            X_batch, y_batch = X[indices], y[indices]
            
            # 计算梯度
            pred = X_batch @ local_theta
            grad = X_batch.T @ (pred - y_batch) / len(X_batch)
            
            # 动量更新
            local_momentum = self.momentum * local_momentum + grad
            local_theta -= self.learning_rate * local_momentum
        
        # 压缩并存储客户端动量
        momentum_compressed = self.P.T @ local_momentum
        self.client_states[client_id]['momentum_compressed'] = momentum_compressed.copy()
            
        return {
            'theta': local_theta,
            'client_id': client_id
        }
    
    def aggregate(self, client_updates: List[Dict]):
        """服务器聚合"""
        if not client_updates:
            return
            
        # 简单平均聚合
        new_theta = np.zeros(self.d)
        for update in client_updates:
            new_theta += update['theta']
        new_theta /= len(client_updates)
        
        # 动量投影（MP）
        delta_theta = new_theta - self.theta
        self.server_momentum = self.Pi @ (self.momentum * self.server_momentum + delta_theta)
        
        # 更新全局模型
        self.theta = self.theta + self.server_momentum
        
        # 每轮刷新投影器
        self._refresh_projector()
        
    def get_communication_cost(self) -> int:
        """计算通信成本（字节）"""
        if self.delta == 1.0:
            return self.theta.nbytes  # 完整模型
        else:
            return self.r * 8  # 只传输r维压缩表示（假设float64）


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client_data, self.true_theta = LinearRegressionData.generate_federated_data(config)
        
    def compute_mse(self, theta: np.ndarray) -> float:
        """计算全局MSE"""
        total_mse = 0
        total_samples = 0
        
        for client_data in self.client_data:
            X, y = client_data['X'], client_data['y']
            pred = X @ theta
            mse = np.mean((pred - y) ** 2)
            total_mse += mse * len(X)
            total_samples += len(X)
            
        return total_mse / total_samples
    
    def compute_parameter_error(self, theta: np.ndarray) -> float:
        """计算参数误差"""
        return np.linalg.norm(theta - self.true_theta)
    
    def run_single_experiment(self, method_name: str, method, local_steps: int) -> Dict:
        """运行单个实验"""
        print(f"\n运行实验: {method_name} (τ={local_steps})")
        
        # 跟踪指标
        history = {
            'mse': [],
            'param_error': [],
            'communication_cost': [],
            'memory_usage': [],
            'wall_time': []
        }
        
        start_time = time.time()
        total_comm_cost = 0
        
        for round_idx in range(self.config.num_rounds):
            round_start = time.time()
            
            # 选择参与的客户端
            num_selected = max(1, int(self.config.client_fraction * self.config.num_clients))
            selected_clients = np.random.choice(
                self.config.num_clients, num_selected, replace=False
            )
            
            # 客户端更新
            client_updates = []
            for client_idx in selected_clients:
                update = method.client_update(
                    self.client_data[client_idx], 
                    local_steps, 
                    self.config.batch_size
                )
                client_updates.append(update)
            
            # 服务器聚合
            method.aggregate(client_updates)
            
            # 记录指标
            current_mse = self.compute_mse(method.theta)
            current_param_error = self.compute_parameter_error(method.theta)
            current_memory = method.get_memory_usage()
            current_comm_cost = method.get_communication_cost()
            total_comm_cost += current_comm_cost
            
            history['mse'].append(current_mse)
            history['param_error'].append(current_param_error)
            history['communication_cost'].append(total_comm_cost)
            history['memory_usage'].append(current_memory)
            history['wall_time'].append(time.time() - start_time)
            
            if round_idx % 20 == 0:
                print(f"  Round {round_idx}: MSE={current_mse:.6f}, "
                      f"Param Error={current_param_error:.6f}, "
                      f"Comm Cost={total_comm_cost:,} bytes")
        
        final_results = {
            'method': method_name,
            'local_steps': local_steps,
            'final_mse': history['mse'][-1],
            'final_param_error': history['param_error'][-1],
            'total_communication_cost': total_comm_cost,
            'peak_memory_usage': max(history['memory_usage']),
            'total_time': time.time() - start_time,
            'history': history
        }
        
        if hasattr(method, 'delta'):
            final_results['delta'] = method.delta
            
        return final_results
    
    def run_all_experiments(self) -> Dict:
        """运行所有实验"""
        print("="*80)
        print("SFedAvg vs Baselines Linear Regression Experiment")
        print("="*80)
        
        all_results = {}
        
        # 设置随机种子
        np.random.seed(42)
        
        for local_steps in self.config.local_steps_options:
            print(f"\n{'='*60}")
            print(f"本地步数 τ = {local_steps}")
            print(f"{'='*60}")
            
            results_for_tau = []
            
            # 1. FedAvg
            method = FedAvgBaseline(
                d=self.config.d, 
                learning_rate=self.config.learning_rate, 
                use_momentum=False
            )
            result = self.run_single_experiment(f"FedAvg", method, local_steps)
            results_for_tau.append(result)
            
            # 2. FedAvgM  
            method = FedAvgBaseline(
                d=self.config.d, 
                learning_rate=self.config.learning_rate, 
                use_momentum=True, 
                momentum=self.config.momentum
            )
            result = self.run_single_experiment(f"FedAvgM", method, local_steps)
            results_for_tau.append(result)
            
            # 3. Local-SGD
            method = LocalSGD(
                d=self.config.d, 
                learning_rate=self.config.learning_rate, 
                momentum=self.config.momentum
            )
            result = self.run_single_experiment(f"Local-SGD", method, local_steps)
            results_for_tau.append(result)
            
            # 4. SFedAvg with different δ
            for delta in self.config.delta_options:
                method = SFedAvgMethod(
                    d=self.config.d, 
                    delta=delta,
                    learning_rate=self.config.learning_rate, 
                    momentum=self.config.momentum
                )
                method_name = f"SFedAvg-δ{delta:.2f}"
                result = self.run_single_experiment(method_name, method, local_steps)
                results_for_tau.append(result)
            
            all_results[f'tau_{local_steps}'] = results_for_tau
            
        return all_results


class ResultsAnalyzer:
    """结果分析器"""
    
    def __init__(self, results: Dict, config: ExperimentConfig):
        self.results = results
        self.config = config
        
    def print_summary_table(self):
        """打印结果摘要表"""
        print("\n" + "="*120)
        print("实验结果摘要表")
        print("="*120)
        
        header = f"{'方法':<15} {'τ':<3} {'δ':<6} {'最终MSE':<12} {'参数误差':<12} {'通信成本(KB)':<15} {'内存(KB)':<12} {'时间(s)':<10}"
        print(header)
        print("-" * 120)
        
        for tau_key, results_list in self.results.items():
            tau_val = int(tau_key.split('_')[1])
            
            for result in results_list:
                delta_str = f"{result.get('delta', 1.0):.2f}" if 'delta' in result else "N/A"
                
                print(f"{result['method']:<15} "
                      f"{tau_val:<3} "
                      f"{delta_str:<6} "
                      f"{result['final_mse']:<12.6f} "
                      f"{result['final_param_error']:<12.6f} "
                      f"{result['total_communication_cost']/1024:<15.1f} "
                      f"{result['peak_memory_usage']/1024:<12.1f} "
                      f"{result['total_time']:<10.2f}")
        
    def analyze_efficiency(self):
        """分析通信和内存效率"""
        print("\n" + "="*80)
        print("效率分析")
        print("="*80)
        
        for tau_key, results_list in self.results.items():
            tau_val = int(tau_key.split('_')[1])
            print(f"\n本地步数 τ = {tau_val}:")
            
            # 找到FedAvg作为基线
            fedavg_result = None
            for result in results_list:
                if result['method'] == 'FedAvg':
                    fedavg_result = result
                    break
                    
            if fedavg_result is None:
                continue
                
            print(f"{'方法':<15} {'通信压缩率':<12} {'内存压缩率':<12} {'性能保持率':<12}")
            print("-" * 60)
            
            for result in results_list:
                if result['method'] == 'FedAvg':
                    continue
                    
                comm_ratio = result['total_communication_cost'] / fedavg_result['total_communication_cost']
                memory_ratio = result['peak_memory_usage'] / fedavg_result['peak_memory_usage']
                perf_ratio = fedavg_result['final_mse'] / result['final_mse']  # 越大越好
                
                print(f"{result['method']:<15} "
                      f"{1-comm_ratio:<12.1%} "
                      f"{1-memory_ratio:<12.1%} "
                      f"{perf_ratio:<12.2f}x")
    
    def generate_plots(self):
        """生成可视化图表"""
        print("\n生成可视化图表...")
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SFedAvg vs Baselines: Linear Regression Experiment Results', 
                     fontsize=16, fontweight='bold')
        
        colors = {
            'FedAvg': '#1f77b4',
            'FedAvgM': '#ff7f0e', 
            'Local-SGD': '#2ca02c',
            'SFedAvg-δ1.00': '#d62728',
            'SFedAvg-δ0.50': '#9467bd',
            'SFedAvg-δ0.25': '#8c564b'
        }
        
        # 只分析τ=5的结果（更有代表性）
        results_tau5 = self.results.get('tau_5', [])
        
        # 图1: MSE收敛曲线
        ax1.set_title('MSE Convergence', fontweight='bold')
        for result in results_tau5:
            method = result['method']
            rounds = range(1, len(result['history']['mse']) + 1)
            ax1.plot(rounds, result['history']['mse'], 
                    color=colors.get(method, 'gray'), 
                    linewidth=2, label=method, alpha=0.8)
        
        ax1.set_xlabel('Communication Round')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # 图2: 参数误差收敛
        ax2.set_title('Parameter Error Convergence', fontweight='bold')
        for result in results_tau5:
            method = result['method']
            rounds = range(1, len(result['history']['param_error']) + 1)
            ax2.plot(rounds, result['history']['param_error'], 
                    color=colors.get(method, 'gray'), 
                    linewidth=2, label=method, alpha=0.8)
        
        ax2.set_xlabel('Communication Round')
        ax2.set_ylabel('||θ - θ*||')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # 图3: 通信-性能权衡
        ax3.set_title('Communication vs Performance Trade-off', fontweight='bold')
        
        comm_costs = []
        mse_values = []
        method_names = []
        method_colors = []
        
        for result in results_tau5:
            comm_costs.append(result['total_communication_cost'] / 1024)  # KB
            mse_values.append(result['final_mse'])
            method_names.append(result['method'])
            method_colors.append(colors.get(result['method'], 'gray'))
        
        scatter = ax3.scatter(comm_costs, mse_values, c=method_colors, s=100, alpha=0.7)
        
        for i, method in enumerate(method_names):
            ax3.annotate(method, (comm_costs[i], mse_values[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Total Communication Cost (KB)')
        ax3.set_ylabel('Final MSE')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 图4: 内存-性能权衡
        ax4.set_title('Memory vs Performance Trade-off', fontweight='bold')
        
        memory_usage = []
        for result in results_tau5:
            memory_usage.append(result['peak_memory_usage'] / 1024)  # KB
        
        scatter = ax4.scatter(memory_usage, mse_values, c=method_colors, s=100, alpha=0.7)
        
        for i, method in enumerate(method_names):
            ax4.annotate(method, (memory_usage[i], mse_values[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('Peak Memory Usage (KB)')
        ax4.set_ylabel('Final MSE')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('linear_regression_experiment_results.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 'linear_regression_experiment_results.png'")
        
        # 显示图表
        plt.show()
    
    def analyze_ablation_studies(self):
        """消融实验分析"""
        print("\n" + "="*80)
        print("消融实验分析")
        print("="*80)
        
        # 1. 子空间秩δ的影响
        print("\n1. 子空间秩δ对SFedAvg的影响:")
        print(f"{'δ值':<8} {'最终MSE':<12} {'通信压缩':<12} {'性能损失':<12}")
        print("-" * 50)
        
        sfedavg_results = []
        for tau_key, results_list in self.results.items():
            for result in results_list:
                if 'SFedAvg' in result['method'] and 'delta' in result:
                    sfedavg_results.append(result)
        
        # 按δ值排序
        sfedavg_results.sort(key=lambda x: x['delta'], reverse=True)
        
        baseline_mse = None
        baseline_comm = None
        
        for result in sfedavg_results:
            if result['delta'] == 1.0:  # δ=1.0作为基线
                baseline_mse = result['final_mse']
                baseline_comm = result['total_communication_cost']
                break
        
        for result in sfedavg_results:
            delta = result['delta']
            mse = result['final_mse']
            comm_compression = 1 - (result['total_communication_cost'] / baseline_comm) if baseline_comm else 0
            perf_loss = (mse / baseline_mse - 1) * 100 if baseline_mse else 0
            
            print(f"{delta:<8.2f} {mse:<12.6f} {comm_compression:<12.1%} {perf_loss:<12.1f}%")
        
        # 2. 本地步数τ的影响
        print(f"\n2. 本地步数τ的影响:")
        print(f"{'方法':<15} {'τ=1 MSE':<12} {'τ=5 MSE':<12} {'变化率':<12}")
        print("-" * 55)
        
        method_results = {}
        for tau_key, results_list in self.results.items():
            tau_val = int(tau_key.split('_')[1])
            for result in results_list:
                method = result['method']
                if method not in method_results:
                    method_results[method] = {}
                method_results[method][tau_val] = result['final_mse']
        
        for method, tau_results in method_results.items():
            if 1 in tau_results and 5 in tau_results:
                mse_1 = tau_results[1]
                mse_5 = tau_results[5]
                change_rate = (mse_5 / mse_1 - 1) * 100
                
                print(f"{method:<15} {mse_1:<12.6f} {mse_5:<12.6f} {change_rate:<12.1f}%")


def main():
    """主函数"""
    print("SFedAvg vs Baselines Linear Regression Experiment")
    print("基于详细实验方案的完整对比实验")
    
    # 实验配置
    config = ExperimentConfig(
        num_clients=10,
        client_fraction=0.3,
        d=50,
        samples_per_client=100,
        num_rounds=100,
        local_steps_options=[1, 5],
        batch_size=10,
        learning_rate=0.01,
        momentum=0.9,
        delta_options=[1.0, 0.5, 0.25],
        noise_std=0.1,
        heterogeneity=0.3
    )
    
    print(f"\n实验配置:")
    print(f"  客户端数量: {config.num_clients}")
    print(f"  客户端参与率: {config.client_fraction}")
    print(f"  特征维度: {config.d}")
    print(f"  每客户端样本数: {config.samples_per_client}")
    print(f"  通信轮数: {config.num_rounds}")
    print(f"  本地步数: {config.local_steps_options}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  动量: {config.momentum}")
    print(f"  SFedAvg δ值: {config.delta_options}")
    
    # 运行实验
    runner = ExperimentRunner(config)
    results = runner.run_all_experiments()
    
    # 分析结果
    analyzer = ResultsAnalyzer(results, config)
    analyzer.print_summary_table()
    analyzer.analyze_efficiency()
    analyzer.analyze_ablation_studies()
    analyzer.generate_plots()
    
    print("\n" + "="*80)
    print("✓ 实验完成！所有结果已生成。")
    print("✓ 图表已保存为 'linear_regression_experiment_results.png'")
    print("="*80)


if __name__ == "__main__":
    main()