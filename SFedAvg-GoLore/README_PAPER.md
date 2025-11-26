# SFedAvg-GoLore 论文项目

## 文件结构

### 核心文件
- `SFedAvg-GoLore.tex` - 主要的LaTeX论文文档
- `references.bib` - 参考文献数据库
- `sfedavg_implementation.py` - SFedAvg算法核心实现
- `improved_linear_experiment.py` - 包含超参数搜索的实验
- `simplified_linear_experiment.py` - 简化版实验

### 实验结果
- `improved_linear_regression_results.png` - 主要实验结果图表
- `simplified_linear_regression_results.png` - 简化实验图表
- 各种分析报告 (*.md 文件)

## 论文章节结构

1. **Abstract** - 算法概述和主要贡献
2. **Problem Setup and Notation** - 问题定义和数学符号
3. **Algorithm** - SFedAvg算法详细描述
4. **Assumptions and Main Results** - 理论假设和收敛性分析
5. **Federated Local-Drift Lemma** - 核心理论结果
6. **Experiments** - 完整的实验评估
   - 实验设置 (数据生成、参数配置)
   - 超参数优化过程
   - 实验结果分析 (性能表格、图表)
   - 通信效率分析
   - 算法差异讨论
7. **Conclusion** - 总结和未来工作

## 编译论文

### 使用批处理文件 (推荐)
运行 `compile.bat` 自动执行完整的LaTeX编译过程。

### 手动编译
```bash
pdflatex SFedAvg-GoLore.tex
bibtex SFedAvg-GoLore
pdflatex SFedAvg-GoLore.tex
pdflatex SFedAvg-GoLore.tex
```

## 运行实验

### 完整实验 (包含超参数搜索)
```bash
python improved_linear_experiment.py
```

### 快速验证实验
```bash
python simplified_linear_experiment.py
```

### 算法比较分析
```bash
python algorithm_comparison_analysis.py
```

## 主要实验结果

根据最新实验结果 (包含超参数优化):

| 方法 | 最优学习率 | 最优动量 | 最终损失 | 参数误差 | 总通信量(KB) |
|------|-----------|----------|----------|----------|-------------|
| FedAvg | 0.010 | 0.6 | 0.0146 | 0.0295 | 14.1 |
| FedAvgM | 0.010 | 0.6 | 0.0146 | 0.0295 | 14.1 |
| SFedAvg-δ1.00 | 0.005 | 0.6 | 0.0147 | 0.0285 | 14.1 |
| SFedAvg-δ0.50 | 0.020 | 0.6 | 0.0151 | 0.0389 | 7.0 |
| SFedAvg-δ0.25 | 0.050 | 0.6 | 0.0159 | 0.0492 | 3.3 |

### 关键发现
- SFedAvg-δ0.50: 50%通信节省，仅3.2%性能损失
- SFedAvg-δ0.25: 76.7%通信节省，8.6%性能损失  
- 压缩算法需要更高学习率但保持一致动量系数
- 通信效率比率达到16.7× (δ=0.5)

## 技术要求

### Python依赖
- numpy
- scipy  
- matplotlib

### LaTeX依赖
- pdflatex
- bibtex
- 标准LaTeX包 (amsmath, graphicx, booktabs等)

## 文件说明

### 实现文件
- `sfedavg_implementation.py`: 核心SFedAvg算法和Stiefel流形采样
- `*experiment.py`: 不同复杂度的实验实现
- `algorithm_comparison_analysis.py`: 详细的算法对比分析

### 文档文件  
- 各种`.md`报告: 实验分析和项目总结
- `compile.bat`: LaTeX编译脚本

论文完整展示了SFedAvg算法的理论基础、实现细节和实验验证，为通信高效的联邦学习提供了有价值的贡献。