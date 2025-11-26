# 参数调优说明

**场景**: robustness_to_label_noise

**调优时间**: 2025-11-19 15:35:42

## 文件结构

```
tuning/
├── experiment_tuning.py    ← 调优时使用的代码（包含 --enable_tuning 支持）
├── plot.py                 ← 绘图脚本
├── all_configs.json        ← 所有配置的汇总结果
├── README.md               ← 本说明文件
├── config_1/               ← 配置1的结果
│   ├── final_info.json    ← 实验结果
│   └── config.json        ← 配置参数
├── config_2/               ← 配置2的结果
│   ├── final_info.json
│   └── config.json
└── ...
```

## 调优配置

共测试了 8 个参数配置：

### Config 1
```json
{
  "subspace_ratio": 0.1,
  "lr": 0.12,
  "batch_size": 64,
  "rationale": "Strong projection (smaller r/d) with moderate lr and a larger batch to aggressively filter noisy components while maintaining learning speed."
}
```

### Config 2
```json
{
  "subspace_ratio": 0.15,
  "lr": 0.1,
  "batch_size": 64,
  "rationale": "Slightly larger subspace with conservative lr to enhance stability; batch_size 64 reduces gradient variance from label noise."
}
```

### Config 3
```json
{
  "subspace_ratio": 0.2,
  "lr": 0.12,
  "batch_size": 128,
  "rationale": "Moderate projection with a bigger batch to counteract noise and stabilize updates; keeps lr moderate for faster convergence."
}
```

### Config 4
```json
{
  "subspace_ratio": 0.25,
  "lr": 0.15,
  "batch_size": 64,
  "rationale": "Baseline-like r/d but slightly reduced lr relative to default, plus larger batch to test if projection at standard dimension remains robust."
}
```

### Config 5
```json
{
  "subspace_ratio": 0.3,
  "lr": 0.12,
  "batch_size": 96,
  "rationale": "Higher r/d to retain more signal, balanced by moderate lr and enlarged batch to keep noisy gradients in check."
}
```

### Config 6
```json
{
  "subspace_ratio": 0.1,
  "lr": 0.15,
  "batch_size": 128,
  "rationale": "Aggressive filtering with small r/d paired with higher lr, compensated by very large batch to dampen noise-induced instability."
}
```

### Config 7
```json
{
  "subspace_ratio": 0.35,
  "lr": 0.1,
  "batch_size": 64,
  "rationale": "Large subspace approaching full space but with conservative lr to avoid amplifying noise; batch 64 provides variance reduction."
}
```

### Config 8
```json
{
  "subspace_ratio": 0.2,
  "lr": 0.08,
  "batch_size": 96,
  "rationale": "Very conservative lr for maximum stability under noise, with moderate r/d and larger batch to ensure steady progress."
}
```


## 使用说明

### 查看结果
```bash
# 查看所有配置的汇总结果
cat all_configs.json | jq .

# 查看单个配置的参数
cat config_1/config.json | jq .

# 查看单个配置的实验结果
cat config_1/final_info.json | jq .

# 对比所有配置的最终准确率
for dir in config_*/; do
  echo "$dir:"
  cat "$dir/config.json" | jq -r '.parameters | to_entries | map("\(.key)=\(.value)") | join(", ")'
  cat "$dir/final_info.json" | jq '.SFedAvg.test_accuracy.means[-1]' 2>/dev/null || echo 'N/A'
  echo
done
```

### 重现实验
```bash
# 使用调优时的代码和配置重现实验
python experiment_tuning.py --out_dir=new_run --enable_tuning
```
