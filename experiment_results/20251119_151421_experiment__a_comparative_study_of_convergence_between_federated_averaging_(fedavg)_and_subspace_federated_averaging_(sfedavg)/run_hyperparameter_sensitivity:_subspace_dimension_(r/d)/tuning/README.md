# 参数调优说明

**场景**: hyperparameter_sensitivity:_subspace_dimension_(r/d)

**调优时间**: 2025-11-19 15:45:37

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
  "subspace_ratio": 0.05,
  "lr": 0.12,
  "client_frac": 0.2,
  "rationale": "Small r/d to assess slower learning due to strong projection; baseline lr and participation."
}
```

### Config 2
```json
{
  "subspace_ratio": 0.1,
  "lr": 0.12,
  "client_frac": 0.2,
  "rationale": "Mid r/d expected sweet spot balancing convergence speed and projection benefit."
}
```

### Config 3
```json
{
  "subspace_ratio": 0.25,
  "lr": 0.12,
  "client_frac": 0.2,
  "rationale": "Large r/d approaching momentum SGD; tests diminished projection benefits."
}
```

### Config 4
```json
{
  "subspace_ratio": 0.1,
  "lr": 0.1,
  "client_frac": 0.2,
  "rationale": "Sweet-spot r/d with conservative lr to probe stability improvements."
}
```

### Config 5
```json
{
  "subspace_ratio": 0.1,
  "lr": 0.15,
  "client_frac": 0.2,
  "rationale": "Sweet-spot r/d with higher lr to test faster progress vs potential instability."
}
```

### Config 6
```json
{
  "subspace_ratio": 0.1,
  "lr": 0.12,
  "client_frac": 0.3,
  "rationale": "Sweet-spot r/d with higher participation to reduce aggregation variance and accelerate convergence."
}
```

### Config 7
```json
{
  "subspace_ratio": 0.05,
  "lr": 0.12,
  "client_frac": 0.3,
  "rationale": "Small r/d with more clients per round to compensate for slower learning via reduced variance."
}
```

### Config 8
```json
{
  "subspace_ratio": 0.25,
  "lr": 0.12,
  "client_frac": 0.1,
  "rationale": "Large r/d with fewer clients to explore drift and communication trade-offs under reduced participation."
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
