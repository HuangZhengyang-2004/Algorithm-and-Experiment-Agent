# 参数调优说明

**场景**: scalability:_many_clients_and_higher_dimensional_features

**调优时间**: 2025-11-19 15:41:05

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
  "subspace_ratio": 0.03,
  "feature_multiplier": 2.0,
  "client_frac": 0.1,
  "rationale": "Smaller r/d to strongly filter updates under doubled feature dimension while keeping participation modest."
}
```

### Config 2
```json
{
  "subspace_ratio": 0.05,
  "feature_multiplier": 2.5,
  "client_frac": 0.1,
  "rationale": "Scenario baseline-like r/d with more aggressive feature expansion to test projection’s stability in higher d."
}
```

### Config 3
```json
{
  "subspace_ratio": 0.08,
  "feature_multiplier": 2.0,
  "client_frac": 0.1,
  "rationale": "Moderate r/d to retain more signal at 2x features, probing faster convergence vs drift."
}
```

### Config 4
```json
{
  "subspace_ratio": 0.05,
  "feature_multiplier": 3.0,
  "client_frac": 0.15,
  "rationale": "Larger feature expansion with slightly higher client participation to reduce gradient variance."
}
```

### Config 5
```json
{
  "subspace_ratio": 0.1,
  "feature_multiplier": 2.0,
  "client_frac": 0.05,
  "rationale": "Higher r/d for more signal retention when very few clients participate per round."
}
```

### Config 6
```json
{
  "subspace_ratio": 0.07,
  "feature_multiplier": 1.5,
  "client_frac": 0.1,
  "rationale": "Moderate projection with modest feature growth as a conservative stability–speed trade-off."
}
```

### Config 7
```json
{
  "subspace_ratio": 0.04,
  "feature_multiplier": 3.0,
  "client_frac": 0.1,
  "rationale": "Aggressive projection (smaller r/d) for very high dimensionality to cap communication and drift."
}
```

### Config 8
```json
{
  "subspace_ratio": 0.06,
  "feature_multiplier": 2.5,
  "client_frac": 0.2,
  "rationale": "Balanced r/d with higher participation to accelerate averaging and reduce stochasticity in large-N settings."
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
