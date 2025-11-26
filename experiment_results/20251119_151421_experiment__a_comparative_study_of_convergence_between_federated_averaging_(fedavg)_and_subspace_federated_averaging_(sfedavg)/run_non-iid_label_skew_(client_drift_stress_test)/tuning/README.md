# 参数调优说明

**场景**: non-iid_label_skew_(client_drift_stress_test)

**调优时间**: 2025-11-19 15:30:54

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
  "lr": 0.15,
  "local_steps": 4,
  "rationale": "Smaller subspace with slightly reduced lr and fewer local steps to curb drift while leveraging projection benefits."
}
```

### Config 2
```json
{
  "subspace_ratio": 0.15,
  "lr": 0.12,
  "local_steps": 5,
  "rationale": "Moderate subspace size with conservative lr at default tau to balance learning speed and stability."
}
```

### Config 3
```json
{
  "subspace_ratio": 0.25,
  "lr": 0.1,
  "local_steps": 3,
  "rationale": "Default r/d with lower lr and shorter local blocks to reduce client drift yet keep adequate update magnitude."
}
```

### Config 4
```json
{
  "subspace_ratio": 0.05,
  "lr": 0.2,
  "local_steps": 3,
  "rationale": "Very small subspace to strongly filter drift, compensated by default lr and fewer steps to avoid underfitting."
}
```

### Config 5
```json
{
  "subspace_ratio": 0.35,
  "lr": 0.08,
  "local_steps": 6,
  "rationale": "Larger subspace approaching full space; reduce lr and allow slightly longer local blocks to test stability vs speed."
}
```

### Config 6
```json
{
  "subspace_ratio": 0.2,
  "lr": 0.18,
  "local_steps": 4,
  "rationale": "Intermediate r/d with modest lr and reduced tau to probe faster convergence without excessive drift."
}
```

### Config 7
```json
{
  "subspace_ratio": 0.1,
  "lr": 0.25,
  "local_steps": 2,
  "rationale": "Aggressive lr with very short local blocks and small subspace to test if quick updates can still be stable under projection."
}
```

### Config 8
```json
{
  "subspace_ratio": 0.3,
  "lr": 0.12,
  "local_steps": 8,
  "rationale": "Larger r/d with conservative lr and longer local blocks to explore projection’s ability to stabilize extended local training."
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
