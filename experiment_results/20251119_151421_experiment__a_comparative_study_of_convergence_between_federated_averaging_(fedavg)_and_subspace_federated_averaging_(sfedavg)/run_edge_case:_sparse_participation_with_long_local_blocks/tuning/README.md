# 参数调优说明

**场景**: edge_case:_sparse_participation_with_long_local_blocks

**调优时间**: 2025-11-19 15:52:19

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
  "client_frac": 0.05,
  "local_steps": 20,
  "rationale": "Strong projection (smaller r/d) with baseline sparse participation and long blocks to maximally damp drift."
}
```

### Config 2
```json
{
  "subspace_ratio": 0.1,
  "client_frac": 0.05,
  "local_steps": 20,
  "rationale": "Scenario’s baseline r/d; tests whether moderate projection suffices to stabilize high drift."
}
```

### Config 3
```json
{
  "subspace_ratio": 0.15,
  "client_frac": 0.05,
  "local_steps": 20,
  "rationale": "Larger r/d closer to momentum SGD; probes potential speed gains versus reduced projection benefits under extreme drift."
}
```

### Config 4
```json
{
  "subspace_ratio": 0.1,
  "client_frac": 0.03,
  "local_steps": 20,
  "rationale": "Even sparser participation to stress drift; evaluates projection’s robustness as aggregation variance increases."
}
```

### Config 5
```json
{
  "subspace_ratio": 0.1,
  "client_frac": 0.1,
  "local_steps": 20,
  "rationale": "More clients per round to reduce variance; checks if increased averaging compensates long local blocks."
}
```

### Config 6
```json
{
  "subspace_ratio": 0.1,
  "client_frac": 0.05,
  "local_steps": 25,
  "rationale": "Longer local blocks further increase drift; assesses whether projection prevents oscillations."
}
```

### Config 7
```json
{
  "subspace_ratio": 0.1,
  "client_frac": 0.05,
  "local_steps": 15,
  "rationale": "Shorter local blocks reduce drift; compares stability improvements without changing participation."
}
```

### Config 8
```json
{
  "subspace_ratio": 0.05,
  "client_frac": 0.1,
  "local_steps": 25,
  "rationale": "Strong projection with more clients but very long blocks; tests trade-off between filtering, averaging, and drift."
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
