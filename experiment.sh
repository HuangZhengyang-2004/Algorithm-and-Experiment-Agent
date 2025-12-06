#!/usr/bin/env bash
# 通过 AIHubMix 的 OpenAI 接口调用 launch_experiment.py

set -euo pipefail


# 预设参数
# BASE="https://aihubmix.com/v1"
BASE="https://xh.v1api.cc/v1"
KEY="sk-Z4L5IKRfLXwDMllWX2rWMsuZ0Is0ceIxJz2HMIp7RPdUpSbN"  # 请确保安全存储 API 密钥
MODEL="gpt-5"  # 使用标准的 OpenAI 模型名称
# BASE="https://api.deepseek.com/v1"
# KEY="sk-2a5e2ec89e23451a87bf333f04c582cc"  # 请确保安全存储 API 密钥
# MODEL="deepseek-reasoner"
OUT_DIR="./experiment_results"
IDEA_FILE="1.json"
ALGORITHM_TEX="algorithm.tex"  # 算法伪代码文件

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [-h]

  -h             显示本帮助
EOF
}

# 组装传给 launch_experiment.py 的 CLI 参数
CLI_ARGS=(--idea-file "$IDEA_FILE" --output-dir "$OUT_DIR" --model "$MODEL" --algorithm-tex "$ALGORITHM_TEX")
ENV_ARGS=()

# 导出环境变量 - Aider/LiteLLM 需要这些环境变量
# 注意：URL 需要包含 /v1 后缀
export OPENAI_API_BASE="$BASE"
export OPENAI_API_KEY="$KEY"
# export DEEPSEEK_API_KEY="$KEY"
# LiteLLM 可能还需要这些变量
export LITELLM_LOG="DEBUG"  # 启用调试日志以查看详细错误
echo "ℹ️ 设置 OPENAI_API_BASE=$BASE"
echo "ℹ️ 设置 OPENAI_API_KEY=(hidden)"

# 最终参数
FINAL_ARGS=("${CLI_ARGS[@]}" "${ENV_ARGS[@]}")

# 调用 launch_experiment.py
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "ℹ️ 调用: python3 $SCRIPT_DIR/launch_experiment.py ${FINAL_ARGS[*]}"
python3 "$SCRIPT_DIR/launch_experiment.py" "${FINAL_ARGS[@]}"