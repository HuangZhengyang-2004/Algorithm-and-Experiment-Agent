"""
算法实验启动器 (Algorithm Agent)
============================

功能: 根据 idea.json 中的实验描述，自动执行算法实验

核心功能:
  1. 读取实验想法
  2. 初始化 AI Coder
  3. 执行实验流程
  4. 生成结果和报告

使用方法:
    python launch_experiment.py \\
        --idea-file idea.json \\
        --output-dir ./results \\
        --model gpt-4o-mini
"""

import argparse
import json
import os
import os.path as osp
import sys
from datetime import datetime

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

# 导入核心实验函数
from perform_experiments import perform_experiments


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="算法实验启动器 (Algorithm Agent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python launch_experiment.py \\
    --idea-file idea.json \\
    --output-dir ./results \\
    --model gpt-4o-mini
        """
    )
    
    parser.add_argument(
        "--idea-file",
        type=str,
        required=True,
        help="实验想法 JSON 文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_results",
        help="输出目录路径 (默认: ./experiment_results)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM 模型名称 (默认: gpt-4o-mini)"
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="Baseline 结果 JSON 文件路径 (可选)"
    )
    parser.add_argument(
        "--algorithm-tex",
        type=str,
        default=None,
        help="Path to algorithm.tex (optional). If provided, this file will be used as pseudocode source."
    )
    
    return parser.parse_args()


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def validate_idea_json(idea):
    """
    验证 idea.json 的格式
    
    必需字段:
      - Title: 实验标题
      - Experiment: 实验描述
    """
    required_fields = ["Title", "Experiment"]
    for field in required_fields:
        if field not in idea:
            raise ValueError(f"idea.json 缺少必需字段: {field}")
    
    return True


def main():
    args = parse_arguments()
    
    print("=" * 80)
    print("🚀 算法实验启动器 (Algorithm Agent)")
    print("=" * 80)
    print()
    
    # 1. 读取实验想法
    print(f"📝 读取实验想法: {args.idea_file}")
    try:
        with open(args.idea_file, "r", encoding="utf-8") as f:
            idea = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在: {args.idea_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON 解析失败: {e}")
        sys.exit(1)
    
    # 验证格式
    try:
        validate_idea_json(idea)
    except ValueError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    
    idea_title = idea.get("Title", "Unknown")
    print(f"   实验标题: {idea_title}")
    experiment_desc = idea.get("Experiment", "")
    if len(experiment_desc) > 100:
        print(f"   实验描述: {experiment_desc[:100]}...")
    else:
        print(f"   实验描述: {experiment_desc}")
    print()
    
    # 2. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用实验标题作为文件夹名，替换所有特殊字符
    safe_name = (idea_title
                 .replace(" ", "_")
                 .replace("/", "_")
                 .replace(":", "_")  # 替换冒号
                 .replace("?", "_")  # 替换问号
                 .replace("*", "_")  # 替换星号
                 .replace("|", "_")  # 替换竖线
                 .replace("<", "_")  # 替换小于号
                 .replace(">", "_")  # 替换大于号
                 .replace("\"", "_") # 替换双引号
                 .replace("\\", "_") # 替换反斜杠
                 .lower())
    folder_name = osp.join(args.output_dir, f"{timestamp}_{safe_name}")
    
    print(f"📁 创建实验文件夹: {folder_name}")
    os.makedirs(folder_name, exist_ok=True)
    
    # 保存原始 idea.json 到实验文件夹
    with open(osp.join(folder_name, "idea.json"), "w", encoding="utf-8") as f:
        json.dump(idea, f, indent=2, ensure_ascii=False)
    
    # 创建初始文件
    experiment_file = osp.join(folder_name, "experiment.py")
    plot_file = osp.join(folder_name, "plot.py")
    notes_file = osp.join(folder_name, "notes.txt")
    
    # 创建初始框架代码（为 Aider diff 模式提供可匹配的内容）
    if not osp.exists(experiment_file):
        with open(experiment_file, "w") as f:
            f.write("""# Experiment file - to be implemented by AI

import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # TODO: Implement algorithm here
    
    # Save results
    results = {}
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
""")
    
    if not osp.exists(plot_file):
        with open(plot_file, "w") as f:
            f.write("""# Plot file - to be implemented by AI

import matplotlib.pyplot as plt
import json
import os

labels = {}

def plot_results():
    # TODO: Implement plotting here
    pass

if __name__ == "__main__":
    plot_results()
""")
    
    # 创建初始 notes.txt
    with open(notes_file, "w") as f:
        f.write(f"# Title: {idea_title}\n")
        f.write(f"# Experiment description: {experiment_desc}\n")
        f.write(f"# Timestamp: {timestamp}\n")
        f.write(f"\n## Experiment Log\n\n")
    
    print()
    
    # 3. 读取 baseline 结果（如果提供）
    baseline_results = {}
    if args.baseline_results:
        print(f"📊 读取 Baseline 结果: {args.baseline_results}")
        try:
            with open(args.baseline_results, "r", encoding="utf-8") as f:
                baseline_results = json.load(f)
            print(f"   Baseline 结果: {baseline_results}")
        except Exception as e:
            print(f"⚠️ 警告: 无法读取 baseline 结果: {e}")
            baseline_results = {}
        print()
    
    # 4. 初始化 Aider Coder
    print(f"🤖 初始化 AI Coder (模型: {args.model})")
    
    fnames = [experiment_file, plot_file, notes_file]
    io = InputOutput(
        yes=True,
        chat_history_file=f"{folder_name}/aider_chat.txt"
    )
    
    # 根据模型类型创建 Model 对象
    if args.model == "deepseek-coder-v2-0724":
        main_model = Model("deepseek/deepseek-coder")
    elif args.model == "deepseek-reasoner":
        main_model = Model("deepseek/deepseek-reasoner")
    elif args.model == "llama3.1-405b":
        main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
    else:
        main_model = Model(args.model)
    
    coder = Coder.create(
        main_model=main_model,
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )
    print()
    
    # 5. 执行实验
    print("=" * 80)
    print("🧪 开始执行实验")
    print("=" * 80)
    print()
    print_time()
    
    try:
        success = perform_experiments(
            idea=idea,
            folder_name=folder_name,
            coder=coder,
            baseline_results=baseline_results,
            algorithm_tex_path=args.algorithm_tex
        )
        
        print()
        print_time()
        print()
        
        if success:
            print("=" * 80)
            print("✅ 实验成功完成!")
            print("=" * 80)
            print()
            print(f"📂 结果保存在: {folder_name}")
            print()
            print("生成的文件:")
            print(f"  - experiment.py          : 实验运行脚本")
            print(f"  - plot.py                : 可视化脚本")
            print(f"  - run_1/, run_2/, ...    : 实验结果")
            print(f"  - *.png                  : 可视化图表")
            print(f"  - notes.txt              : 实验笔记")
            print(f"  - aider_chat.txt         : AI 对话历史")
            print()
            return 0
        else:
            print("=" * 80)
            print("❌ 实验执行失败")
            print("=" * 80)
            print()
            print(f"请检查:")
            print(f"  - {folder_name}/aider_chat.txt (AI 对话历史)")
            print(f"  - {folder_name}/notes.txt (实验笔记)")
            print()
            return 1
            
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ 实验执行出错")
        print("=" * 80)
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

