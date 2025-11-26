#!/usr/bin/env python3
"""
修复实验结果目录中的 plot.py labels 字典

使用方法：
    python fix_plot_labels.py <experiment_results_dir>

例如：
    python fix_plot_labels.py experiment_results/20251117_155922_experiment_...
"""

import os
import sys
import re
import subprocess

def fix_main_plot_py(exp_dir):
    """修复主目录的 plot.py，填充 labels 字典"""
    plot_file = os.path.join(exp_dir, "plot.py")
    
    if not os.path.exists(plot_file):
        print(f"❌ plot.py 不存在: {plot_file}")
        return False
    
    # 查找所有 run_N/ 目录
    run_dirs = []
    for item in os.listdir(exp_dir):
        item_path = os.path.join(exp_dir, item)
        if os.path.isdir(item_path) and item.startswith("run_"):
            # 检查是否有 final_info.json
            if os.path.exists(os.path.join(item_path, "final_info.json")):
                run_dirs.append(item)
    
    if not run_dirs:
        print(f"⚠️  未找到有效的 run_N/ 目录")
        return False
    
    # 按数字排序
    run_dirs.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 999)
    
    # 读取 plot.py
    with open(plot_file, 'r', encoding='utf-8') as f:
        plot_content = f.read()
    
    # 构建 labels 字典
    labels_lines = ["labels = {"]
    for run_dir in run_dirs:
        labels_lines.append(f'    "{run_dir}": "{run_dir}",')
    labels_lines.append("}")
    labels_str = "\n".join(labels_lines)
    
    # 替换 labels 字典
    pattern = r'labels\s*=\s*\{[^}]*\}'
    if re.search(pattern, plot_content):
        plot_content = re.sub(pattern, labels_str, plot_content)
        
        # 写回
        with open(plot_file, 'w', encoding='utf-8') as f:
            f.write(plot_content)
        
        print(f"✅ 已修复主 plot.py，填充了 {len(run_dirs)} 个 run 目录")
        return True
    else:
        print("⚠️  无法找到 labels 字典定义")
        return False


def regenerate_plots(exp_dir):
    """重新生成图表"""
    plot_file = os.path.join(exp_dir, "plot.py")
    plots_dir = os.path.join(exp_dir, "plots")
    
    # 创建 plots 目录
    os.makedirs(plots_dir, exist_ok=True)
    
    # 运行 plot.py
    print(f"\n📊 重新生成图表...")
    try:
        result = subprocess.run(
            ["python", plot_file, f"--out_dir={plots_dir}"],
            cwd=exp_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ 图表生成成功")
            print(result.stdout)
            return True
        else:
            print(f"❌ 图表生成失败")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python fix_plot_labels.py <experiment_results_dir>")
        sys.exit(1)
    
    exp_dir = sys.argv[1]
    
    if not os.path.exists(exp_dir):
        print(f"❌ 目录不存在: {exp_dir}")
        sys.exit(1)
    
    print(f"🔧 修复实验结果: {exp_dir}\n")
    
    # 修复主 plot.py
    if fix_main_plot_py(exp_dir):
        # 重新生成图表
        regenerate_plots(exp_dir)
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()












