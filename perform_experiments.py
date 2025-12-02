"""
实验执行引擎 (Experiment Execution Engine)
==========================================

本模块负责执行 AI 驱动的自动化实验流程，包括：
0. 代码生成阶段（根据伪代码生成初始实验代码）
1. 迭代实验循环（AI 修改代码 → 执行 → 反馈）
2. 可视化生成
3. 文档更新

核心流程：
  阶段 0: 代码生成与验证 (generate_code_from_pseudocode)
    - AI 根据伪代码生成 experiment.py 和 plot.py
    - 系统验证代码可运行性（语法检查 + 试运行）
    - 迭代修复直到代码可以正常运行
    - 检验代码是否符合算法伪代码，迭代修复知道运行成功 --new
  
  阶段 1: 实验循环 (perform_experiments)
    - AI 根据想法生成/修改 experiment.py
    - 系统执行实验并收集结果
    - 将结果反馈给 AI，进行下一轮迭代
  
  阶段 2: 可视化生成 (run_plotting)
    - AI 修改 plot.py 生成对比图表
    - 系统执行绘图脚本
  
  阶段 3: 文档更新
    - AI 更新 notes.txt，记录实验发现和结论
"""

import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from datetime import datetime
from subprocess import TimeoutExpired

# ============================================================================
# 配置常量
# ============================================================================

MAX_ITERS = 4           # 每次运行失败后的最大重试次数
MAX_RUNS = 5            # 总共允许的最大实验运行次数
MAX_STDERR_OUTPUT = 1500  # 错误信息的最大显示长度（字符数）
MAX_CODE_GEN_ITERS = 10  # 代码生成阶段的最大迭代次数

# 参数调优配置
ENABLE_HYPERPARAMETER_TUNING = True  # 是否启用场景级参数调优（场景执行成功后立即调优）
MAX_TUNING_CONFIGS_PER_SCENARIO = 8  # 每个场景最多测试的参数配置数量

# ============================================================================
# AI 提示词模板 - 阶段 0: 代码生成
# ============================================================================

code_generation_prompt = """You are an expert algorithm researcher and programmer. Your task is to implement a complete experimental framework based on the following algorithm pseudocode.

# Algorithm Title
{title}

# Algorithm Pseudocode
{pseudocode}

# Experiment Description
{experiment_description}

# Your Task
The files experiment.py and plot.py currently contain only basic skeleton code. You need to REPLACE the TODO sections with complete implementations of the algorithm and experimental framework.

Implement the following:

## 1. experiment.py
This file must contain:
- Complete implementation of the algorithm described in the pseudocode
- Data generation/loading functions
- Training/optimization loop
- Evaluation metrics computation
- Command-line argument parsing with **required** --out_dir parameter
- Result saving to {{out_dir}}/final_info.json

**Critical Requirements for experiment.py:**
```python
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    
    # CRITICAL: Expose ALL algorithm-specific parameters as command-line arguments!
    # For federated learning / subspace methods, include:
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of rounds/epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum coefficient')
    # For subspace-based algorithms (SFedAvg, GaLore, etc.):
    parser.add_argument('--subspace_dim', type=int, default=10, help='Subspace dimension r')
    parser.add_argument('--local_steps', type=int, default=5, help='Local SGD steps tau')
    parser.add_argument('--client_fraction', type=float, default=0.5, help='Client sampling fraction')
    # Add other algorithm-specific parameters as needed
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # ... your algorithm implementation ...
    
    # Save results in the required format
    results = {{
        "metric_name": {{
            "means": [value1, value2, ...],  # List of values over iterations/epochs
            "stds": [std1, std2, ...]        # Standard deviations (can be zeros)
        }}
    }}
    
    with open(f"{{args.out_dir}}/final_info.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

**IMPORTANT: Parameter Exposure**
- DO NOT hardcode algorithm hyperparameters (learning_rate, momentum, subspace_dim, etc.)
- ALWAYS expose them as command-line arguments with reasonable defaults
- This enables hyperparameter tuning to optimize ALL aspects of the algorithm
- For comparison studies (e.g., FedAvg vs SFedAvg), ensure both algorithms can use the same parameter interface

**CRITICAL OUTPUT STRUCTURE REQUIREMENT:**
- Results MUST be saved to: {{out_dir}}/baseline/final_info.json
- Create a 'baseline' subdirectory to store results
- This enables future parameter tuning experiments to be organized separately
- File structure:
  ```
  {{out_dir}}/
      ├── baseline/
      │   └── final_info.json    ← Save results here
      └── experiment.py (snapshot)
  ```
- Implementation:
  ```python
  baseline_dir = os.path.join(args.out_dir, "baseline")
  os.makedirs(baseline_dir, exist_ok=True)
  
  # ... run experiment ...
  
  with open(os.path.join(baseline_dir, "final_info.json"), "w") as f:
      json.dump(results, f, indent=2)
  ```

**CRITICAL IMPLEMENTATION REQUIREMENTS:**

1. **NO PLACEHOLDER CODE:**
   - NEVER use `np.random.randn()` or `np.random.rand()` for gradients, losses, or training data
   - NEVER leave `TODO` comments or `Placeholder` text in the final code
   - ALL functions must be FULLY implemented with real computations
   - Every value must be computed from actual data and model predictions

2. **REAL GRADIENT COMPUTATION:**
   - Gradients MUST be computed from actual model predictions and data
   - For regression: `gradient = X.T @ (predictions - y) / n_samples`
   - For classification: use appropriate loss function derivatives
   - NEVER use random values as gradients

3. **PROPER MODEL INITIALIZATION:**
   - In federated learning: local models MUST copy global model weights
   - Use: `local_model.weights = global_model.weights.copy()`
   - DO NOT create new random models for each client

4. **LEARNING VERIFICATION:**
   - Your implementation MUST show learning progress
   - Loss/error values SHOULD decrease over training iterations
   - If metrics don't change, your implementation is WRONG
   - For optimization: final values should be significantly better than initial values

5. **DATA USAGE:**
   - Use the ACTUAL training data provided or generated
   - Compute predictions using the model: `predictions = model.predict(X)`
   - Compute errors/losses from predictions and true labels
   - Update model based on these real computations

## 2. plot.py
This file must contain:
- Functions to read results from multiple run directories
- Plotting code for convergence curves and comparisons
- A `labels` dictionary to specify which runs to plot
- MUST accept --out_dir command-line argument for saving plots
- Save plots as PNG files

**Example structure:**
```python
import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Dictionary mapping run directories to labels
    labels = {{
        "run_1": "Baseline",
        "run_2": "Variant 1",
        # Will be updated later with more runs
    }}
    
    # Initialize matplotlib with professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    
    # Colors for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    
    for i, (run_dir, label) in enumerate(labels.items()):
        if os.path.exists(f"{{run_dir}}/final_info.json"):
            with open(f"{{run_dir}}/final_info.json") as f:
                data = json.load(f)
            
            # Plot each metric
            for metric_name, metric_data in data.items():
                means = metric_data["means"]
                stds = metric_data["stds"]
                iterations = range(len(means))
                
                # Plot mean with standard deviation shading
                axes.plot(iterations, means, label=f"{{label}} - {{metric_name}}", 
                         color=colors[i], linewidth=2)
                axes.fill_between(iterations, 
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.2, color=colors[i])
    
    # Customize plot
    axes.set_xlabel('Iterations/Epochs', fontsize=12)
    axes.set_ylabel('Metric Value', fontsize=12)
    axes.set_title('Algorithm Performance Comparison', fontsize=14)
    axes.legend()
    axes.grid(True, alpha=0.3)
    
    # Save plots to the specified output directory
    plot_path = os.path.join(args.out_dir, "comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to: {{plot_path}}")

if __name__ == "__main__":
    main()
```
**CRITICAL REQUIREMENTS for plot.py:**
    - MUST use matplotlib for all visualizations
    - MUST accept --out_dir argument for specifying where to save plots
    - MUST save plots to {{args.out_dir}}/plot_name.png
    - Should create professional-looking plots with:
        - Clear labels and legends
        - Grid for readability
        - Error bars or shaded regions for standard deviations
        - High resolution (dpi=300)
        - Should handle multiple metrics and multiple runs appropriately
**IMPORTANT:**
Always test your plot.py locally before submission
- Run: python plot.py --out_dir=test_plots
- Check that it creates the directory and saves PNG files
- Ensure no import errors or runtime errors

## 3. Key Implementation Guidelines
- Follow the pseudocode logic closely
- Use appropriate libraries (numpy, scipy, sklearn, torch, etc.)
- Include error handling and input validation
- Add comments explaining key algorithmic steps
- Make the code production-ready and well-documented
- Ensure experiment.py can run standalone with: `python experiment.py --out_dir=test_run`

## 4. Testing Requirements
The generated code must:
- Pass Python syntax validation (no syntax errors)
- Run without crashing (handle imports, data generation, etc.)
- Generate the required output file (final_info.json) with correct format
- Complete execution within reasonable time (for initial test)

## 5. Important Note on Future Modifications
After initial generation, all future modifications should be MINIMAL and TARGETED:
- Use search/replace for specific fixes
- Make surgical edits to only the problematic sections
- DO NOT regenerate entire files unless absolutely necessary
- This is to conserve computational resources

Please generate BOTH experiment.py and plot.py now. Make sure they are complete, runnable, and follow all requirements above.

**Note: This is initial generation, so full file content is needed. Future fixes will use targeted edits only.**
"""

# ============================================================================
# AI 提示词模板 - 阶段 0: 代码逻辑验证
# ============================================================================

logic_validation_prompt = """Now I need you to verify that the current implementation in experiment.py correctly implements the algorithm pseudocode.

# Algorithm Title
{title}

# Algorithm Pseudocode
{pseudocode}

# Current experiment.py Code
{current_code}

Please carefully compare the implementation with the pseudocode and check for:

1. **Algorithm Fidelity:**
   - Does the code correctly implement all steps from the pseudocode?
   - Are there any missing algorithmic components?
   - Are there any deviations from the pseudocode logic?

2. **Key Components:**
   - Are all variables, functions, and procedures from the pseudocode properly implemented?
   - Are the data structures and control flows consistent with the pseudocode?
   - Are the mathematical formulas and computations correctly translated?

3. **Critical Checks:**
   - If the pseudocode mentions specific conditions, are they properly handled?
   - If there are loops or iterations, do they match the pseudocode's structure?
   - Are the termination conditions and convergence criteria correctly implemented?

4. **Implementation Quality:**
   - Is the code using appropriate data types and operations?
   - Are there any placeholder or dummy implementations that should be real computations?
   - Does the implementation show meaningful learning progress (not constant values)?

Please respond with:
- "LOGIC_VALIDATION_PASSED" if the code correctly implements the pseudocode
- Otherwise, explain what needs to be fixed and make MINIMAL, TARGETED changes

**CRITICAL: If fixes are needed, DO NOT rewrite the entire file!**
- Identify the specific algorithmic discrepancies
- Make SURGICAL edits to fix only those parts
- Use search/replace for targeted modifications
- Preserve all correctly implemented sections
"""

# ============================================================================
# AI 提示词模板 - 阶段 1: 实验迭代
# ============================================================================

coder_prompt = """Your goal is to implement the following idea: {title}.
The proposed experiment is as follows: {idea}.
You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

After you complete each change, we will run the command `python experiment.py --out_dir=run_i' where i is the run number and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
You can then implement the next thing on your list.

**CRITICAL INSTRUCTION FOR ALL MODIFICATIONS:**
- Use MINIMAL, TARGETED EDITS when modifying experiment.py
- DO NOT rewrite or output the entire file
- Use search/replace or focused edits for specific changes
- Only modify the sections that need to change for each run
- This conserves computational resources and reduces errors"""


# ============================================================================
# 新增：AI 自主场景设计阶段（阶段 0.5）
# ============================================================================

def design_experiment_scenarios(idea, folder_name, coder, baseline_results=None, algorithm_tex_path=None):
    """
    AI 自主设计实验场景 - 新增阶段 0.5
    在代码生成之后、实验循环之前执行

    本函数将基于 algorithm.tex（优先）读取伪代码；若提供了 idea["Experiment"]
    则可作为参考但不是主要来源。
    """
    print("\n" + "="*80)
    print("🎯 阶段 0.5: AI 自主场景设计")
    print("="*80 + "\n")
    
    # 从 algorithm.tex 读取伪代码（不使用 idea['Pseudocode']）
    pseudocode_text = read_pseudocode_from_tex(folder_name=folder_name, tex_path=algorithm_tex_path) or ""
    
    # 准备算法信息
    algorithm_info = {
        "title": idea.get("Title", "Unknown Algorithm"),
        "description": idea.get("Experiment", ""),
        "pseudocode": pseudocode_text,
        "key_parameters": extract_available_parameters(folder_name)
    }
    
    # 让 AI 设计实验场景
    scenario_prompt = generate_scenario_design_prompt(algorithm_info, baseline_results)
    print("🤖 AI 正在设计实验场景...")
    ai_response = coder.run(scenario_prompt)
    print(ai_response)
    
    # 解析 AI 设计的场景
    scenario_design = parse_ai_scenario_design(ai_response)
    
    if not scenario_design or "scenarios" not in scenario_design:
        print("❌ AI 未能正确设计实验场景，使用默认场景")
        scenario_design = get_default_scenarios()
    
    scenarios = scenario_design["scenarios"]
    
    # 显示设计的场景
    print(f"\n📋 AI 设计了 {len(scenarios)} 个实验场景:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"   {i}. {scenario.get('name', f'Scenario_{i}')}")
        print(f"      描述: {scenario.get('description', 'No description')}")
        if 'parameters' in scenario:
            params_str = ' '.join([f"{k}={v}" for k, v in scenario['parameters'].items()])
            print(f"      参数: {params_str}")
        if 'expected_insight' in scenario:
            print(f"      预期发现: {scenario['expected_insight']}")
        print()
    
    # 保存场景设计到文件
    scenarios_file = osp.join(folder_name, "ai_designed_scenarios.json")
    with open(scenarios_file, 'w', encoding='utf-8') as f:
        json.dump(scenario_design, f, indent=2, ensure_ascii=False)
    
    print(f"💾 场景设计已保存到: {scenarios_file}")
    
    return scenario_design


def extract_available_parameters(folder_name):
    """
    从 experiment.py 中提取可用的命令行参数
    """
    exp_file = osp.join(folder_name, "experiment.py")
    
    if not osp.exists(exp_file):
        return ["--learning_rate", "--num_iterations", "--dataset_size"]  # 默认参数
    
    try:
        with open(exp_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取 argparse 参数
        import re
        param_pattern = r'parser\.add_argument\([^)]*?--([a-zA-Z_]+)[^)]*?\)'
        params = re.findall(param_pattern, content)
        
        # 排除 out_dir 和 scenario（如果存在）
        excluded_params = ['out_dir', 'scenario']
        params = [f"--{p}" for p in params if p not in excluded_params]
        
        return params if params else ["--learning_rate", "--num_iterations", "--dataset_size"]
        
    except Exception as e:
        print(f"⚠️ 提取参数时出错: {e}")
        return ["--learning_rate", "--num_iterations", "--dataset_size"]


# ...existing code...
def generate_scenario_design_prompt(algorithm_info, existing_results=None):
    """
    Generate an English prompt that instructs the AI to design experimental scenarios.
    Primary sources: algorithm title and pseudocode. The optional 'description' field
    (idea["Experiment"]) MAY be used as a reference for dataset preferences, constraints,
    or evaluation priorities, but DO NOT use it as the primary source of scenario design.

    The prompt asks for 3-5 scenarios covering heterogeneity, robustness, scalability,
    and hyperparameter sensitivity, and requests output in a strict JSON format.
    """
    title = algorithm_info.get('title', 'Unknown Algorithm')
    pseudocode = algorithm_info.get('pseudocode', '').strip()
    description = algorithm_info.get('description', '').strip()
    key_params = algorithm_info.get('key_parameters', [])
    params_str = ', '.join(key_params) if key_params else 'no specific CLI parameters detected'

    prompt = f"""
You are an expert algorithm researcher. DESIGN experimental scenarios PRIMARILY based on the
Algorithm TITLE and its PSEUDOCODE below. You MAY consult the optional "Experiment description"
for helpful context (e.g., suggested datasets, constraints, or evaluation preferences), but
DO NOT rely on it as the main source of scenario design. The scenarios must be justified by
the TITLE and PSEUDOCODE.

Algorithm Title:
{title}

Algorithm Pseudocode:
{pseudocode if pseudocode else '<NO PSEUDOCODE PROVIDED>'}

Optional Experiment description (use only as reference, not primary source):
{description if description else '<NO EXPERIMENT DESCRIPTION PROVIDED>'}

Available command-line parameters (if any):
{params_str}

Your task:
Produce 3 to 5 well-motivated experimental scenarios that thoroughly evaluate the algorithm.
Each scenario must include:
- A concise scenario name
- A one-sentence objective describing what aspect is tested
- Specific command-line parameter settings (use only the available parameters above; if none, propose sensible parameter names and values)
- A short description of the dataset or data modifications to use (e.g., heterogeneity across clients, added label noise, different data sizes)
- The expected outcome or insight
- Any special evaluation metrics or plots that should be produced

Ensure the set of scenarios collectively covers (but is not limited to):
- Heterogeneity: performance when data distributions differ across splits/clients
- Robustness: sensitivity to noise, outliers, or corrupted labels
- Scalability: behavior with increasing dataset size or model capacity
- Hyperparameter sensitivity: learning rate, number of iterations, regularization, etc.
- Edge cases: extreme settings that may reveal failure modes

Output format (MUST be valid JSON). Provide only JSON in your response (no extra explanation):

{{
  "scenarios": [
    {{
      "name": "scenario_name",
      "description": "brief description of what this scenario tests",
      "parameters": {{
        "--learning_rate": 0.01,
        "--num_iterations": 100,
        "--dataset_size": 1000
      }},
      "dataset": "brief description of dataset / data modifications",
      "expected_insight": "what you expect to observe",
      "metrics": ["metric1", "metric2"]
    }}
  ],
  "rationale": "Short explanation why these scenarios were chosen and how they complement each other"
}}

If existing_results are provided, you may incorporate them to suggest follow-up or targeted scenarios, but primary scenarios must still be justified from the TITLE and PSEUDOCODE.
"""
    if existing_results:
        prompt += "\n# Existing results (for reference):\n" + json.dumps(existing_results, indent=2) + "\n"

    return prompt
# ...existing code...

def parse_ai_scenario_design(ai_response):
    """
    解析 AI 返回的场景设计
    """
    try:
    # 尝试从 AI 响应中提取 JSON
        import re
        json_match = re.search(r'{.*}', ai_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            scenario_data = json.loads(json_str)
            return scenario_data
        else:
            # 如果没有找到 JSON，尝试解析文本格式
            return extract_scenarios_from_text(ai_response)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"❌ Failed to parse AI scenario design as JSON: {e}")
        return extract_scenarios_from_text(ai_response)

def extract_scenarios_from_text(text):
    """
    从文本中提取场景信息（备选方案）
    """
    scenarios = []
    lines = text.split('\n')
    current_scenario = None

    for line in lines:
        line = line.strip()
        
        if line.startswith('Scenario:') or line.startswith('Scenario Name:'):
            if current_scenario:
                scenarios.append(current_scenario)
            current_scenario = {
                'name': line.split(':', 1)[1].strip(), 
                'parameters': {},
                'description': '',
                'expected_insight': ''
            }
        
        elif line.startswith('Description:') and current_scenario:
            current_scenario['description'] = line.split(':', 1)[1].strip()
        
        elif line.startswith('Parameters:') and current_scenario:
            param_part = line.split(':', 1)[1].strip()
            params = param_part.split()
            for param in params:
                if param.startswith('--'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        current_scenario['parameters'][key] = try_parse_value(value)
        
        elif line.startswith('Expected:') and current_scenario:
            current_scenario['expected_insight'] = line.split(':', 1)[1].strip()

    if current_scenario:
        scenarios.append(current_scenario)

    return {
        "scenarios": scenarios, 
        "rationale": "Extracted from text response - please check the formatting"
    }

def try_parse_value(value_str):
    """
    尝试解析参数值
    """
    try:
        return float(value_str)
    except ValueError:
        try:
            return int(value_str)
        except ValueError:
            return value_str

def get_default_scenarios():
    """
    备用默认场景
    """
    return {
    "scenarios": [
    {
    "name": "baseline",
    "description": "Standard baseline configuration",
    "parameters": {
    "--learning_rate": 0.01,
    "--num_iterations": 100,
    "--dataset_size": 1000
    },
    "expected_insight": "Establish baseline performance for comparison"
    },
    {
    "name": "high_learning_rate",
    "description": "Test with higher learning rate",
    "parameters": {
    "--learning_rate": 0.1,
    "--num_iterations": 100,
    "--dataset_size": 1000
    },
    "expected_insight": "Observe convergence speed and stability with high learning rate"
    },
    {
    "name": "noisy_data",
    "description": "Test robustness to noisy data",
    "parameters": {
    "--learning_rate": 0.01,
    "--num_iterations": 100,
    "--noise_level": 0.5
    },
    "expected_insight": "Evaluate algorithm robustness under noisy conditions"
    }
    ],
    "rationale": "Default scenarios for basic algorithm testing covering learning rate sensitivity and robustness"
    }


def reset_and_prime_coder(coder, algorithm_info, stage_description):
    """
    清空 AI 历史并重新注入关键上下文。
    防止上下文超长，同时确保 AI 知道当前的任务目标。
    
    Args:
        coder: Aider Coder 对象
        algorithm_info: 字典，包含 'title' 和 'pseudocode'
        stage_description: 当前阶段的描述
    
    Returns:
        str: 上下文重注提示词，可选择性地添加到下一个 prompt 前面
    """
    print(f"\n🧹 [上下文管理] 清理历史，准备进入阶段: {stage_description}")
    
    # 1. 暴力清空 Aider 的对话历史
    if hasattr(coder, 'done_messages'):
        cleared_count = len(coder.done_messages)
        coder.done_messages = []
        print(f"   ✓ 已清理 {cleared_count} 条历史对话")
    
    if hasattr(coder, 'cur_messages'):
        coder.cur_messages = []
    
    # 2. 构造"上下文重注"提示词 (Re-priming Prompt)
    # 注意：Coder 会自动读取当前文件的最新内容，所以不需要把代码贴进去
    prime_prompt = f"""I have cleared your chat history to free up context window space. 
We are currently working on the following project:

# Algorithm Title
{algorithm_info.get('title', 'Unknown')}

# Algorithm Pseudocode
{algorithm_info.get('pseudocode', 'See algorithm.tex for details')}

# Current Stage
{stage_description}

# Status
The file 'experiment.py' contains the current implementation.
The file 'notes.txt' contains the experiment logs.

Please wait for my next specific instruction for this stage.
"""
    
    print(f"   ✓ 上下文重注提示词已准备（{len(prime_prompt)} 字符）")
    
    return prime_prompt


# ============================================================================
# 辅助函数：验证 Python 代码语法
# ============================================================================

def validate_python_syntax(file_path):
    """
    验证 Python 文件的语法是否正确
    
    参数：
      file_path: Python 文件路径
    
    返回：
      (is_valid, error_message): 是否有效和错误信息
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 尝试编译代码
        compile(code, file_path, 'exec')
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax error in {file_path}:\n"
        error_msg += f"  Line {e.lineno}: {e.msg}\n"
        error_msg += f"  {e.text}"
        return False, error_msg
    except Exception as e:
        return False, f"Error validating {file_path}: {str(e)}"


def read_pseudocode_from_tex(folder_name=None, tex_path=None):
    """
    从指定路径或 folder_name/algorithm.tex 读取伪代码内容，找不到则返回 None

    优先使用 tex_path（可以是相对或绝对路径）。如果未提供 tex_path，则在
    folder_name/algorithm.tex 中查找。
    """
    # 如果直接给出 tex_path，则优先使用
    if tex_path:
        candidate = tex_path if osp.isabs(tex_path) else osp.abspath(tex_path)
        if osp.exists(candidate):
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content if content.strip() else None
            except Exception as e:
                print(f"⚠️ 读取 {candidate} 时出错: {e}")
                return None
        else:
            return None

    # 否则从 folder_name 中查找 algorithm.tex
    if folder_name:
        tex_path_local = osp.join(folder_name, "algorithm.tex")
        if osp.exists(tex_path_local):
            try:
                with open(tex_path_local, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content if content.strip() else None
            except Exception as e:
                print(f"⚠️ 读取 {tex_path_local} 时出错: {e}")
                return None

    return None

# ============================================================================
# 辅助函数：测试运行 experiment.py
# ============================================================================

def test_run_experiment(folder_name, timeout=300):
    """
    测试运行 experiment.py，验证其是否能正常执行
    
    工作流程：
      1. 执行命令：python experiment.py --out_dir=test_run
      2. 检查是否生成 test_run/final_info.json
      3. 验证 JSON 格式是否正确
      4. 清理测试目录
    
    参数：
      folder_name: 实验文件夹路径
      timeout: 超时时间（秒），默认 5 分钟
    
    返回：
      (success, error_message): 是否成功和错误信息
    """
    cwd = osp.abspath(folder_name)
    test_dir = osp.join(cwd, "test_run")
    
    # 清理之前的测试目录
    if osp.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # 执行测试命令
    command = ["python", "experiment.py", "--out_dir=test_run"]
    
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=timeout
        )
        
        # 检查返回码
        if result.returncode != 0:
            error_msg = f"Test run failed with return code {result.returncode}\n"
            error_msg += f"STDERR:\n{result.stderr}"
            return False, error_msg
        
        # 检查输出文件是否存在（新结构：baseline/final_info.json）
        result_file = osp.join(test_dir, "baseline", "final_info.json")
        if not osp.exists(result_file):
            error_msg = "Test run succeeded but did not generate baseline/final_info.json\n"
            error_msg += f"Expected file: {result_file}\n"
            if osp.exists(test_dir):
                contents = os.listdir(test_dir)
                error_msg += f"test_run/ contents: {contents}\n"
                
                # 检查是否有 baseline 子目录
                baseline_dir = osp.join(test_dir, "baseline")
                if not osp.exists(baseline_dir):
                    error_msg += "\n⚠️ CRITICAL ERROR: baseline/ subdirectory was not created!\n"
                    error_msg += "\nThe experiment.py file MUST save results to:\n"
                    error_msg += "  {{out_dir}}/baseline/final_info.json\n"
                    error_msg += "\nImplementation:\n"
                    error_msg += "  baseline_dir = os.path.join(args.out_dir, 'baseline')\n"
                    error_msg += "  os.makedirs(baseline_dir, exist_ok=True)\n"
                    error_msg += "  with open(os.path.join(baseline_dir, 'final_info.json'), 'w') as f:\n"
                    error_msg += "      json.dump(results, f)\n"
                elif osp.exists(baseline_dir):
                    baseline_contents = os.listdir(baseline_dir)
                    error_msg += f"\nbaseline/ exists but final_info.json not found.\n"
                    error_msg += f"baseline/ contents: {baseline_contents}\n"
            else:
                error_msg += "test_run/ directory was not created"
            return False, error_msg
        
        # 验证 JSON 格式
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # 检查是否有正确的格式
            if not isinstance(results, dict):
                return False, "final_info.json must be a dictionary"
            
            # 检查是否至少有一个指标
            if len(results) == 0:
                return False, "final_info.json is empty"
            
            # 检查每个指标是否有 means 和 stds
            for metric_name, metric_data in results.items():
                if not isinstance(metric_data, dict):
                    return False, f"Metric '{metric_name}' must be a dictionary"
                if "means" not in metric_data:
                    return False, f"Metric '{metric_name}' missing 'means' field"
                if "stds" not in metric_data:
                    return False, f"Metric '{metric_name}' missing 'stds' field"
                
                # 检查 means 是否是列表且非空
                means = metric_data["means"]
                if not isinstance(means, list) or len(means) == 0:
                    return False, f"Metric '{metric_name}': 'means' must be a non-empty list"
            
            # ================================================================
            # 新增验证：检查结果的合理性
            # ================================================================
            validation_errors = []
            
            # 验证 1：检查是否所有值都相同（可能是常数，表示没有学习）
            for metric_name, metric_data in results.items():
                means = metric_data["means"]
                if len(means) > 5:
                    # 检查前后差异
                    initial_values = means[:min(5, len(means)//4)]
                    final_values = means[-min(5, len(means)//4):]
                    
                    initial_mean = sum(initial_values) / len(initial_values)
                    final_mean = sum(final_values) / len(final_values)
                    
                    # 如果初始值和最终值几乎相同（变化小于1%），可能有问题
                    if abs(initial_mean) > 1e-6:  # 避免除零
                        change_ratio = abs(final_mean - initial_mean) / abs(initial_mean)
                        if change_ratio < 0.01:  # 变化小于1%
                            validation_errors.append(
                                f"⚠️ Metric '{metric_name}': Values barely changed "
                                f"(initial: {initial_mean:.4f}, final: {final_mean:.4f}). "
                                f"This may indicate the algorithm is not learning properly."
                            )
            
            # 验证 2：扫描代码中的常见问题模式
            exp_file = osp.join(cwd, "experiment.py")
            if osp.exists(exp_file):
                with open(exp_file, 'r') as f:
                    code_content = f.read()
                
                # 检查是否使用了 placeholder 或随机值
                problematic_patterns = [
                    ('np.random.randn', 'using random values for gradients/data'),
                    ('np.random.rand', 'using random values for gradients/data'),
                    ('TODO', 'unfinished implementation (TODO comments)'),
                    ('Placeholder', 'placeholder code that needs implementation'),
                    ('pass  # implement', 'empty implementation'),
                ]
                
                for pattern, description in problematic_patterns:
                    if pattern in code_content:
                        # 检查是否在注释中（允许在注释中出现）
                        lines_with_pattern = [line for line in code_content.split('\n') 
                                             if pattern in line and not line.strip().startswith('#')]
                        if lines_with_pattern:
                            validation_errors.append(
                                f"⚠️ Found '{pattern}' in code ({description}). "
                                f"Example: {lines_with_pattern[0][:80]}..."
                            )
            
            # 如果有验证错误，返回警告
            if validation_errors:
                error_msg = "Test run completed but found potential issues:\n\n"
                error_msg += "\n".join(validation_errors)
                error_msg += "\n\n"
                error_msg += "Common causes:\n"
                error_msg += "1. Using random values instead of computing from real data\n"
                error_msg += "2. Not properly implementing the training loop\n"
                error_msg += "3. Incorrect gradient computation or model updates\n"
                error_msg += "4. Model not learning from data\n\n"
                error_msg += "Please review and fix these issues. The code should:\n"
                error_msg += "- Compute gradients from actual model predictions and data\n"
                error_msg += "- Update model parameters based on gradients\n"
                error_msg += "- Show learning progress (metrics should change over iterations)\n"
                return False, error_msg
            
            # 测试成功
            return True, ""
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON in final_info.json: {str(e)}"
        except Exception as e:
            return False, f"Error reading final_info.json: {str(e)}"
            
    except TimeoutExpired:
        return False, f"Test run timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Error during test run: {str(e)}"
    finally:
        # 清理测试目录
        if osp.exists(test_dir):
            shutil.rmtree(test_dir)


# ============================================================================
# 阶段 0: 代码生成与验证
# ============================================================================

def generate_code_from_pseudocode(idea, folder_name, coder, algorithm_tex_path=None):
    """
    根据 algorithm.tex 中的伪代码生成初始实验代码并验证其可运行性

    注意：本函数不使用 idea["Pseudocode"]，而是使用传入的 algorithm_tex_path
    或者在 folder_name/algorithm.tex 中查找。
    """
    
    print("\n" + "="*80)
    print("🔧 阶段 0: 代码生成与验证（使用 algorithm.tex 作为伪代码）")
    print("="*80 + "\n")
    
    # 从指定路径或 folder 中读取伪代码
    pseudocode = read_pseudocode_from_tex(folder_name=folder_name, tex_path=algorithm_tex_path)
    if not pseudocode:
        print("⚠️ 警告: 未找到 algorithm.tex（既未提供 --algorithm-tex，也未在文件夹中找到），跳过代码生成阶段")
        return True
    
    title = idea.get("Title", "Unknown")
    experiment_desc = idea.get("Experiment", "")
    
    # 生成初始提示词
    initial_prompt = code_generation_prompt.format(
        title=title,
        pseudocode=pseudocode,
        experiment_description=experiment_desc
    )
    
    print("🤖 AI 正在根据 algorithm.tex 中的伪代码生成初始代码...")
    print(f"   伪代码长度: {len(pseudocode)} 字符")
    print()
    
    # 迭代生成和验证
    for iteration in range(MAX_CODE_GEN_ITERS):
        print(f"\n--- 代码生成迭代 {iteration + 1}/{MAX_CODE_GEN_ITERS} ---\n")
        
        # ====================================================================
        # Step 0.1: AI 生成/修改代码
        # ====================================================================
        if iteration == 0:
            # 首次生成 - 直接要求 AI 生成完整文件
            print("🤖 AI 正在生成 experiment.py 和 plot.py...")
            print("   提示：第一次生成，AI 将创建完整的代码文件")
            coder_out = coder.run(initial_prompt)
        else:
            # 修复代码 - 使用 diff 模式进行增量修改
            print("🤖 AI 正在根据错误反馈修复代码...")
            coder_out = coder.run(next_prompt)
        
        print(coder_out)
        
        # ====================================================================
        # Step 0.2: 验证 experiment.py 语法
        # ====================================================================
        print("\n📝 验证 experiment.py 语法...")
        exp_file = osp.join(folder_name, "experiment.py")
        
        if not osp.exists(exp_file):
            print("❌ experiment.py 未生成")
            next_prompt = """experiment.py file was not created. Please CREATE the experiment.py file now.

Remember the critical requirements:
1. Must accept --out_dir command-line argument
2. Must save results to {out_dir}/baseline/final_info.json
3. JSON format: {"metric": {"means": [...], "stds": [...]}}

NOTE: This is initial file creation, so full file content is needed.
"""
            continue
        
        syntax_valid, syntax_error = validate_python_syntax(exp_file)
        
        if not syntax_valid:
            print(f"❌ 语法错误:\n{syntax_error}")
            next_prompt = f"""The experiment.py file has syntax errors:

{syntax_error}

Please fix these syntax errors using TARGETED EDITS ONLY.

**CRITICAL: DO NOT output the entire file!**
- Identify the exact lines with errors
- Make minimal, surgical fixes to those specific lines
- Use the edit/replace functionality to modify only the problematic code sections

Focus on fixing ONLY the syntax errors mentioned above, nothing else.
"""
            continue
        
        print("✅ experiment.py 语法正确")
        
        # ====================================================================
        # Step 0.3: 验证 plot.py 语法（可选）
        # ====================================================================
        print("\n📝 验证 plot.py 语法...")
        plot_file = osp.join(folder_name, "plot.py")
        
        if osp.exists(plot_file):
            syntax_valid, syntax_error = validate_python_syntax(plot_file)
            if not syntax_valid:
                print(f"⚠️ plot.py 有语法错误:\n{syntax_error}")
                print("   (将在后续阶段修复)")
            else:
                print("✅ plot.py 语法正确")
        else:
            print("⚠️ plot.py 未生成（将在后续阶段生成）")
        
        # ====================================================================
        # Step 0.4: 测试运行 experiment.py
        # ====================================================================
        print("\n⚙️ 测试运行 experiment.py...")
        print("   执行: python experiment.py --out_dir=test_run")
        
        test_success, test_error = test_run_experiment(folder_name, timeout=300)
        
        if not test_success:
            print(f"❌ 测试运行失败:\n{test_error}")
            
            # 截断过长的错误信息
            if len(test_error) > MAX_STDERR_OUTPUT:
                test_error = "..." + test_error[-MAX_STDERR_OUTPUT:]
            
            next_prompt = f"""The experiment.py file has runtime errors:

{test_error}

Please fix these errors using MINIMAL, TARGETED EDITS.

**CRITICAL: DO NOT rewrite or output the entire file!**

Common issues to check and fix:
1. Missing import statements → Add only the missing imports at the top
2. Undefined variables or functions → Fix the specific line/function
3. Incorrect data types or shapes → Adjust the problematic operation
4. File I/O errors → Fix the file handling code
5. Missing --out_dir argument handling → Add argument parsing if missing
6. Incorrect final_info.json format → Fix the save logic

**Instructions:**
- Analyze the error traceback to identify the EXACT problematic lines
- Make SURGICAL fixes to only those specific locations
- Use search/replace or edit commands for targeted modifications
- DO NOT output code that's already working correctly
"""
            continue

        print("✅ 测试运行成功！")
        print("✅ final_info.json 格式正确")
        
        # ====================================================================
        # Step 0.5: 测试运行 plot.py
        # ====================================================================
        print("\n📊 测试 plot.py 的可视化功能...")
        plot_test_success, plot_test_error = test_plot_script(folder_name)

        if not plot_test_success:
            print(f"❌ plot.py 测试失败:\n{plot_test_error}")
            next_prompt = f"""The plot.py file has issues:

        {plot_test_error}

        Please fix plot.py using MINIMAL, TARGETED EDITS.

        **CRITICAL: DO NOT rewrite the entire file!**

        Required fixes:
        1. Accept --out_dir command-line argument (add if missing)
        2. Use matplotlib for visualizations (fix import/usage if broken)
        3. Save plots to the specified output directory as PNG files
        4. Fix any runtime errors

        **Instructions:**
        - Identify the EXACT issue from the error message
        - Make SURGICAL fixes to only the problematic code
        - Use search/replace for targeted modifications
        - DO NOT output working code sections
        """
            continue
        else:
            print(f"✅ plot.py 测试成功: {plot_test_error}")
        
        # ====================================================================
        # Step 0.6: 代码逻辑验证（新增步骤）
        # ====================================================================
        print("\n🔍 验证代码逻辑是否符合伪代码...")
        
        # 读取当前生成的代码
        with open(exp_file, 'r', encoding='utf-8') as f:
            current_code = f.read()
        
        # 生成逻辑验证提示词
        logic_prompt = logic_validation_prompt.format(
            title=title,
            pseudocode=pseudocode,
            current_code=current_code
        )
        
        print("🤖 AI 正在验证代码逻辑...")
        logic_response = coder.run(logic_prompt)
        print(logic_response)
        
        # 检查 AI 是否确认代码逻辑正确
        if "LOGIC_VALIDATION_PASSED" in logic_response:
            print("✅ 代码逻辑验证通过！")
            print()
            print("="*80)
            print("🎉 代码生成阶段完成！")
            print("="*80)
            print()
            return True
        else:
            print("❌ 代码逻辑验证失败，AI 发现与伪代码不符之处")
            print("🔄 继续迭代修复...")
            
            # 生成下一轮修复的提示词
            next_prompt = f"""The code logic validation failed. The AI identified discrepancies between the implementation and the pseudocode.

# Algorithm Pseudocode
{pseudocode}

Key issues identified:
- The implementation does not correctly follow the pseudocode logic
- Some algorithmic steps may be missing or incorrectly implemented
- Mathematical formulas or procedures may not match

**CRITICAL: Make TARGETED FIXES ONLY - DO NOT rewrite the entire file!**

**Instructions:**
1. Identify the SPECIFIC algorithmic components that don't match the pseudocode
2. Make SURGICAL edits to fix only those components
3. Use search/replace to modify specific functions/sections
4. Preserve all correctly implemented parts

Focus on the algorithmic logic discrepancies, not cosmetic changes.
"""
            continue
    
    # ========================================================================
    # 达到最大迭代次数仍未成功
    # ========================================================================
    print("\n" + "="*80)
    print(f"❌ 代码生成失败：达到最大迭代次数 ({MAX_CODE_GEN_ITERS})")
    print("="*80)
    print()
    print("建议:")
    print("  1. 检查 algorithm.tex 中的伪代码是否清晰完整")
    print("  2. 检查实验描述是否提供了足够的上下文")
    print("  3. 手动检查生成的 experiment.py 并修复")
    print()
    
    return False

# ============================================================================
# 辅助函数：测试运行plot.py
# ============================================================================
def test_plot_script(folder_name):
    """
    测试 plot.py 是否能正确处理 --out_dir 参数
    """
    cwd = osp.abspath(folder_name)
    test_plots_dir = osp.join(cwd, "test_plots")
    
    # 清理之前的测试目录
    if osp.exists(test_plots_dir):
        shutil.rmtree(test_plots_dir)
    
    # 执行测试命令
    command = ["python", "plot.py", f"--out_dir={test_plots_dir}"]
    
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=120  # 2分钟超时
        )
        
        # 检查返回码
        if result.returncode != 0:
            error_msg = f"Plot test failed with return code {result.returncode}\n"
            error_msg += f"STDERR:\n{result.stderr}"
            return False, error_msg
        
        # 检查是否创建了输出目录和图表文件
        if not osp.exists(test_plots_dir):
            return False, "Plot test failed: output directory was not created"
        
        plot_files = [f for f in os.listdir(test_plots_dir) if f.endswith('.png')]
        if not plot_files:
            return False, "Plot test failed: no PNG files were generated"
        
        return True, f"Successfully generated plot files: {plot_files}"
        
    except TimeoutExpired:
        return False, f"Plot test timed out after 120 seconds"
    except Exception as e:
        return False, f"Error during plot test: {str(e)}"
    finally:
        # 清理测试目录
        if osp.exists(test_plots_dir):
            shutil.rmtree(test_plots_dir)


# ============================================================================
# 辅助函数：执行单次实验
# ============================================================================

def run_experiment(folder_name, run_num, timeout=7200):
    """
    执行单次实验运行
    
    工作流程：
      1. 保存当前 experiment.py 的快照（用于追溯）
      2. 执行命令：python experiment.py --out_dir=run_{run_num}
      3. 检查执行结果：
         - 成功：读取 final_info.json，返回结果数据
         - 失败：清理失败的运行，返回错误信息
      4. 生成反馈提示词给 AI
    
    参数：
      folder_name: 实验文件夹路径
      run_num: 运行编号（1, 2, 3, ...）
      timeout: 超时时间（秒），默认 2 小时
    
    返回：
      (return_code, next_prompt): 返回码和下一步的 AI 提示词
    """
    cwd = osp.abspath(folder_name)
    
    # ========================================================================
    # Step 1: 构造并执行命令
    # ========================================================================
    # 固定格式：python experiment.py --out_dir=run_1
    # experiment.py 必须支持 --out_dir 参数，并会创建 run_1/ 目录
    command = [
        "python",
        "experiment.py",
        f"--out_dir=run_{run_num}",
    ]
    
    try:
        # 执行命令，捕获标准错误输出
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        # 打印错误输出（如果有）
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # ====================================================================
        # Step 3: 处理执行结果
        # ====================================================================
        
        if result.returncode != 0:
            # ----------------------------------------------------------------
            # 情况 A: 执行失败
            # ----------------------------------------------------------------
            print(f"Run {run_num} failed with return code {result.returncode}")
            
            # 清理失败的运行目录（避免脏数据）
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            
            print(f"Run failed with the following error {result.stderr}")
            
            # 截断过长的错误信息
            stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            
            # 生成错误反馈提示词（AI 会根据错误修复代码）
            next_prompt = f"Run failed with the following error {stderr_output}"
            
        else:
            # ----------------------------------------------------------------
            # 情况 B: 执行成功
            # ----------------------------------------------------------------
            
            # ================================================================
            # Step 3.1: 保存代码快照到 run_N/ 目录
            # ================================================================
            # 将当前的 experiment.py 和 plot.py 复制到 run_N/ 目录
            # 作用：追溯每次运行使用的代码版本（因为 AI 会不断修改这些文件）
            run_dir = osp.join(cwd, f"run_{run_num}")
            
            shutil.copy(
                osp.join(cwd, "experiment.py"),
                osp.join(run_dir, "experiment.py"),
            )
            
            # 如果 plot.py 存在，也复制一份
            plot_file = osp.join(cwd, "plot.py")
            if osp.exists(plot_file):
                shutil.copy(plot_file, osp.join(run_dir, "plot.py"))
            
            # ================================================================
            # Step 3.2: 读取实验结果
            # ================================================================
            # 读取实验结果文件 run_{run_num}/baseline/final_info.json
            # 格式要求：{"metric": {"means": [...], "stds": [...]}}
            result_file = osp.join(run_dir, "baseline", "final_info.json")
            if not osp.exists(result_file):
                # 兼容性：尝试旧结构
                result_file = osp.join(run_dir, "final_info.json")
            
            with open(result_file, "r") as f:
                results = json.load(f)
            
            # 提取 means 值（主要结果）
            results = {k: v["means"] for k, v in results.items()}

            # 生成成功反馈提示词（包含结果数据和下一步指示）
            next_prompt = f"""Run {run_num} completed. Here are the results:
{results}

Decide if you need to re-plan your experiments given the result (you often will not need to).

Someone else will be using `notes.txt` to perform a writeup on this in the future.
Please include *all* relevant information for the writeup on Run {run_num}, including an experiment description and the run number. Be as verbose as necessary.

Then, implement the next thing on your list.
We will then run the command `python experiment.py --out_dir=run_{run_num + 1}'.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
If you are finished with experiments, respond with 'ALL_COMPLETED'."""
        
        return result.returncode, next_prompt
        
    except TimeoutExpired:
        # ====================================================================
        # 情况 C: 执行超时
        # ====================================================================
        print(f"Run {run_num} timed out after {timeout} seconds")
        
        # 清理超时的运行目录
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
        
        # 生成超时反馈提示词（AI 会优化代码以减少运行时间）
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt


# ============================================================================
# 辅助函数：执行可视化生成
# ============================================================================


## 2. 修改可视化执行函数


# ============================================================================
# 辅助函数：执行可视化生成
# ============================================================================

def run_plotting(folder_name, timeout=600):
    """
    执行可视化脚本生成图表
    
    工作流程：
      1. 执行命令：python plot.py --out_dir=plots
      2. plot.py 应该读取 run_1/, run_2/ 等目录的结果
      3. 生成 PNG 图表文件到 plots/ 目录
    
    参数：
      folder_name: 实验文件夹路径
      timeout: 超时时间（秒），默认 10 分钟
    
    返回：
      (return_code, next_prompt): 返回码和反馈提示词
    """
    cwd = osp.abspath(folder_name)
    
    # 创建 plots 目录
    plots_dir = osp.join(cwd, "plots")
    if osp.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir)
    
    # 检查 plot.py 是否存在
    plot_file = osp.join(cwd, "plot.py")
    if not osp.exists(plot_file):
        return False, "plot.py 文件不存在"
    
    # 执行绘图命令
    command = ["python", "plot.py", f"--out_dir={plots_dir}"]
    
    try:
        print("🎨 生成多场景可视化结果...")
        result = subprocess.run(
            command, 
            cwd=cwd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=600  # 10分钟超时
        )
        
        if result.returncode == 0:
            # 检查生成的图表文件
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
            if plot_files:
                print(f"✅ 成功生成 {len(plot_files)} 个图表文件:")
                for plot_file in plot_files:
                    print(f"   📊 {plot_file}")
                return True, f"生成 {len(plot_files)} 个可视化图表"
            else:
                return False, "绘图完成但未生成图表文件"
        else:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            return False, f"绘图失败: {error_msg}"
            
    except TimeoutExpired:
        return False, "绘图超时"
    except Exception as e:
        return False, f"绘图异常: {str(e)}"


# ============================================================================
# 主函数：执行完整的实验流程
# ============================================================================

def perform_experiments(idea, folder_name, coder, baseline_results, algorithm_tex_path=None) -> bool:
    """
    执行完整的 AI 驱动实验流程

    新增参数:
      algorithm_tex_path: 可选，指向 algorithm.tex 的路径（相对或绝对）。如果提供，
                          所有伪代码读取将优先使用该路径。
    """
    # ========================================================================
    # 阶段 0: 代码生成与验证（使用 algorithm.tex）
    # ========================================================================
    tex_pseudocode = read_pseudocode_from_tex(folder_name=folder_name, tex_path=algorithm_tex_path)
    
    # 准备算法信息对象（用于上下文重注）
    algo_info = {
        "title": idea.get("Title", "Algorithm Experiment"),
        "pseudocode": tex_pseudocode if tex_pseudocode else "Refer to algorithm.tex or idea JSON"
    }
    
    if tex_pseudocode:
        print("\n" + "="*80)
        print("📋 检测到 algorithm.tex，启动代码生成阶段（不使用 idea['Pseudocode']）")
        print("="*80 + "\n")
        
        success = generate_code_from_pseudocode(idea, folder_name, coder, algorithm_tex_path=algorithm_tex_path)
        
        if not success:
            print("\n❌ 代码生成阶段失败，无法继续后续实验")
            return False
        
        print("✅ 代码生成阶段完成，进入实验迭代阶段\n")
        
        # [上下文管理] 代码生成完成后，清理历史，准备进入场景设计阶段
        # 此时 experiment.py 已经生成好了，AI 只要读文件就行，不需要知道生成的曲折过程
        # reset_and_prime_coder(coder, algo_info, "Phase 0.5: AI Scenario Design")  # 临时注释掉用于测试
    else:
        print("\n" + "="*80)
        print("ℹ️  未检测到 algorithm.tex，跳过代码生成阶段（不使用 idea['Pseudocode']）")
        print("="*80 + "\n")

    # ========================================================================
    # 阶段 0.5: AI 自主场景设计（新增）
    # ========================================================================
    scenario_design = design_experiment_scenarios(idea, folder_name, coder, baseline_results, algorithm_tex_path=algorithm_tex_path)
    ai_scenarios = scenario_design.get("scenarios", [])
    
    if not ai_scenarios:
        print("❌ 未能获得有效的 AI 设计场景，无法继续")
        return False
    
    # [上下文管理] 场景设计完成后，清理历史，准备进入正式执行阶段
    # 此时我们有了 scenarios 列表，不需要 AI 记得它是怎么想出这些场景的
    # reset_and_prime_coder(coder, algo_info, "Phase 1: Experiment Execution")  # 临时注释掉用于测试

    # ========================================================================
    # 阶段 1: 迭代实验循环（修改为支持 AI 设计场景）
    # ========================================================================
    print("\n" + "="*80)
    print("🔬 阶段 1: 迭代实验循环（集成 AI 设计场景）")
    print("="*80 + "\n")
    
    # 执行 AI 设计的场景（每个场景失败时自动迭代修复）
    # 传入 algo_info 以便在场景切换时使用
    scenario_results = execute_ai_designed_scenarios(folder_name, ai_scenarios, coder, algo_info=algo_info)
    
    # 分析执行结果
    successful_scenarios = [s for s in scenario_results if s["status"] == "success"]
    failed_scenarios = [s for s in scenario_results if s["status"] != "success"]
    
    print(f"\n📊 AI 设计场景执行总结:")
    print(f"   ✅ 成功: {len(successful_scenarios)}/{len(ai_scenarios)}")
    print(f"   ❌ 失败: {len(failed_scenarios)}/{len(ai_scenarios)}")
    
    if successful_scenarios:
        print("\n✅ 成功执行的 AI 设计场景:")
        for scenario in successful_scenarios:
            print(f"   • {scenario['name']}")
    
    if failed_scenarios:
        print("\n❌ 失败的 AI 设计场景:")
        for scenario in failed_scenarios:
            print(f"   • {scenario['name']}: {scenario.get('error', 'Unknown error')}")
    
    # 保存场景执行结果
    results_file = osp.join(folder_name, "scenario_execution_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "scenario_design": scenario_design,
            "execution_results": scenario_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 场景执行结果已保存到: {results_file}")
    
    # ========================================================================
    # 原有的迭代实验循环（保持不变，作为备选）
    # ========================================================================
    if not successful_scenarios:
        print("\n⚠️  AI 设计场景全部失败，回退到原有实验循环")
        return perform_original_experiment_loop(idea, folder_name, coder, baseline_results)
    
    # [上下文管理] 场景执行完成后，清理历史，准备进入可视化阶段
    # 此时已有所有实验结果，不需要 AI 记得实验执行的曲折过程
    # reset_and_prime_coder(coder, algo_info, "Phase 2: Visualization Generation")  # 临时注释掉用于测试
    
    # ========================================================================
    # 阶段 2: 可视化生成（增强版，支持多场景）
    # ========================================================================
    print("\n" + "="*80)
    print("📊 阶段 2: 多场景可视化生成")
    print("="*80 + "\n")
    
    plot_success, plot_message = generate_comprehensive_visualization(
        folder_name, 
        coder=coder, 
        scenario_results=scenario_results
    )
    
    if plot_success:
        print(f"✅ {plot_message}")
    else:
        print(f"❌ {plot_message}")
        
        # 让 AI 修复可视化代码
        print("🤖 AI 正在修复可视化代码...")
        fix_plot_prompt = """
The multi-scenario plotting failed. Please fix plot.py using MINIMAL, TARGETED EDITS.

**CRITICAL: DO NOT rewrite the entire file!**

Key requirements to fix:
1. Read results from all successful scenario directories
2. Generate comprehensive comparison plots showing different scenarios
3. Use clear labels and legends to distinguish different scenarios
4. Create multiple plot types to show different aspects of the results:
   - Convergence curves for each metric across scenarios
   - Scenario comparison bar charts
   - Performance summary plots

**Instructions:**
- Identify the specific issue causing the failure
- Make SURGICAL fixes to only the problematic section
- Use search/replace for targeted modifications
- DO NOT output code that's working correctly

Please provide TARGETED fixes to plot.py.
"""
        coder.run(fix_plot_prompt)
        
        # 重新尝试绘图
        print("🔄 重新尝试生成可视化...")
        plot_success, plot_message = generate_comprehensive_visualization(
            folder_name, 
            coder=coder, 
            scenario_results=scenario_results
        )
        
        if plot_success:
            print(f"✅ 修复成功: {plot_message}")
        else:
            print(f"❌ 修复后仍然失败: {plot_message}")
    
    # ========================================================================
    # 阶段 3: 文档更新（增强版，包含 AI 场景设计信息）
    # ========================================================================
    print("\n" + "="*80)
    print("📝 阶段 3: 实验文档更新（包含 AI 场景设计）")
    print("="*80 + "\n")
    
    documentation_prompt = f"""
Please update notes.txt with a comprehensive summary of the AI-driven experimental process, including:

# AI-Designed Experimental Scenarios
{json.dumps(scenario_design, indent=2)}

# Execution Results
Successful scenarios: {len(successful_scenarios)}/{len(ai_scenarios)}
Failed scenarios: {len(failed_scenarios)}/{len(ai_scenarios)}

Please include the following sections in your documentation:

1. **Scenario Design Rationale**: Explain why the AI chose these specific scenarios and how they test different aspects of the algorithm
2. **Results Analysis**: Detailed analysis of results from each successful scenario, comparing them with expectations
3. **Algorithm Insights**: What we learned about the algorithm's behavior across different conditions
4. **Visualization Explanation**: Describe what each generated plot shows and the insights it provides
5. **Conclusions and Recommendations**: Overall conclusions and suggestions for further investigation based on the multi-scenario analysis

Make sure to reference specific scenarios and their results in your analysis, and connect the findings back to the original algorithm design.
"""
    
    print("🤖 AI 正在更新实验文档...")
    coder.run(documentation_prompt)
    print("✅ 文档更新完成")
    
    # ========================================================================
    # 实验流程完成
    # ========================================================================
    print("\n" + "="*80)
    print("🎉 AI 驱动的多场景实验流程完成！")
    print("="*80 + "\n")
    
    return len(successful_scenarios) > 0

def execute_ai_designed_scenarios(folder_name, scenarios, coder, max_retries=MAX_ITERS, algo_info=None):
    """
    执行 AI 设计的场景，为每个场景定制代码实现
    
    核心思想：公共代码基础 + 场景特定定制
    
    参数:
        folder_name: 实验文件夹路径
        scenarios: AI 设计的场景列表
        coder: Aider Coder 对象（用于定制每个场景的代码）
        max_retries: 每个场景失败时的最大重试次数（默认使用 MAX_ITERS）
        algo_info: 算法信息字典（包含 title 和 pseudocode），用于上下文重注
    
    工作流程：
        1. 场景1: 使用初始代码运行
        2. 场景2: AI根据场景2需求调整代码（如添加Non-IID逻辑）→ 运行
        3. 场景3: AI根据场景3需求调整代码（如添加标签噪声）→ 运行
        ...
        每个场景都可以有自己定制的实现，而非仅通过参数区分
    """
    cwd = osp.abspath(folder_name)
    results = []
    
    # 准备算法信息用于上下文重注
    if algo_info is None:
        # 尝试读取算法伪代码
        tex_content = read_pseudocode_from_tex(folder_name=folder_name)
        algo_info = {
            "title": "Algorithm Experiment",
            "pseudocode": tex_content if tex_content else "Refer to algorithm.tex"
        }
    
    print(f"🚀 开始执行 {len(scenarios)} 个 AI 设计的场景（失败时自动迭代修复）")
    
    for i, scenario in enumerate(scenarios, 1):
        scenario_name = scenario.get('name', f'scenario_{i}').replace(' ', '_').lower()
        parameters = scenario.get('parameters', {})
        
        print(f"\n{'='*80}")
        print(f"场景 {i}/{len(scenarios)}: {scenario_name}")
        print(f"{'='*80}")
        print(f"描述: {scenario.get('description', 'No description')}")
        print(f"数据集: {scenario.get('dataset', 'No dataset info')}")
        print(f"参数: {json.dumps(parameters, indent=2)}")
        print()
        
        # ====================================================================
        # 场景准备：为当前场景调整代码（场景特定定制）
        # ====================================================================
        if i > 1:  # 从第二个场景开始，需要根据场景需求调整代码
            print(f"🔧 为场景 {i} 调整代码实现...")
            
            # [上下文管理] 在场景切换时清理历史并重注上下文
            # context_prefix = reset_and_prime_coder(  # 临时注释掉用于测试
            #     coder, 
            #     algo_info, 
            #     stage_description=f"Scenario {i}/{len(scenarios)}: {scenario_name}"
            # )
            
            # 将上下文重注提示词添加到 preparation_prompt 前面
            # preparation_prompt = context_prefix + f"""  # 临时注释掉用于测试
            preparation_prompt = f"""

Now prepare the code for the next scenario: {scenario_name}

# Scenario Details
Description: {scenario.get('description', '')}
Dataset: {scenario.get('dataset', '')}
Expected Insight: {scenario.get('expected_insight', '')}
Parameters: {json.dumps(parameters, indent=2)}

# Your Task
Modify experiment.py to properly implement this scenario's requirements:

## Critical Implementation Requirements

1. **Data Partition Strategy**:
   - If the scenario mentions "IID" or "uniform": Use simple random partition
   - If the scenario mentions "Non-IID", "label skew", "heterogeneous", or "each client has 2-3 classes":
     ```python
     def partition_non_iid(X, y, n_clients, classes_per_client=2, seed=None):
         # Assign specific classes to each client
         # Each client predominantly holds samples from classes_per_client classes
     ```
   
2. **Label Noise Injection**:
   - If the scenario mentions "label noise", "label flip", "corrupted labels", or "X% noise":
     ```python
     def add_label_noise(y, noise_rate=0.2, n_classes=10, seed=None):
         # Randomly flip noise_rate proportion of labels
         # Return corrupted labels
     ```
   
3. **Data Generation/Loading**:
   - Adjust data generation parameters if needed (samples, features, classes)
   - Ensure the dataset matches the scenario description

4. **Algorithm Selection**:
   - Ensure the algorithm parameter (--algo or --algorithm) is properly handled
   - Support both FedAvg and SFedAvg if needed

## Important Notes
- Keep the command-line interface consistent (--out_dir is required)
- Maintain the output format (final_info.json with means and stds)
- Add any new parameters needed for this scenario
- The code should be FULLY FUNCTIONAL for this specific scenario

**CRITICAL: Use MINIMAL, TARGETED EDITS - DO NOT rewrite the entire file!**

**Modification Strategy:**
1. Identify which specific functions/sections need changes for this scenario
2. Add new functions if needed (e.g., partition_non_iid, add_label_noise)
3. Modify only the data generation/loading section to call these new functions
4. Add new command-line arguments if required
5. DO NOT touch working code sections (model, training loop, etc.)

**Use search/replace or focused edits - preserve all existing working code!**

Please modify experiment.py now with TARGETED changes for this scenario."""

            print(f"🤖 AI 正在调整代码以匹配场景需求...")
            coder_out = coder.run(preparation_prompt)
            print(coder_out)
            print()
            print(f"✅ 代码调整完成，准备运行场景 {i}")
            print()
        
        # 构建命令参数列表
        args_list = []
        for key, value in parameters.items():
            args_list.append(f"{key}={value}")
        
        # ====================================================================
        # 迭代执行场景（失败时修复）
        # ====================================================================
        success = False
        final_error = None
        
        for attempt in range(max_retries):
            print(f"尝试 {attempt + 1}/{max_retries}: 执行场景 '{scenario_name}'")
            
            # 执行场景
            success, result_info = execute_single_scenario(folder_name, scenario_name, args_list)
            
            if success:
                print(f"✅ 场景 '{scenario_name}' 执行成功！")
                
                # 记录场景实现摘要
                print(f"📝 为场景 '{scenario_name}' 生成实现摘要...")
                summary_prompt = f"""Please document what special implementations were made for scenario '{scenario_name}'.

Scenario description: {scenario.get('description', '')}
Dataset: {scenario.get('dataset', '')}

Please append to notes.txt a brief summary of:
1. What data partition strategy was used (IID, Non-IID, etc.)
2. Whether label noise was added and at what rate
3. Any special data processing or algorithm modifications
4. The key implementation details that make this scenario different from others

Keep it concise (3-5 lines) and factual."""
                
                coder.run(summary_prompt)
                print(f"✅ 实现摘要已添加到 notes.txt")
                
                break
            else:
                print(f"❌ 场景 '{scenario_name}' 执行失败")
                final_error = result_info
                
                # 如果还有重试机会，让 AI 修复代码
                if attempt < max_retries - 1:
                    print(f"🤖 让 AI 根据错误修复代码...")
                    
                    # 截断过长的错误信息
                    error_msg = result_info
                    if len(error_msg) > MAX_STDERR_OUTPUT:
                        error_msg = "..." + error_msg[-MAX_STDERR_OUTPUT:]
                    
                    # 生成修复提示词
                    fix_prompt = f"""The scenario '{scenario_name}' failed with the following error:

{error_msg}

Scenario description:
{scenario.get('description', 'No description')}

Parameters used:
{json.dumps(parameters, indent=2)}

Expected insight:
{scenario.get('expected_insight', 'No expected insight provided')}

**CRITICAL: Fix using MINIMAL, TARGETED EDITS - DO NOT rewrite the entire file!**

Common issues to check and fix:
1. Parameter validation and handling → Fix argument parsing/validation
2. Data generation/loading → Fix the specific data preparation function
3. Model compatibility → Adjust model initialization if needed
4. Edge cases → Add boundary checks where missing
5. File I/O and result saving → Fix save operations

**Instructions:**
- Analyze the error to pinpoint the EXACT problematic function/line
- Make SURGICAL fixes to only that specific location
- Use search/replace for targeted modifications
- DO NOT modify code that's working correctly

Please make TARGETED changes to experiment.py now.
"""
                    
                    # AI 修复代码
                    coder_out = coder.run(fix_prompt)
                    print(coder_out)
                    print()
                    print(f"🔄 重试场景 '{scenario_name}'...")
                    
                    # 添加短暂延迟
                    import time
                    time.sleep(2)
                else:
                    print(f"⚠️ 场景 '{scenario_name}' 达到最大重试次数 ({max_retries})，标记为失败")
        
        # ====================================================================
        # 记录场景执行结果
        # ====================================================================
        scenario_result = {
            "name": scenario_name,
            "description": scenario.get('description', ''),
            "dataset": scenario.get('dataset', ''),
            "parameters": parameters,
            "expected_insight": scenario.get('expected_insight', ''),
            "status": "success" if success else "failed",
            "run_dir": f"run_{scenario_name}",
            "attempts": attempt + 1,  # 记录实际尝试次数
            "implementation_note": "Code was customized for this scenario's specific requirements" if i > 1 else "Uses initial generated code"
        }
        
        if not success:
            scenario_result["error"] = final_error
        
        # ====================================================================
        # 场景成功后立即进行参数调优（新增）
        # ====================================================================
        if success and ENABLE_HYPERPARAMETER_TUNING:
            try:
                print(f"\n{'='*80}")
                print(f"🎯 场景 '{scenario_name}' 执行成功，立即开始参数调优")
                print(f"{'='*80}\n")
                
                tuning_report = tune_scenario_immediately(
                    folder_name=folder_name,
                    scenario_info=scenario_result,
                    coder=coder,
                    algo_info=algo_info
                )
                
                if tuning_report:
                    scenario_result["tuning_report"] = tuning_report
                    print(f"\n✅ 场景 '{scenario_name}' 调优成功完成")
                else:
                    print(f"\n⚠️  场景 '{scenario_name}' 未执行调优")
                    
            except Exception as e:
                print(f"\n❌ 场景 '{scenario_name}' 调优失败: {str(e)}")
                import traceback
                traceback.print_exc()
                scenario_result["tuning_error"] = str(e)
        
        results.append(scenario_result)
        
        print()
        if success:
            print(f"✅ 场景 {i}/{len(scenarios)} 完成")
        else:
            print(f"❌ 场景 {i}/{len(scenarios)} 失败")
        print()
        
        # 添加短暂延迟避免资源冲突
        import time
        time.sleep(2)
    
    return results


def execute_single_scenario(folder_name, scenario_name, args_list):
    """
    执行单个场景
    """
    cwd = osp.abspath(folder_name)
    run_dir = f"run_{scenario_name}"
    
    # 构建完整命令
    command = ["python", "experiment.py", f"--out_dir={run_dir}"] + args_list
    
    try:
        print(f"   执行: {' '.join(command)}")
        result = subprocess.run(
            command, 
            cwd=cwd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=3600  # 1小时超时
        )
        
        if result.returncode == 0:
            print(f"   ✅ 场景 '{scenario_name}' 完成")
            
            # ================================================================
            # 保存代码快照到场景目录（与 run_experiment 保持一致）
            # ================================================================
            scenario_dir = osp.join(cwd, run_dir)
            
            # 复制 experiment.py
            exp_file = osp.join(cwd, "experiment.py")
            if osp.exists(exp_file):
                shutil.copy(exp_file, osp.join(scenario_dir, "experiment.py"))
                print(f"   📄 已保存 experiment.py 快照")
            
            # 复制 plot.py
            plot_file = osp.join(cwd, "plot.py")
            if osp.exists(plot_file):
                shutil.copy(plot_file, osp.join(scenario_dir, "plot.py"))
                print(f"   📄 已保存 plot.py 快照")
            
            # 复制 notes.txt
            notes_file = osp.join(cwd, "notes.txt")
            if osp.exists(notes_file):
                shutil.copy(notes_file, osp.join(scenario_dir, "notes.txt"))
            
            # 检查是否生成了结果文件
            result_file = osp.join(scenario_dir, "final_info.json")
            if osp.exists(result_file):
                print(f"   📄 结果文件: {result_file}")
            else:
                print(f"   ⚠️  场景完成但未生成结果文件")
            
            return True, "Success"
        else:
            print(f"   ❌ 场景 '{scenario_name}' 失败")
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            
            # 清理失败的运行目录
            failed_dir = osp.join(cwd, run_dir)
            if osp.exists(failed_dir):
                shutil.rmtree(failed_dir)
            
            return False, error_msg
            
    except TimeoutExpired:
        print(f"   ⏰ 场景 '{scenario_name}' 超时")
        
        # 清理超时的运行目录
        timeout_dir = osp.join(cwd, run_dir)
        if osp.exists(timeout_dir):
            shutil.rmtree(timeout_dir)
            
        return False, "Timeout"
    except Exception as e:
        print(f"   💥 场景 '{scenario_name}' 异常: {str(e)}")
        return False, str(e)


# ============================================================================
# 参数调优功能模块（场景级即时调优）
# ============================================================================

def extract_tunable_parameters_from_code(folder_name):
    """
    从 experiment.py 中提取可调参数（改进版：支持更灵活的参数定义格式）
    
    返回: {param_name: {"type": str, "current_value": value, "description": str}}
    """
    exp_file = osp.join(folder_name, "experiment.py")
    
    if not osp.exists(exp_file):
        return {}
    
    try:
        with open(exp_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        params = {}
        
        # 改进的正则表达式：更灵活地匹配各种格式
        # 匹配 parser.add_argument('--param', ..., type=int, ..., default=10, ...)
        # 支持多行、空格变化、顺序变化
        
        # 首先找到所有 add_argument 调用
        arg_pattern = r"parser\.add_argument\(\s*['\"]--([a-zA-Z_][a-zA-Z0-9_]*)['\"]\s*,([^)]+)\)"
        arg_matches = re.findall(arg_pattern, content)
        
        # 排除不应调优的参数
        excluded_params = ['out_dir', 'seed', 'dataset', 'enable_tuning', 'algorithm', 'scenario']
        
        for param_name, arg_content in arg_matches:
            if param_name in excluded_params:
                continue
            
            # 跳过 action='store_true' 类型的参数（布尔标志）
            if "action=" in arg_content and "store_true" in arg_content:
                continue
            
            # 提取 type
            type_match = re.search(r"type\s*=\s*(\w+)", arg_content)
            if not type_match:
                continue
            param_type = type_match.group(1)
            
            # 只处理 int 和 float
            if param_type not in ['int', 'float']:
                continue
            
            # 提取 default
            default_match = re.search(r"default\s*=\s*([\d.eE+-]+|None)", arg_content)
            if not default_match:
                continue
            default_str = default_match.group(1)
            
            # 跳过 default=None
            if default_str == 'None':
                continue
            
            # 提取 help (可选)
            help_match = re.search(r"help\s*=\s*['\"]([^'\"]+)['\"]", arg_content)
            description = help_match.group(1) if help_match else f"Parameter {param_name}"
            
            # 解析默认值
            try:
                if param_type == 'int':
                    params[param_name] = {
                        "type": "int",
                        "current_value": int(float(default_str)),  # 先转 float 再转 int，支持科学计数法
                        "description": description
                    }
                elif param_type == 'float':
                    params[param_name] = {
                        "type": "float",
                        "current_value": float(default_str),
                        "description": description
                    }
            except (ValueError, TypeError) as e:
                print(f"   ⚠️ 跳过参数 {param_name}: 无法解析默认值 '{default_str}' ({e})")
                continue
        
        print(f"   ✓ 成功提取 {len(params)} 个可调参数")
        if params:
            print(f"   参数列表: {', '.join(params.keys())}")
        
        return params
        
    except Exception as e:
        print(f"⚠️ 提取参数时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}


def clean_json_from_response(text):
    """
    从 AI 响应中清理并提取纯 JSON
    
    处理多种格式：
    - 代码块包裹的 JSON
    - 包含 diff 标记的响应
    - 纯 JSON 对象
    """
    import re
    
    # 1. 移除 diff 标记
    if '<<<<<<< SEARCH' in text or '=======' in text or '>>>>>>> REPLACE' in text:
        # 提取 ======= 和 >>>>>>> 之间的内容
        match = re.search(r'=======\s*(.*?)\s*>>>>>>>', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    
    # 2. 移除代码块标记
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # 3. 提取 JSON 对象（支持嵌套）
    # 使用更健壮的方法：从第一个 { 开始，匹配完整的 JSON
    stack = []
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append(char)
        elif char == '}':
            if stack and stack[-1] == '{':
                stack.pop()
                if not stack and start_idx != -1:
                    # 找到完整的 JSON 对象
                    return text[start_idx:i+1]
    
    # 如果上面的方法失败，使用正则
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        return match.group().strip()
    
    return None


def design_tuning_strategy_for_scenario(scenario_info, current_results, tunable_params, coder, folder_name=None):
    """
    让 AI 为当前场景设计参数调优搜索空间（用于随机搜索）
    
    参数:
        scenario_info: 场景信息
        current_results: 当前基线结果
        tunable_params: 可调参数列表
        coder: Aider Coder 对象
        folder_name: 实验文件夹路径（用于保存调试日志）
    
    返回: {"search_space": {...}, "rationale": str, "num_trials": int}
    """
    prompt = f"""You are an expert in hyperparameter optimization. Design a Search Space for Random Search hyperparameter tuning.

# Scenario Information
Name: {scenario_info.get('name', 'Unknown')}
Description: {scenario_info.get('description', '')}

# Current Performance (Baseline)
{json.dumps(current_results, indent=2)}

# Available Parameters
{json.dumps(tunable_params, indent=2)}

# Your Task
Select 2-4 key parameters to tune and define their search ranges for Random Search.

## Parameter Type Specifications

### For Float Parameters:
- Specify: `{{"type": "float", "min": <value>, "max": <value>, "scaling": "linear" or "log"}}`
- Use "log" scaling for parameters that span orders of magnitude (e.g., learning_rate: 0.0001 to 0.1)
- Use "linear" for parameters with narrow ranges

### For Integer Parameters:
- Specify: `{{"type": "int", "min": <value>, "max": <value>}}`
- Examples: num_layers, num_iterations, batch_size

### For Categorical Parameters:
- Specify: `{{"type": "categorical", "values": [val1, val2, ...]}}`
- Examples: activation_function, optimizer

## Requirements
1. Select 2-4 parameters that are most likely to affect performance in THIS scenario
2. For each parameter, define appropriate search range based on its current value and expected impact
3. Choose appropriate scaling (linear vs log) for continuous parameters
4. Recommend number of random trials (typically 8-15)

**CRITICAL: Prioritize algorithm-specific parameters!**
- For subspace-based algorithms: ALWAYS include `subspace_dim` or similar if available
- For momentum-based methods: Include `momentum` coefficient if available
- For federated learning: Consider `local_steps`, `client_fraction`, `batch_size`
- Learning rate is important but not the only parameter to tune!

## Output Format (MUST be valid JSON)
```json
{{
  "search_space": {{
    "learning_rate": {{"type": "float", "min": 0.001, "max": 0.2, "scaling": "log"}},
    "batch_size": {{"type": "categorical", "values": [32, 64, 128]}},
    "num_layers": {{"type": "int", "min": 1, "max": 5}}
  }},
  "rationale": "Explanation of why these parameters and ranges were chosen for THIS scenario",
  "num_trials": 10,
  "expected_improvement": "What improvement we expect to see with optimized parameters"
}}
```

IMPORTANT:
- Return ONLY valid JSON, no extra text
- DO NOT include file editing instructions or diff markers (<<<<<<< SEARCH, =======, >>>>>>> REPLACE)
- DO NOT wrap JSON in code blocks or create new files
- Just output the raw JSON object directly
- Search ranges should be informed by baseline performance and scenario characteristics
- For subspace-based algorithms, consider including subspace_dim, momentum as tunable parameters if available
- num_trials should be 8-15 (balancing exploration vs computational cost)

**Example Response Format (output this EXACTLY, no other text):**
{{
  "search_space": {{
    "learning_rate": {{"type": "float", "min": 0.001, "max": 0.1, "scaling": "log"}}
  }},
  "rationale": "...",
  "num_trials": 10,
  "expected_improvement": "..."
}}
"""
    
    print("🤖 AI 正在设计搜索空间...")
    ai_response = coder.run(prompt)
    
    # 保存 AI 原始响应以便调试（如果提供了 folder_name）
    if folder_name:
        debug_file = osp.join(folder_name, f"debug_tuning_response_{scenario_info.get('name', 'unknown')}.txt")
        try:
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=== AI 原始响应 ===\n")
                f.write(ai_response)
                f.write("\n\n=== 提示词 ===\n")
                f.write(prompt)
            print(f"   💾 调试信息已保存到: {debug_file}")
        except Exception as e:
            print(f"   ⚠️ 保存调试信息失败: {e}")
    
    # 解析 AI 响应
    try:
        import re
        
        # 使用新的清理函数
        json_str = clean_json_from_response(ai_response)
        
        # 如果没有提取到任何 JSON
        if not json_str:
            print("❌ 无法从 AI 响应中提取 JSON")
            print(f"AI 响应前 500 字符: {ai_response[:500]}")
            print(f"\n完整响应已保存到调试文件，请检查")
            return None
        
        print(f"   ✓ 成功提取 JSON ({len(json_str)} 字符)")
        
        # 尝试解析 JSON
        strategy = json.loads(json_str)
        
        # 验证必需字段（新格式）
        if "search_space" in strategy:
            num_trials = strategy.get('num_trials', 10)
            print(f"✅ 成功解析搜索空间: {len(strategy['search_space'])} 个参数, {num_trials} 次试验")
            return strategy
        else:
            print("❌ 策略缺少必需字段 (search_space)")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        print(f"尝试解析的内容 (前 300 字符): {json_str[:300] if json_str else 'None'}")
        return None
    except Exception as e:
        print(f"❌ 解析调优策略失败: {e}")
        return None


def tune_scenario_immediately(folder_name, scenario_info, coder, algo_info=None):
    """
    Scenario-level Immediate Tuning - Separate Script Mode
    
    Strategy:
    1. Refactor experiment.py to ensure `run_experiment(args)` is importable.
    2. Generate a temporary `tune_experiment.py` script.
    3. Run the tuning script to test configs and save summary.
    """
    if not ENABLE_HYPERPARAMETER_TUNING:
        return None
    
    scenario_name = scenario_info["name"]
    print(f"\n" + "="*80)
    print(f"🎯 Scenario Tuning: {scenario_name} (Separate Script Mode)")
    print("="*80)
    
    # 1. Extract tunable parameters
    tunable_params = extract_tunable_parameters_from_code(folder_name)
    if not tunable_params:
        print("⚠️ No tunable parameters found, skipping.")
        return None
        
    # 2. Get baseline results (try new nested structure first)
    run_dir = scenario_info.get("run_dir", f"run_{scenario_name}")
    result_file = osp.join(folder_name, run_dir, "baseline", "final_info.json")
    if not osp.exists(result_file):
        result_file = osp.join(folder_name, run_dir, "final_info.json")
    
    baseline_results = {}
    if osp.exists(result_file):
        with open(result_file, 'r') as f:
            baseline_results = json.load(f)
    
    # 3. Design Tuning Strategy (now returns search_space instead of fixed configs)
    print(f"🤖 AI designing tuning strategy...")
    strategy = design_tuning_strategy_for_scenario(
        scenario_info, baseline_results, tunable_params, coder, folder_name
    )
    
    if not strategy or not strategy.get("search_space"):
        print("❌ Invalid strategy or no search_space defined.")
        return None
    
    # ========================================================================
    # Step A: Ensure experiment.py is importable
    # ========================================================================
    print("🔧 Refactoring experiment.py for importability...")
    ensure_importability_prompt = """
We need to run hyperparameter tuning by importing functions from experiment.py into a separate script.

**Your Task:**
Refactor `experiment.py` to ensure the main training logic is encapsulated in a function `run_experiment(args)` that:
1. Accepts an `args` object (Namespace or dict) as input.
2. Returns the results dictionary (the same dict you save to json).
3. Can be imported without running the script (put the `main()` call under `if __name__ == "__main__":`).

**CRITICAL: Use MINIMAL, TARGETED EDITS (Diff).**
- If `run_experiment` already exists, just ensure it returns results.
- If logic is in `main()`, extract it to `run_experiment(args)`.
- DO NOT rewrite the whole file.
"""
    coder.run(ensure_importability_prompt)
    
    # ========================================================================
    # Step B: Generate independent tuning script with RANDOM SEARCH
    # ========================================================================
    print("📜 Generating tune_experiment.py with Random Search...")
    
    search_space_json = json.dumps(strategy['search_space'], indent=2)
    num_trials = strategy.get('num_trials', 10)
    base_params_json = json.dumps(scenario_info.get('parameters', {}))
    
    tuning_script_prompt = f"""
Create a NEW Python script named `tune_experiment.py` to perform Random Search hyperparameter tuning.

**Requirements:**

1. **Imports:**
   Import `run_experiment` from `experiment`, plus `random`, `numpy as np`, `json`, `os`, `math`, and `argparse`.

2. **Define the Search Space:**
{search_space_json}

3. **Implement `sample_parameters(search_space, rng)` function:**
   - For each parameter in search_space, sample a value based on its type:
   
   **Float parameters:**
   ```python
   if param_spec["type"] == "float":
       if param_spec.get("scaling") == "log":
           # Log-uniform sampling
           log_min = math.log(param_spec["min"])
           log_max = math.log(param_spec["max"])
           value = math.exp(rng.uniform(log_min, log_max))
       else:
           # Linear uniform sampling
           value = rng.uniform(param_spec["min"], param_spec["max"])
   ```
   
   **Integer parameters:**
   ```python
   elif param_spec["type"] == "int":
       value = rng.randint(param_spec["min"], param_spec["max"] + 1)
   ```
   
   **Categorical parameters:**
   ```python
   elif param_spec["type"] == "categorical":
       value = rng.choice(param_spec["values"])
   ```
   
   Return a dictionary of sampled parameters.

4. **Main Random Search Loop:**
   - Run **{num_trials} trials**
   - For each trial i:
     * Sample parameters: `params = sample_parameters(search_space, rng)`
     * Merge with base parameters: {base_params_json}
     * Create args object (Namespace or simple class with attributes)
     * Set `args.out_dir = f"tuning/trial_{{i}}"`
     * Create output directory
     * Call `result = run_experiment(args)` inside try-except
     * Save result to `tuning/trial_{{i}}/final_info.json`
     * Track metrics for best result selection

5. **Best Result Selection:**
   - Compare results across all trials
   - Choose best based on final test accuracy (maximize) or final train loss (minimize)
   - Handle missing/failed trials gracefully

6. **Save Summary:**
   Save to `tuning/tuning_summary.json`:
   ```json
   {{
     "best_config": {{...}},  // Parameters of best trial
     "best_result": {{...}},  // Metrics of best trial
     "all_results": [        // Summary of all trials
       {{"trial": 1, "parameters": {{...}}, "metrics": {{...}}, "status": "success"}},
       ...
     ]
   }}
   ```

**Base Parameters (non-tunable):**
{base_params_json}

**Seed:** Use `rng = random.Random(42)` for reproducibility.

**Output:**
Generate the complete `tune_experiment.py` file code implementing Random Search as specified above.
"""
    coder.run(tuning_script_prompt)
    
    # ========================================================================
    # Step C: Execute Tuning Script
    # ========================================================================
    print("⚙️ Executing tune_experiment.py...")
    cwd = osp.abspath(folder_name)
    
    try:
        # Run the generated script
        result = subprocess.run(
            ["python", "tune_experiment.py"], 
            cwd=cwd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=7200 
        )
        
        if result.returncode != 0:
            print(f"❌ Tuning script failed:\n{result.stderr[:500]}")
            return None
            
        print("✅ Tuning completed.")
        
        # Step D: Load Best Results
        # Look for the summary file in the scenario directory (or root depending on implementation)
        # Note: The prompt asks to save to `tuning/tuning_summary.json` relative to execution
        summary_file = osp.join(cwd, "tuning", "tuning_summary.json")
        
        if osp.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            best_config = summary.get("best_config")
            print(f"🏆 Best Config: {best_config}")
            
            # Save report to scenario directory
            scenario_dir = osp.join(folder_name, run_dir)
            report_file = osp.join(scenario_dir, "tuning_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            # 📄 Save tune_experiment.py to scenario directory (code snapshot)
            tune_script_src = osp.join(cwd, "tune_experiment.py")
            if osp.exists(tune_script_src):
                tune_script_dst = osp.join(scenario_dir, "tune_experiment.py")
                shutil.copy(tune_script_src, tune_script_dst)
                print(f"   📄 已保存 tune_experiment.py 快照到场景目录")
                
            return summary
            
    except TimeoutExpired:
        print("⏰ Tuning timed out.")
        return None
    except Exception as e:
        print(f"💥 Tuning exception: {e}")
        return None


def generate_comprehensive_visualization(folder_name, coder=None, scenario_results=None):
    """
    生成综合可视化结果
    
    参数:
        folder_name: 实验文件夹路径
        coder: Aider Coder 对象（用于让 AI 更新 plot.py）
        scenario_results: 场景执行结果列表
    """
    cwd = osp.abspath(folder_name)
    plots_dir = osp.join(cwd, "multi_scenario_plots")
    
    # 清理并创建绘图目录
    if osp.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir)
    
    # 检查 plot.py 是否存在
    plot_file = osp.join(cwd, "plot.py")
    if not osp.exists(plot_file):
        return False, "plot.py 文件不存在"
    
    # ========================================================================
    # 新增：让 AI 更新 plot.py 以识别所有场景目录
    # ========================================================================
    if coder and scenario_results:
        # 收集所有成功的场景目录
        successful_runs = []
        for result in scenario_results:
            if result.get("status") == "success":
                run_dir = result.get("run_dir", "")
                scenario_name = result.get("name", "")
                description = result.get("description", "")
                successful_runs.append({
                    "directory": run_dir,
                    "label": scenario_name,
                    "description": description
                })
        
        if successful_runs:
            print(f"🤖 让 AI 更新 plot.py 以识别 {len(successful_runs)} 个场景目录...")
            
            update_plot_prompt = f"""
Please update plot.py to visualize results from the following scenario directories:

{json.dumps(successful_runs, indent=2)}

**CRITICAL: Make MINIMAL, TARGETED EDITS - DO NOT rewrite the entire file!**

Key requirements:
1. Update the 'labels' dictionary to map these scenario directories to their labels
2. Read final_info.json from each directory (may need to adjust file paths)
3. Generate comparison plots showing all scenarios
4. Use clear legends and labels to distinguish different scenarios
5. Create professional-looking plots with appropriate styling

Example labels dictionary format:
```python
labels = {{
    "run_baseline": "Baseline Configuration",
    "run_high_learning_rate": "High Learning Rate",
    "run_noisy_data": "Noisy Data Test",
    # ... more scenarios
}}
```

**Instructions:**
- Locate the existing 'labels' dictionary in plot.py
- Replace it with the updated mapping above
- If file reading logic needs adjustment, modify only that section
- DO NOT rewrite plotting code that's already working

Please make TARGETED modifications to plot.py now.
"""
            coder.run(update_plot_prompt)
            print("✅ AI 已更新 plot.py")
    
    # 执行绘图命令
    command = ["python", "plot.py", f"--out_dir={plots_dir}"]
    
    try:
        print("🎨 生成多场景可视化结果...")
        result = subprocess.run(
            command, 
            cwd=cwd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=600  # 10分钟超时
        )
        
        if result.returncode == 0:
            # 检查生成的图表文件
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
            if plot_files:
                print(f"✅ 成功生成 {len(plot_files)} 个图表文件:")
                for plot_file in plot_files:
                    print(f"   📊 {plot_file}")
                return True, f"生成 {len(plot_files)} 个可视化图表"
            else:
                return False, "绘图完成但未生成图表文件"
        else:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            return False, f"绘图失败: {error_msg}"
            
    except TimeoutExpired:
        return False, "绘图超时"
    except Exception as e:
        return False, f"绘图异常: {str(e)}"


def perform_original_experiment_loop(idea, folder_name, coder, baseline_results):
    """
    原有的实验循环（作为备选方案）
    """
    print("\n🔄 执行原有实验循环...")
    
    current_iter = 0
    run = 1
    
    # 生成初始提示词
    next_prompt = coder_prompt.format(
        title=idea["Title"],
        idea=idea["Experiment"],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )
    
    # 实验循环
    while run < MAX_RUNS + 1:
        if current_iter >= MAX_ITERS:
            print("⚠️ 达到最大重试次数，停止当前运行")
            break
        
        print(f"\n--- Run {run} (尝试 {current_iter + 1}/{MAX_ITERS}) ---")
        print("🤖 AI 正在分析和修改代码...")
        coder_out = coder.run(next_prompt)
        print(coder_out)
        
        if "ALL_COMPLETED" in coder_out:
            print("✅ AI 表示所有实验已完成")
            break
        
        print(f"⚙️ 执行实验: python experiment.py --out_dir=run_{run}")
        return_code, next_prompt = run_experiment(folder_name, run)
        
        if return_code == 0:
            print(f"✅ Run {run} 成功完成")
            run += 1
            current_iter = 0
        else:
            print(f"❌ Run {run} 失败，准备重试")
            current_iter += 1
    
    if current_iter >= MAX_ITERS:
        print("\n❌ 实验循环未能完成所有运行（达到最大重试次数）")
        return False
    
    print(f"\n✅ 实验循环完成，共完成 {run - 1} 次运行")
    
    # 执行可视化
    print("\n📊 执行可视化生成...")
    current_iter = 0
    next_prompt = """
Please modify `plot.py` to generate the most relevant plots for the final writeup. 

**CRITICAL: Make MINIMAL, TARGETED EDITS - DO NOT rewrite the entire file!**

**Instructions:**
- Locate the existing 'labels' dictionary in plot.py
- Update it with the correct names for each run directory
- If plotting logic needs adjustment, modify only that specific section
- DO NOT output code that's already working

Focus on updating the labels dictionary and any broken plotting logic.
"""
    
    while True:
        print(f"\n🤖 AI 正在修改 plot.py (尝试 {current_iter + 1}/{MAX_ITERS})...")
        _ = coder.run(next_prompt)
        
        print("⚙️ 执行绘图: python plot.py")
        return_code, next_prompt = run_plotting(folder_name)
        
        current_iter += 1
        
        if return_code == 0:
            print("✅ 可视化生成成功")
            break
        elif current_iter >= MAX_ITERS:
            print("⚠️ 可视化生成失败（达到最大重试次数）")
            break
        else:
            print("❌ 绘图失败，准备重试")
    
    # 文档更新
    print("\n📝 更新实验文档...")
    next_prompt = """
Please modify `notes.txt` with a description of what each plot shows along with the filename of the figure. 
Somebody else will be using `notes.txt` to write a report on this in the future.
"""
    coder.run(next_prompt)
    print("✅ 文档更新完成")
    
    return True
