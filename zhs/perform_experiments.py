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
from subprocess import TimeoutExpired

# ============================================================================
# 配置常量
# ============================================================================

MAX_ITERS = 4           # 每次运行失败后的最大重试次数
MAX_RUNS = 5            # 总共允许的最大实验运行次数
MAX_STDERR_OUTPUT = 1500  # 错误信息的最大显示长度（字符数）
MAX_CODE_GEN_ITERS = 10  # 代码生成阶段的最大迭代次数

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

**CRITICAL OUTPUT STRUCTURE REQUIREMENT:**
- Results MUST be saved DIRECTLY to: {{out_dir}}/final_info.json
- DO NOT create subdirectories like {{out_dir}}/r_0.1/final_info.json
- If testing multiple parameter values, do ONE value per run
- Multiple parameter values will be tested in SEPARATE runs (run_1, run_2, etc.)
- Each run should test a SINGLE configuration and save to {{out_dir}}/final_info.json

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

Please generate BOTH experiment.py and plot.py now. Make sure they are complete, runnable, and follow all requirements above.
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
- Otherwise, explain what needs to be fixed and make the necessary changes

If you find issues, please fix them and regenerate the experiment.py file.
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
You can then implement the next thing on your list."""


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
        
        # 检查输出文件是否存在
        result_file = osp.join(test_dir, "final_info.json")
        if not osp.exists(result_file):
            error_msg = "Test run succeeded but did not generate final_info.json\n"
            error_msg += f"Expected file: {result_file}\n"
            if osp.exists(test_dir):
                contents = os.listdir(test_dir)
                error_msg += f"test_run/ contents: {contents}\n"
                
                # 检查是否错误地创建了子目录
                subdirs = [d for d in contents if osp.isdir(osp.join(test_dir, d))]
                if subdirs:
                    error_msg += "\n⚠️ CRITICAL ERROR: Results were saved to subdirectories!\n"
                    error_msg += f"Found subdirectories: {subdirs}\n"
                    error_msg += "\nThe experiment.py file MUST save results DIRECTLY to:\n"
                    error_msg += f"  {{out_dir}}/final_info.json\n"
                    error_msg += "\nDO NOT create subdirectories for different parameter values.\n"
                    error_msg += "If you need to test multiple parameters, they should be handled\n"
                    error_msg += "in SEPARATE runs (run_1, run_2, etc.), not in subdirectories.\n"
                    error_msg += "\nPlease fix the code to save results directly to the output directory."
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
            next_prompt = """experiment.py file was not created. Please generate the complete experiment.py file now.

Remember the critical requirements:
1. Must accept --out_dir command-line argument
2. Must save results to {out_dir}/final_info.json
3. JSON format: {"metric": {"means": [...], "stds": [...]}}
"""
            continue
        
        syntax_valid, syntax_error = validate_python_syntax(exp_file)
        
        if not syntax_valid:
            print(f"❌ 语法错误:\n{syntax_error}")
            next_prompt = f"""The experiment.py file has syntax errors:

{syntax_error}

Please fix these syntax errors and regenerate the file.
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

Please fix these errors. Common issues to check:
1. Missing import statements
2. Undefined variables or functions
3. Incorrect data types or shapes
4. File I/O errors
5. Missing --out_dir argument handling
6. Incorrect final_info.json format

Please regenerate the corrected experiment.py file.
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

        Please fix plot.py to:
        1. Accept --out_dir command-line argument
        2. Use matplotlib for visualizations
        3. Save plots to the specified output directory as PNG files
        4. Run without errors

        Please regenerate the corrected plot.py file.
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

Please carefully review the pseudocode and fix the implementation:

# Algorithm Pseudocode
{pseudocode}

Key issues identified:
- The implementation does not correctly follow the pseudocode logic
- Some algorithmic steps may be missing or incorrectly implemented
- Mathematical formulas or procedures may not match

Please regenerate experiment.py to correctly implement the pseudocode logic.
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
            # 读取实验结果文件 run_{run_num}/final_info.json
            # 格式要求：{"metric": {"means": [...], "stds": [...]}}
            with open(osp.join(run_dir, "final_info.json"), "r") as f:
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
    if tex_pseudocode:
        print("\n" + "="*80)
        print("📋 检测到 algorithm.tex，启动代码生成阶段（不使用 idea['Pseudocode']）")
        print("="*80 + "\n")
        
        success = generate_code_from_pseudocode(idea, folder_name, coder, algorithm_tex_path=algorithm_tex_path)
        
        if not success:
            print("\n❌ 代码生成阶段失败，无法继续后续实验")
            return False
        
        print("✅ 代码生成阶段完成，进入实验迭代阶段\n")
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

    # ========================================================================
    # 阶段 1: 迭代实验循环（修改为支持 AI 设计场景）
    # ========================================================================
    print("\n" + "="*80)
    print("🔬 阶段 1: 迭代实验循环（集成 AI 设计场景）")
    print("="*80 + "\n")
    
    # 执行 AI 设计的场景
    scenario_results = execute_ai_designed_scenarios(folder_name, ai_scenarios)
    
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
    
    # ========================================================================
    # 阶段 2: 可视化生成（增强版，支持多场景）
    # ========================================================================
    print("\n" + "="*80)
    print("📊 阶段 2: 多场景可视化生成")
    print("="*80 + "\n")
    
    plot_success, plot_message = generate_comprehensive_visualization(folder_name)
    
    if plot_success:
        print(f"✅ {plot_message}")
    else:
        print(f"❌ {plot_message}")
        
        # 让 AI 修复可视化代码
        print("🤖 AI 正在修复可视化代码...")
        fix_plot_prompt = """
The multi-scenario plotting failed. Please fix plot.py to properly visualize all the experimental scenarios.

Key requirements:
1. Read results from all successful scenario directories
2. Generate comprehensive comparison plots showing different scenarios
3. Use clear labels and legends to distinguish different scenarios
4. Create multiple plot types to show different aspects of the results:
   - Convergence curves for each metric across scenarios
   - Scenario comparison bar charts
   - Performance summary plots

Please provide the complete, corrected plot.py file.
"""
        coder.run(fix_plot_prompt)
        
        # 重新尝试绘图
        print("🔄 重新尝试生成可视化...")
        plot_success, plot_message = generate_comprehensive_visualization(folder_name)
        
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

def execute_ai_designed_scenarios(folder_name, scenarios):
    """
    执行 AI 设计的场景
    """
    cwd = osp.abspath(folder_name)
    results = []
    
    print(f"🚀 开始执行 {len(scenarios)} 个 AI 设计的场景")
    
    for i, scenario in enumerate(scenarios, 1):
        scenario_name = scenario.get('name', f'scenario_{i}').replace(' ', '_').lower()
        parameters = scenario.get('parameters', {})
        
        print(f"\n--- 场景 {i}/{len(scenarios)}: {scenario_name} ---")
        print(f"   描述: {scenario.get('description', 'No description')}")
        
        # 构建命令参数列表
        args_list = []
        for key, value in parameters.items():
            args_list.append(f"{key}={value}")
        
        # 执行场景
        success, result_info = execute_single_scenario(folder_name, scenario_name, args_list)
        
        scenario_result = {
            "name": scenario_name,
            "description": scenario.get('description', ''),
            "parameters": parameters,
            "expected_insight": scenario.get('expected_insight', ''),
            "status": "success" if success else "failed",
            "run_dir": f"run_{scenario_name}"
        }
        
        if not success:
            scenario_result["error"] = result_info
        
        results.append(scenario_result)
        
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
            
            # 检查是否生成了结果文件
            result_file = osp.join(cwd, run_dir, "final_info.json")
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


def generate_comprehensive_visualization(folder_name):
    """
    生成综合可视化结果
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
Be sure to fill in the "labels" dictionary with the correct names for each run that you want to plot.
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
