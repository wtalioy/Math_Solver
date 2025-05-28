# Math Solver: Large Language Model Fine-Tuning and Reasoning Framework

## Overview
This repository provides a comprehensive framework for fine-tuning, evaluating, and deploying large language models (LLMs) for mathematical reasoning tasks, especially elementary school math problems. It supports multiple advanced training paradigms, including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), Chain-of-Thought (CoT), Function Calling, and Generalized Reward Preference Optimization (GRPO).

## Features
- **Supervised Fine-Tuning (SFT):** Fine-tune LLMs with high-quality, step-by-step reasoning data to improve mathematical problem-solving accuracy.
- **Direct Preference Optimization (DPO):** Leverage preference pairs (better/worse answers) to further optimize model reasoning.
- **Chain-of-Thought (CoT):** Enhance model performance by providing intermediate reasoning steps.
- **Function Calling:** Integrate external tools (e.g., calculators) for complex computation via LLM function-calling capabilities.
- **GRPO:** Use custom reward functions and reinforcement learning to optimize for structured reasoning and answer formats.

## Directory Structure
- `data/` — All training, validation, and test datasets in JSON/JSONL format.
- `basic_sft/` — Basic SFT training and inference scripts.
- `cot/` — Chain-of-Thought experiments and reports.
- `dpo/` — DPO training, inference, and data preparation scripts.
- `function_calling/` — Function-calling inference and related reports.
- `grpo/` — GRPO training, reward functions, validation, and documentation.
- `sft/` — SFT training, batch inference, and reporting.
- `utils/` — Data cleaning and splitting utilities.

## Quick Start
### 1. Environment Setup
- Python 3.8+
- Install dependencies:
  `transformers`, `trl`, `peft`, `swanlab`, `datasets`, `loguru`, etc.

### 2. Supervised Fine-Tuning (SFT)
- Train: `python sft/train.py`
- Batch Inference: `python sft/test.py`

### 3. DPO Training
- Train: `python dpo/qwen_dpo_ft.py`
- Inference: `python dpo/dpo_infer.py`

### 4. GRPO Training
- Train: `python grpo/train_hf.py`
- Validation: `python grpo/validation.py`

### 5. Function Calling
- Prepare MCP: `pip install mcp-server-calculator`
- Inference: `python function_calling/infer_server.py`

## Model Checkpoints
- Pretrained and fine-tuned model checkpoints are referenced in scripts (e.g., `Qwen2.5-0.5B-Instruct`, `Qwen3-0.6B`). Download or specify paths as needed.

## Reports & Documentation
- Each module contains a `Report.md` summarizing methodology, experiments, and results.
- See `grpo/doc/Report.md` for detailed GRPO experiments and ablation studies.

## Citation
If you use this repository, please cite the original authors and models referenced in the code and reports.

## License
This project is for research and educational purposes. See individual files for third-party license information.
