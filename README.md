# LLM Book Source Code

This repository contains localized practice-code bundles for the LLM book series by dirmich, published by Highmaru Press.

## Localized Bundles

```text
sourcecode/
├── ko/   # Korean source code, README, and Colab notebooks
├── en/   # English source code, README, and Colab notebooks
└── jp/   # Japanese source code, README, and Colab notebooks
```

Each language folder contains:

```text
README.md
requirements.txt
run_demo.py
ddp_simulation.py
mini_gpt/
*.ipynb
```

Language-specific notebooks and Python files live only under `sourcecode/`. The repository root is kept for publishing and generation scripts.

## Quick Start

```bash
cd sourcecode/en
pip install -r requirements.txt
python run_demo.py
python ddp_simulation.py
```

For Colab, upload the notebook from the language folder you need and run the cells from top to bottom.

## Books Covered

- **LLM Fundamentals: Building Large Language Models from the Ground Up**
- **Advanced LLM: Building Your Own LLM Service with Modern Techniques**

The implementations are intentionally compact and educational. Production LLM systems need additional engineering such as optimized CUDA kernels, distributed training frameworks, data pipelines, monitoring, and evaluation.
