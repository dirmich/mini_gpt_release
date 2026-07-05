# LLM Fundamentals & Advanced LLM - Complete Source Code

Complete practice code for the two-book Highmaru Press series by dirmich.

- **Book 1**, *LLM Fundamentals: Building Large Language Models from the Ground Up* - tokenizer, embeddings, attention, Transformer blocks, and a mini GPT.
- **Book 2**, *Advanced LLM: Building Your Own LLM Service with Modern Techniques* - RoPE, GQA, MoE, quantization, DPO, RAG, agents, distributed training, Mamba, and multimodal models.

## Layout

```text
.
├── LLM_Fundamentals_Colab_Practice.ipynb
├── Advanced_LLM_Colab_Practice.ipynb
├── run_demo.py
├── ddp_simulation.py
├── requirements.txt
└── mini_gpt/
```

## Usage

### Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com).
2. Upload one of the notebooks.
3. Select a GPU runtime.
4. Run cells from top to bottom with `Shift+Enter`.

### Local Python

```bash
pip install -r requirements.txt
python run_demo.py
python ddp_simulation.py
```

The implementations are intentionally compact and educational. Production LLM systems need additional engineering such as optimized CUDA kernels, distributed training frameworks, data pipelines, monitoring, and evaluation.
