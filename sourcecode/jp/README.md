# LLM基礎 & LLM上級 - ソースコード一式

dirmich 著、Highmaru Press の2冊シリーズに対応する実習コードです。

- **第1巻**『LLM基礎: 大規模言語モデルを基礎から作る』- トークナイザー、埋め込み、アテンション、Transformerブロック、ミニGPT。
- **第2巻**『LLM上級: 最新技術で自分だけのLLMサービスを作る』- RoPE、GQA、MoE、量子化、DPO、RAG、エージェント、分散学習、Mamba、マルチモーダル。

## 構成

```text
.
├── LLM基礎_Colab実習.ipynb
├── LLM上級_Colab実習.ipynb
├── run_demo.py
├── ddp_simulation.py
├── requirements.txt
└── mini_gpt/
```

## 使い方

### Google Colab

1. [colab.research.google.com](https://colab.research.google.com) を開きます。
2. ノートブックをアップロードします。
3. GPUランタイムを選択します。
4. 上から順番に `Shift+Enter` で実行します。

### ローカルPython

```bash
pip install -r requirements.txt
python run_demo.py
python ddp_simulation.py
```

この実装は教育用に小さくまとめたものです。実運用のLLMシステムには、最適化CUDAカーネル、分散学習フレームワーク、データパイプライン、監視、評価などの追加エンジニアリングが必要です。
