# LLM 기초 & LLM 고급 — 소스코드 전체

Highmaru Press · dirmich 지음, 2권 시리즈의 전체 실습 코드입니다.

- **1권** 《LLM 기초: 밑바닥부터 만드는 대규모 언어모델》 — 토크나이저부터 미니 GPT까지 트랜스포머 원리
- **2권** 《LLM 고급: 최신 기법으로 나만의 LLM 서비스 만들기》 — RoPE, MoE, 양자화, DPO, RAG, 에이전트, 분산학습, Mamba, 멀티모달

## 폴더 구성

```
.
├── LLM_기초_Colab_실습.ipynb    # 1권 전체 코드가 담긴 Colab 노트북
├── LLM_고급_Colab_실습.ipynb     # 2권 전체 코드가 담긴 Colab 노트북
├── run_demo.py                    # 1권 mini_gpt 전체 파이프라인 데모 스크립트
├── ddp_simulation.py                # 2권 14장: 데이터 병렬화(All-Reduce) 시뮬레이션 스크립트
├── requirements.txt                  # 필요한 파이썬 패키지 목록

└── mini_gpt/                            # 1권+2권에서 만든 모듈 전체를 담은 패키지
    ├── __init__.py
    ├── tokenizer.py           # [1권 2장] BPE 토크나이저
    ├── embedding.py            # [1권 3장] 토큰 임베딩 + 위치 임베딩
    ├── attention.py             # [1권 4장] Self-Attention, Multi-Head Attention
    ├── transformer_block.py      # [1권 5장] 트랜스포머 블록
    ├── model.py                   # [1권 6장] 미니 GPT 전체 모델
    ├── dataset.py                  # [1권 7장] 학습 데이터셋
    ├── train.py                     # [1권 7장] 학습 루프
    ├── generate.py                    # [1권 8장] 텍스트 생성 (greedy/temperature/top-k/top-p)
    ├── rope.py                         # [2권 2장] RoPE(회전 위치 인코딩)
    ├── gqa.py                           # [2권 3장] Grouped-Query Attention
    ├── moe.py                            # [2권 4장] Mixture-of-Experts + 부하분산 손실
    ├── quantize.py                        # [2권 5장] INT8 양자화
    ├── kv_cache.py                         # [2권 6장] KV 캐시 어텐션
    ├── qlora.py                             # [2권 7장] QLoRA 로딩 헬퍼
    ├── dpo.py                                # [2권 8장] DPO 손실 함수
    ├── reward_model.py                        # [2권 9장] 보상모델 (RLHF)
    ├── rag.py                                  # [2권 11장] 청킹 + 벡터 검색
    ├── tools.py                                 # [2권 12장] 도구 호출(Function Calling)
    ├── agent.py                                  # [2권 13장] ReAct 에이전트 루프
    ├── merge.py                                   # [2권 15장] 모델 병합 (평균/Task Arithmetic/TIES)
    ├── ssm.py                                      # [2권 17장] State Space Model (Mamba)
    └── vision.py                                    # [2권 18장] 멀티모달 비전 인코더
```

## 사용 방법

### 방법 1. Google Colab에서 실행 (추천)

1. [colab.research.google.com](https://colab.research.google.com) 접속
2. `파일 > 노트 업로드`에서 `LLM_기초_Colab_실습.ipynb` 또는 `LLM_고급_Colab_실습.ipynb` 업로드
3. `런타임 > 런타임 유형 변경`에서 GPU 선택 (1권은 T4로 충분, 2권 일부 장은 L4/A100 권장)
4. 위에서부터 순서대로 셀 실행 (`Shift+Enter`)

### 방법 2. 로컬/서버에서 mini_gpt 패키지로 실행

```bash
pip install -r requirements.txt
python run_demo.py           # 1권: 토크나이저 학습 -> 모델 학습 -> 생성까지 전체 파이프라인
python ddp_simulation.py     # 2권 14장: 데이터 병렬화 그래디언트 동기화 시뮬레이션 (CPU 다중 프로세스)
```

### 방법 3. mini_gpt 모듈을 직접 가져다 쓰기

```python
# 1권 모듈
from mini_gpt.tokenizer import BPETokenizer
from mini_gpt.model import MiniGPT
from mini_gpt.generate import generate

# 2권 모듈
from mini_gpt.rope import build_rope_cache, apply_rope
from mini_gpt.gqa import GroupedQueryAttention
from mini_gpt.moe import MoELayer
from mini_gpt.merge import average_merge, ties_merge
```

## 참고

- 9~11장(1권, Hugging Face/LoRA/프롬프트), 그리고 2권의 5·7·8·9·11·16·18장 등 상당수는
  `transformers`, `peft`, `trl`, `bitsandbytes`, `faiss`, `sentence-transformers`,
  `vllm` 같은 외부 라이브러리를 직접 사용하므로, 별도의 `mini_gpt` 모듈 없이 노트북의
  해당 챕터 셀을 그대로 참고하세요.
- 이 코드는 교육용으로 규모를 축소한 구현입니다. 실제 프로덕션급 LLM에는 전용 CUDA
  커널, 검증된 분산학습 프레임워크(Megatron-LM, DeepSpeed), 대규모 데이터 파이프라인
  등 추가 엔지니어링이 필요합니다.
