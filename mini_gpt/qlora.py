"""QLoRA 파인튜닝 설정 헬퍼"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_model_for_qlora(model_name, lora_r=16, lora_alpha=32, target_modules=None):
    """4비트로 모델을 불러오고, 학습 가능한 LoRA 어댑터를 붙인다."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4비트로 양자화된 모델 위에서 그래디언트 체크포인팅 등을 안전하게 쓸 수 있도록 준비
    model = prepare_model_for_kbit_training(model)

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # LLaMA류 모델 기준

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer
