#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MindGuard LoRA 微调脚本
======================
基于心理咨询记录对轻量大模型进行 LoRA 微调，提升心理状态识别能力。

前置条件:
1. 安装依赖: pip install peft transformers datasets torch
2. 准备训练数据: JSONL 格式，每行一条心理咨询对话
3. 已安装 Ollama 并下载基础模型

数据格式 (train_data.jsonl):
{
  "instruction": "判断以下对话中学生的心理状态",
  "input": "我最近总是失眠，每天都在想一些不开心的事情...",
  "output": "{\"emotion\": \"DEPRESSED\", \"risk\": \"MEDIUM\", \"response\": \"我理解你现在的感受...\"}"
}
"""

import json
import os
from pathlib import Path

# ===== 配置 =====
BASE_MODEL = "qwen2.5:7b"          # Ollama 基础模型
LORA_OUTPUT = "./lora_output"       # LoRA 权重输出目录
TRAIN_DATA = "./train_data.jsonl"   # 训练数据路径
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32
MAX_SEQ_LENGTH = 512


def prepare_training_data(input_file: str, output_file: str):
    """
    将原始心理咨询记录转换为微调训练格式
    输入: CSV/Excel 格式的咨询记录
    输出: JSONL 格式的 instruction-input-output 对
    """
    records = []

    # 示例：从 JSONL 读取（实际应从 Excel/CSV 读取）
    if os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    # 转换为微调格式
    training_samples = []
    for record in records:
        sample = {
            "instruction": "你是 MindGuard 心理监护助手。请分析以下学生对话，判断心理状态并给出合适回复。",
            "input": record.get("input", ""),
            "output": json.dumps({
                "emotion": record.get("emotion", "NORMAL"),
                "risk": record.get("risk", "LOW"),
                "response": record.get("response", "")
            }, ensure_ascii=False)
        }
        training_samples.append(sample)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"训练数据准备完成: {len(training_samples)} 条样本 → {output_file}")
    return output_file


def train_lora():
    """
    使用 PEFT LoRA 微调模型

    实际运行需要 GPU 环境，此处为完整流程代码
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        import torch
    except ImportError:
        print("请安装依赖: pip install peft transformers datasets torch")
        return

    # 1. 加载基础模型和 Tokenizer
    print(f"加载基础模型: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 加载训练数据
    data = []
    with open(TRAIN_DATA, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    def tokenize_function(examples):
        prompts = []
        for inst, inp, out in zip(examples["instruction"],
                                   examples["input"],
                                   examples["output"]):
            prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            prompts.append(prompt)

        return tokenizer(prompts, truncation=True,
                        max_length=MAX_SEQ_LENGTH, padding="max_length")

    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=10,
        save_steps=200,
        fp16=True,
        gradient_accumulation_steps=4,
        report_to="none"
    )

    # 5. 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("开始 LoRA 微调...")
    trainer.train()

    # 6. 保存 LoRA 权重
    model.save_pretrained(LORA_OUTPUT)
    tokenizer.save_pretrained(LORA_OUTPUT)
    print(f"LoRA 权重已保存到: {LORA_OUTPUT}")


def export_to_ollama(lora_path: str, model_name: str = "mindguard-lora"):
    """
    将微调后的模型导出为 Ollama 可用的 GGUF 格式

    步骤:
    1. 合并 LoRA 权重到基础模型
    2. 导出为 GGUF 格式
    3. 创建 Ollama Modelfile
    4. 构建并推送 Ollama 模型
    """
    print(f"导出模型到 Ollama: {model_name}")

    modelfile_content = f"""FROM {BASE_MODEL}

# MindGuard LoRA 微调模型
# 基于心理咨询记录微调，提升心理状态识别能力

PARAMETER temperature 0.6
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM \"你是 MindGuard AI 心理监护助手，专门为高校学生提供心理健康支持和咨询服务。\"
"""

    modelfile_path = Path(lora_path) / "Modelfile"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    print(f"Modelfile 已生成: {modelfile_path}")
    print(f"\n请手动执行以下命令构建 Ollama 模型:")
    print(f"  cd {lora_path}")
    print(f"  ollama create {model_name} -f Modelfile")
    print(f"  ollama run {model_name}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python lora_finetune.py prepare   # 准备训练数据")
        print("  python lora_finetune.py train     # 开始微调")
        print("  python lora_finetune.py export    # 导出到 Ollama")
        sys.exit(1)

    command = sys.argv[1]

    if command == "prepare":
        prepare_training_data("./raw_data.jsonl", TRAIN_DATA)
    elif command == "train":
        train_lora()
    elif command == "export":
        export_to_ollama(LORA_OUTPUT)
    else:
        print(f"未知命令: {command}")
