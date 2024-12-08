import os
import random

import numpy as np
import pandas as pd
import torch
import wandb 

from datasets import Dataset
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The random seed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)      
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.

    Args:
        model (nn.Module): The model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )

def load_model(cfg):
    """
    Load model and tokenizer using configuration.

    Args:
        cfg (OmegaConf): Configuration object.

    Returns:
        tuple: (model, tokenizer)
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = cfg.bnb_config.load_in_4bit,
        bnb_4bit_use_double_quant = cfg.bnb_config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type = cfg.bnb_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = torch.float16
    )

    if cfg.auth_token is not None:
        model = AutoModelForCausalLM.from_pretrained(cfg.model, quantization_config=bnb_config, token=cfg.auth_token)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model, token=cfg.auth_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    return model, tokenizer

def get_system_prompt(category: str) -> str:
    """
    Return a system prompt based on the given category.

    Args:
        category (str): The category of the prompt. 
                        One of ["engineering", "social", "natural", "humanities", "total"].

    Returns:
        str: The system prompt.
    """
    prompts = {
        "engineering": "지원서 내용을 바탕으로 공학 분야 지원자를 위한 잘 구조화된 자기소개서 답변을 생성하세요. 기술적 역량, 문제 해결 능력, 공학 원리의 실질적 응용 경험을 강조하세요. 분석적 사고, 혁신성, 프로젝트 환경에서의 팀워크를 중점으로 다루고, 기술이나 산업 도구와 관련된 실무 경험을 부각시킵니다. 전체적으로 전문적이고 목표 지향적인 톤을 유지하세요.",
        "social": "지원서 내용을 바탕으로 사회과학 분야 지원자를 위한 종합적인 자기소개서 답변을 작성하세요. 분석적 사고, 사회적 역학에 대한 이해, 연구 또는 사회 프로그램과 관련된 경험을 강조하세요. 효과적인 의사소통 능력, 데이터 기반 분석 처리 능력, 사회 문제를 비판적으로 평가할 수 있는 역량을 중심으로 합니다. 이론과 현실 적용 사이의 균형을 맞추며, 협업 작업이나 커뮤니티 참여 사례를 포함하세요.",
        "natural": "지원서 내용을 바탕으로 자연과학 분야 지원자를 위한 정확하고 과학적으로 근거된 자기소개서 답변을 생성하세요. 연구 경험, 분석적 기술, 실증적 데이터를 다룰 수 있는 능력을 강조하세요. 문제 해결 방식, 과학적 방법에 대한 이해, 실험, 논문 또는 과학 프로젝트에의 기여 사항을 포함하세요. 톤은 논리적이고 증거 기반이어야 하며, 자연 세계에 대한 발견과 혁신에 대한 열정을 반영해야 합니다.",
        "humanities": "지원서 내용을 바탕으로 인문학 지원자를 위한 사려 깊고 성찰적인 자기소개서 답변을 작성하세요. 분석적 사고력과 비판적 사고력, 복잡한 텍스트와 아이디어에 대한 이해 능력을 강조하세요. 의사소통 능력, 문화적 인식, 인간 문화와 역사 연구를 통한 개인적 성장을 중점으로 다룹니다. 톤은 지적이고 성찰적이어야 하며, 인문학적 탐구와 그것이 사회에 미치는 관련성을 보여주세요.",
        "total": "지원서 내용을 바탕으로 다양한 분야에 적용 가능한 다재다능하고 적응력 있는 자기소개서 답변을 작성하세요. 의사소통 능력, 팀워크, 리더십, 적응력과 같은 핵심 역량을 강조하세요. 다양한 환경에서 빠르게 학습하고 효과적으로 기여할 수 있는 능력을 부각시키세요. 톤은 긍정적이고 전문적이어야 하며, 다양한 직무에 적합한 폭넓은 역량을 반영하세요."
    }
    return prompts.get(category, prompts["total"])

def run(cfg):
    """
    Main function to run the training and evaluation pipeline.

    Args:
        cfg (OmegaConf): Configuration object.
    """
    set_random_seed(cfg.training_args.seed)
        
    # wandb config
    project_name = 'Resume_Rewriting'
    run_name = f'{cfg.run_name}'
    wandb.login(key='')
    wandb.init(project=project_name, name=run_name, config=OmegaConf.to_container(cfg, resolve=True))
    
    model_name = cfg.model.split('/')[-1]
    checkpoint_path = f"./checkpoint/{model_name}/{cfg.checkpoint_path}"
    os.makedirs(checkpoint_path, exist_ok=True)

    model, tokenizer = load_model(cfg)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    model = prepare_model_for_kbit_training(model)
    model.resize_token_embeddings(len(tokenizer))

    print(model.config)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Dataset Preparation
    clean_resume_df = pd.read_csv(cfg.dataset)

    dataset = pd.DataFrame(columns=['text'])
    system_prompt = get_system_prompt(cfg.category)

    for _, row in tqdm(clean_resume_df.iterrows(), total=len(clean_resume_df), desc="Load Dataset..."):
        questions = row[row.index.str.startswith('q')].dropna().tolist()
        answers = row[row.index.str.startswith('a')].dropna().tolist()
        
        for q, a in zip(questions, answers):
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
            input_data = tokenizer.apply_chat_template(
                message, 
                max_length=2048, 
                tokenize=False, 
                add_generation_prompt=False, 
                return_tensors="pt"
            )
            
            dataset.loc[len(dataset)] = [input_data]

    # Train, Valid Split
    train, valid = train_test_split(dataset, test_size=0.1, random_state=cfg.training_args.seed)

    train_dataset = Dataset.from_pandas(train)
    valid_dataset = Dataset.from_pandas(valid)
        
    config = LoraConfig(
        r = cfg.lora_config.r, 
        lora_alpha = cfg.lora_config.lora_alpha, 
        target_modules = list(cfg.lora_config.target_modules),
        lora_dropout = cfg.lora_config.lora_dropout, 
        bias = cfg.lora_config.bias, 
        task_type = cfg.lora_config.task_type
    )

    model = get_peft_model(model, config)
    model.resize_token_embeddings(len(tokenizer))

    print_trainable_parameters(model)

    training_args = TrainingArguments( 
        output_dir = checkpoint_path,
        num_train_epochs = cfg.training_args.num_train_epochs,
        per_device_train_batch_size = cfg.training_args.per_device_train_batch_size,
        per_device_eval_batch_size = cfg.training_args.per_device_eval_batch_size,
        gradient_accumulation_steps = cfg.training_args.gradient_accumulation_steps,
        evaluation_strategy = cfg.training_args.evaluation_strategy,
        save_strategy = cfg.training_args.save_strategy,
        save_total_limit = cfg.training_args.save_total_limit,
        save_steps = cfg.training_args.save_steps,
        eval_steps = cfg.training_args.eval_steps,
        logging_steps = cfg.training_args.logging_steps,
        learning_rate = cfg.training_args.learning_rate,
        weight_decay = cfg.training_args.weight_decay,
        seed = cfg.training_args.seed,
        fp16 = cfg.training_args.fp16,
        load_best_model_at_end = cfg.training_args.load_best_model_at_end,
        report_to = cfg.training_args.report_to
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=2048,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    
    results = trainer.evaluate()
    print(f"Perplexity: {torch.exp(torch.tensor(results['eval_loss']))}")
    
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    
if __name__=='__main__':
    args = OmegaConf.from_cli()
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(cfg, args)
    run(cfg)