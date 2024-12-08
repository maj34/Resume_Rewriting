import os
import random
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import PeftModel, prepare_model_for_kbit_training
from omegaconf import OmegaConf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(cfg, model_name: str, checkpoint_path: str):
    """
    Load the model and tokenizer based on the configuration.

    Args:
        cfg (OmegaConf): Configuration object containing model and quantization settings.
        model_name (str): The name or path of the pretrained model.
        checkpoint_path (str): The path to the checkpoint for PEFT model.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    # Load the base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side="left", 
        add_eos_token=True, 
        add_bos_token=True
    )

    # Configure BitsAndBytes quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.bnb_config.load_in_4bit,
        bnb_4bit_use_double_quant=cfg.bnb_config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.bnb_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16
    )

    # Reload model with quantization config
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    model = prepare_model_for_kbit_training(model)
    model.resize_token_embeddings(len(tokenizer))

    # Load PEFT model
    model = PeftModel.from_pretrained(
        model=model,
        model_id=checkpoint_path,
        peft_config=bnb_config
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    return model, tokenizer


def torch_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    
    np.random.seed(seed) 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class CustomDataset(Dataset):
    """
    A custom dataset class that handles text data.

    Args:
        dataset (pd.DataFrame): A DataFrame containing a 'text' column.
    """
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.inputs = self.dataset['text'].tolist()
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        if isinstance(idx, list):
            return [self.inputs[i] for i in idx]
        return self.inputs[idx]


def get_prompt_template(category: str) -> str:
    """
    Get the prompt template based on the given category.
    
    Args:
        category (str): The category of the prompt. One of ["engineering", "natural", "social", "humanities", "total"].
    
    Returns:
        str: The prompt template as a string.
    """
    category_map = {
        "engineering": "지원서 내용을 바탕으로 공학 분야 지원자를 위한 잘 구조화된 자기소개서 답변을 생성하세요. 기술적 역량, 문제 해결 능력, 공학 원리의 실질적 응용 경험을 강조하세요. 분석적 사고, 혁신성, 프로젝트 환경에서의 팀워크를 중점으로 다루고, 기술이나 산업 도구와 관련된 실무 경험을 부각시킵니다. 전체적으로 전문적이고 목표 지향적인 톤을 유지하세요.",
        "social": "지원서 내용을 바탕으로 사회과학 분야 지원자를 위한 종합적인 자기소개서 답변을 작성하세요. 분석적 사고, 사회적 역학에 대한 이해, 연구 또는 사회 프로그램과 관련된 경험을 강조하세요. 효과적인 의사소통 능력, 데이터 기반 분석 처리 능력, 사회 문제를 비판적으로 평가할 수 있는 역량을 중심으로 합니다. 이론과 현실 적용 사이의 균형을 맞추며, 협업 작업이나 커뮤니티 참여 사례를 포함하세요.",
        "natural": "지원서 내용을 바탕으로 자연과학 분야 지원자를 위한 정확하고 과학적으로 근거된 자기소개서 답변을 생성하세요. 연구 경험, 분석적 기술, 실증적 데이터를 다룰 수 있는 능력을 강조하세요. 문제 해결 방식, 과학적 방법에 대한 이해, 실험, 논문 또는 과학 프로젝트에의 기여 사항을 포함하세요. 톤은 논리적이고 증거 기반이어야 하며, 자연 세계에 대한 발견과 혁신에 대한 열정을 반영해야 합니다.",
        "humanities": "지원서 내용을 바탕으로 인문학 지원자를 위한 사려 깊고 성찰적인 자기소개서 답변을 작성하세요. 분석적 사고력과 비판적 사고력, 복잡한 텍스트와 아이디어에 대한 이해 능력을 강조하세요. 의사소통 능력, 문화적 인식, 인간 문화와 역사 연구를 통한 개인적 성장을 중점으로 다룹니다. 톤은 지적이고 성찰적이어야 하며, 인문학적 탐구와 그것이 사회에 미치는 관련성을 보여주세요.",
        "total": "지원서 내용을 바탕으로 다양한 분야에 적용 가능한 다재다능하고 적응력 있는 자기소개서 답변을 작성하세요. 의사소통 능력, 팀워크, 리더십, 적응력과 같은 핵심 역량을 강조하세요. 다양한 환경에서 빠르게 학습하고 효과적으로 기여할 수 있는 능력을 부각시키세요. 톤은 긍정적이고 전문적이어야 하며, 다양한 직무에 적합한 폭넓은 역량을 반영하세요."
    }

    return category_map.get(category, category_map["total"])


def run(cfg):
    """
    Main function to generate modified resumes based on given configuration and category.

    Args:
        cfg (OmegaConf): Configuration object that includes:
                         - cfg.model: Model name or path.
                         - cfg.checkpoint_path: Path to checkpoint for PEFT model.
                         - cfg.training_args.seed: Random seed.
                         - cfg.category: Category to select appropriate prompt and data from.
                         - cfg.auth_token (optional): Authentication token if required.
    
    Returns:
        None
    """
    model_name = cfg.model.split('/')[-1]
    torch_seed(cfg.training_args.seed)
    
    model, tokenizer = load_model(cfg, cfg.model, cfg.checkpoint_path)

    # Generation log path
    generation_log_path = f'./generation_log/{model_name}/'
    os.makedirs(generation_log_path, exist_ok=True)
    
    with open('./data/final_self_introduction_classified_with_score.json', 'r', encoding='utf-8') as f:
        resume = json.load(f)

    # Use the category from config
    category = cfg.category

    # Load prompt template according to category
    prompt_template = get_prompt_template(category)

    # Randomly select one entry from the chosen category
    if category not in resume:
        raise ValueError(f"The category '{category}' does not exist in the data.")
    random_key = random.choice(list(resume[category].keys()))

    original_resume = f"질문: {resume[category][random_key]['q1']}\n답변: {resume[category][random_key]['a1']}"
    eval_results = resume[category][random_key]['score']['q1']

    # Format the prompt with the chosen data
    prompt_filled = (
        f"{prompt_template}\n\n"
        f"원본 자기소개서:\n{original_resume}\n\n"
        f"평가 결과:\n{eval_results}\n\n"
        f"수정된 자기소개서:"
    )

    final_test_dataset = pd.DataFrame(columns=['text'])
    final_test_dataset.loc[0, 'text'] = prompt_filled
    
    input_test_dataset = CustomDataset(dataset=final_test_dataset)
    test_dataloader = DataLoader(dataset=input_test_dataset, batch_size=1, shuffle=False)
    
    generated_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating..."):
            batch = tokenizer(batch, padding='max_length', truncation=True, return_tensors="pt", max_length=4096)
            batch = batch.to(device)

            if cfg.auth_token is not None:
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                outputs = model.generate(
                    **batch, 
                    max_new_tokens=2048, 
                    no_repeat_ngram_size=5, 
                    do_sample=True,
                    temperature=1e-12, 
                    use_cache=True, 
                    eos_token_id=terminators
                )
            else:
                outputs = model.generate(
                    **batch, 
                    max_new_tokens=2048, 
                    no_repeat_ngram_size=5, 
                    do_sample=True,
                    temperature=1e-12, 
                    use_cache=True, 
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
            generated_list.append(generated)
            print(generated)
        
    flatten_generated_list = [item for sublist in generated_list for item in sublist]

    # Post-processing generated text
    final = []
    for gen in flatten_generated_list:
        # Split on '수정된 자기소개서:' and take the last part
        if '수정된 자기소개서:' in gen:
            final.append(gen.split('수정된 자기소개서:')[1].strip())
        else:
            # If the prompt does not contain '수정된 자기소개서:', append the whole generation
            final.append(gen.strip())

    final_test_dataset['generated_resume'] = final
    final_test_dataset.to_csv(
        f'{generation_log_path}/{category}_final_clean_resume.csv', 
        index=False, 
        encoding='utf-8'
    )


if __name__ == '__main__':
    args = OmegaConf.from_cli()
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(cfg, args)
    run(cfg)