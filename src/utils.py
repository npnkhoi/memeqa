from typing import Dict
from src.const import MODEL_FAMILY
from transformers import (
    LlavaForConditionalGeneration, LlamaForCausalLM, Blip2ForConditionalGeneration, 
    InstructBlipForConditionalGeneration, Qwen2VLForConditionalGeneration, 
    BitsAndBytesConfig, AutoProcessor, InstructBlipProcessor
)
import torch
from enum import Enum
import wandb

def get_metric_names(task, mode):
    assert task in ['QA']
    assert mode in ['train', 'eval']
    return ['acc']

def get_is_enc_dec(name: str) -> bool:
    if 'flan-t5' in name:
        return True
    return False

class MODEL_ID(Enum):
    INSTRUCTBLIP = 'Salesforce/instructblip-vicuna-7b'
    LLAVA_v15 = 'llava-hf/llava-1.5-7b-hf'
    QWEN = 'Qwen/Qwen2-VL-7B-Instruct'
    BLIP2 = 'Salesforce/blip2-flan-t5-xl'
    
    INSTRUCTBLIP_XXL = 'Salesforce/instructblip-flan-t5-xxl'
    BLIP2_XXL = 'Salesforce/blip2-flan-t5-xxl'
    BLIP2_SMALL = 'Salesforce/blip2-opt-2.7b'
    LLAVA_v16 = 'llava-hf/llava-v1.6-mistral-7b-hf'
    CLIP = 'openai/clip-vit-large-patch14'

    # below is not working
    COGVLM = 'THUDM/cogvlm2-llama3-chat-19B'
    BLIP_VQA = 'Salesforce/blip-vqa-base'


def get_model_family(model_name: str) -> MODEL_FAMILY:
    if 'llama' in model_name:
        return MODEL_FAMILY.LLAMA
    elif 'blip2' in model_name:
        return MODEL_FAMILY.BLIP2
    elif 'instructblip' in model_name:
        return MODEL_FAMILY.INSTRUCT_BLIP
    elif 'llava' in model_name:
        return MODEL_FAMILY.LLAVA
    elif 'Qwen' in model_name:
        return MODEL_FAMILY.QWEN
    elif 'gpt' in model_name:
        return MODEL_FAMILY.GPT
    else:
        raise ValueError(f"Unknown model family for model_name: {model_name}")

def model_factory(model_id: str):
    if get_model_family(model_id) == MODEL_FAMILY.LLAMA:
        base_model = LlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map='auto',
        )
    elif get_model_family(model_id) == MODEL_FAMILY.BLIP2:
        print('Using Blip2')
        print("WARNING: This model cannot be run in a multi-gpu env.")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map='auto',
        ) 
    elif get_model_family(model_id) == MODEL_FAMILY.INSTRUCT_BLIP:
        base_model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map='auto',
        ) 
    elif get_model_family(model_id) == MODEL_FAMILY.QWEN:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map='auto',
        ) 
    elif get_model_family(model_id) == MODEL_FAMILY.LLAVA:
        base_model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        # tokenizer.padding_side = "left" 
    else:
        raise ValueError(f"Unknown model family for model_name: {model_id}")

    return base_model

def processor_n_tokenizer_factory(model_id: str):
    if get_model_family(model_id) == MODEL_FAMILY.INSTRUCT_BLIP:
        processor = InstructBlipProcessor.from_pretrained(model_id)
    elif get_model_family(model_id) == MODEL_FAMILY.QWEN:
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels = 256*28*28,
            max_pixels = 480*28*28
        )
    else:
        processor = AutoProcessor.from_pretrained(model_id)
    
    # the tokenizer is sometimes the processor, sometimes not
    # so we explicitly pin down its reference here
    if get_model_family(model_id) == MODEL_FAMILY.LLAMA:
        tokenizer = processor
    else:
        tokenizer = processor.tokenizer

    return processor, tokenizer


from enum import Enum
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Dict, Generator, List, Union
from transformers import HfArgumentParser
import wandb


def log(d: Dict):
    wandb.log(d)
    print(d)


def batch_iter(lst: List, batch_size: int) -> Generator:
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)

@dataclass
class Question:
    id: str
    img: str
    question: str
    options: List[str]
    general_type: str
    specific_type: str
    answer_key: Union[int, List[int]]
    

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'


# CONSTANTS
MAX_ITER = 20

@dataclass
class MyArguments:
    num_iter: int = field(default=MAX_ITER)
    run_name: str = field(default="af_run")


def af_init() -> MyArguments:
    """
    Parse the arguments and initialize wandb.
    Returns the parsed arguments.
    """
    parser = HfArgumentParser((MyArguments))
    args: MyArguments = parser.parse_args_into_dataclasses()[0]
    wandb.init(
        project='memeqa',
        group='af',
        name=args.run_name,
    )
    args.run_id = wandb.run.id
    return args