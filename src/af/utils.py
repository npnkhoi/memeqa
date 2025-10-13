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
    img: str
    question: str
    options: List[str]
    general_type: str
    specific_type: str
    answer_key: Union[int, List[int]]
    method: str
    id: str | None = None
    mask: str=None
    masked_sentence: str=None
    active: bool=None
    # output: int
    correct: Union[bool, Dict]=None
    valid: bool=None
    previous_distractors: List[str]=field(default_factory=list)
    

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
        entity='khoi-ml',
        name=args.run_name,
    )
    args.run_id = wandb.run.id
    return args