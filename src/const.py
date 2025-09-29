from enum import Enum, StrEnum

DEFAULT_DATA_DIR = 'data/none_plus'

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'


class SubsetEnum(Enum):
    ALL = "all"
    HALF = "half"
    PILOT = "pilot"


class MODEL_FAMILY(Enum):
    LLAMA = "llama"
    BLIP2 = "blip2"
    INSTRUCT_BLIP = "instruct_blip"
    LLAVA = "llava"
    QWEN = "qwen"
    GPT = "gpt"

QTYPE_SHORT2LONG = {
    "bk_id": "Background Identification",
    "condiional_sentiment_id": "Conditional Sentiment Identification",
    "conditional_action_id": "Conditional Action Identification",
    "deriv_id": "Derivation Identification",
    "intent_comp": "Intent Completion",
    "intent_completion": "Derivation Completion",
    "intent_id": "Intent Identification",
    "tar_id": "Target Identification",
    "target_action_id": "Target-Action Identification",
    "target_sentiment_id": "Target-Sentiment Identification",
    "viz_id": "Visual Identification"
}

MULTI_QTYPES = [
    "Background Identification",
    "Derivation Identification",
    "Target Identification",
    "Target-Action Identification",
    "Target-Sentiment Identification",
    "Visual Identification"
]

SINGLE_QTYPES = [
    "Conditional Sentiment Identification",
    "Conditional Action Identification",
    "Intent Completion",
    "Derivation Completion",
    "Intent Identification",
]

ALL_TYPES = MULTI_QTYPES + SINGLE_QTYPES

QTYPE_LONG2SHORT = {v: k for k, v in QTYPE_SHORT2LONG.items()}


class GENERAL_TYPE(StrEnum):
    SINGLE = 'single'
    MULTIPLE = 'multi'