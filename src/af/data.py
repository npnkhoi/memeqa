"""
Data adapters
"""

from enum import Enum
import json
import os
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Graph:
    nodes: List[Dict]
    edges: List[Dict]

SEMEVAL_IMAGES: List[str] = list(json.load(open('data/all_implicit_messages.json')).keys()) # img names for each meme 
img_fn_to_id = lambda img_path: str(SEMEVAL_IMAGES.index(img_path) + 1)

MEME_INTENTS = json.load(open('data/meme_intent.json'))               # text, image, background, and intent
MEME_INTENTS = {v['img']: {**v, 'meme_index': k} for k, v in MEME_INTENTS.items()}  # key by image

OLD_QUESTIONS = json.load(open('data/old_qa_pairs.json'))
PARSED_INTERPRETATIONS : Dict[str, Dict[str, Graph]] = json.load(open('data/parsed_interpretations.json'))
IMG_FOLDER = 'data/semeval_img'

# for target and sentiment id
TARGETS_SENTIMENTS_ACTIONS:Dict[str,List[dict]] = json.load(open('data/target_sentiment_action.json'))
ACTION_SENTIMENT = {
    'criticizes': 2,
    'accuses': 2,
    'mocks': 2,
    'makes fun of': 2,
    'insults': 2,
    'attacks': 2,
    'discourages': 2,
    'praises': 0,
    'stands by': 0,
    'supports': 0,
    'encourages': 0,
    'asserts': 1,
    'urges': 1,
    'suggests': 1,
    'recommends': 1,
}
GROUP_1 = ['insults','attacks','criticizes','accuses'] 
GROUP_2 = ['mocks','makes fun of'] 
GROUP_3 = ['discourages'] 
LEGAL_SAME_SENTIMENT_DISTRACTORS:Dict[str,List[str]] = {
    'praises': ['encourages'],
    'stands by': [],
    'supports': [],
    'encourages': ['praises']
}
for action,sentiment in ACTION_SENTIMENT.items():
    if sentiment == 1: continue
    if action in GROUP_1:
        LEGAL_SAME_SENTIMENT_DISTRACTORS[action] = GROUP_2 + GROUP_3
    elif action in GROUP_2:
        LEGAL_SAME_SENTIMENT_DISTRACTORS[action] = GROUP_1 + GROUP_3
    elif action in GROUP_3:
        LEGAL_SAME_SENTIMENT_DISTRACTORS[action] = GROUP_1 + GROUP_2

SIMILAR_TARGETS:Dict[str,List[str]] = json.load(open('data/dict_groups.json'))
    

def get_group(action:str) -> List[str]:
    if action in GROUP_1:
        return GROUP_1
    if action in GROUP_2:
        return GROUP_2
    if action in GROUP_3:
        return GROUP_3
    return []




class MODEL_TYPE(Enum):
    VLM = 'vlm'
    LLM = 'llm'