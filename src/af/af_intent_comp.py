"""
Adversarial Filtering
ready to be batchified.
Only for intent completion questions.
"""

from accelerate import infer_auto_device_map, dispatch_model
from transformers import (
    HfArgumentParser
)
# import argparse
import os
import json
from typing import Dict, List, Tuple, Any
from src.const import GENERAL_TYPE
from src.af.discriminator import AFDiscriminator
from src.af.generator import NUM_DISTRACTORS, AFGenerator
from src.af.utils import MyArguments, Question
from src.af.data import IMG_FOLDER
from src.af.data import SEMEVAL_IMAGES, MEME_INTENTS, OLD_QUESTIONS
import spacy
import random
import wandb
# import datetime
# from torch.utils.data import Dataset, DataLoader
from src.af.af_base import AFBase
from typing_extensions import override

from src.utils import MODEL_ID

QUESTION_TEXT = 'Fill in the blank to complete the intent of the meme: {}'

class IntentCompAF(AFBase):
    @override
    def _get_init_questions(self):

        all_init_questions = []
        for img in self.img_set:
            memeintent_annotations = MEME_INTENTS[img]
            intents = memeintent_annotations['intents']
            if len(intents) > 1: # multiple intents, not applicable
                continue
            

            intent = intents[0]
            
            # standardize the intent
            assert intent.startswith('the meme')
            intent = 'T' + intent[1:]

            #  find spots to create blanks
            masks = self._meaningful_spans(intent, token_types=['VERB', 'ADJ', 'ADV'])
            
            for mask in masks:
                masked_intent = intent[:mask['start']] + '____' + intent[mask['end']:]
                
                distractors = self._get_old_distractors(img, mask)
                
                options = [mask['text']] + distractors
                # REMEMBER TO ALWAYS SHUFFLE THE OPTIONS
                random.shuffle(options)
                full_question = QUESTION_TEXT.format(masked_intent)
                q = Question(
                    img=os.path.join(IMG_FOLDER, img),
                    question=full_question,
                    options=options,
                    general_type='single',
                    answer_key=options.index(mask['text']),
                    specific_type='intent_completion',
                    mask=mask,
                    method='symbolic',
                    masked_sentence=masked_intent,
                    previous_distractors=[],
                    active=True, # whether this question is still actively being renewed
                ) 
                all_init_questions.append(q)
            
        return all_init_questions, [], None
    
    def _good_for_masking(self, s):
        # check if a word is a good one for masking
        return s[0].isalpha() and s.lower() not in ['meme', 'image', 'text', 'the meme']
    
    def _meaningful_spans(self, sentence: str, token_types: List[str]=['VERB', 'ADJ', 'ADV']) -> List[Dict]:
        """
        Returns a list of meaningful spans (start, end, text, type) in the sentence.
        Each span will corespond to a question.
        """
        doc = self.nlp(sentence)
        spans = []
        span_index = set() # to avoid duplicate spans

        def try_add_span(start, end, text, type):
            if (start, end) in span_index:
                return
            spans.append({
                'start': start,
                'end': end,
                'text': text,
                'type': type
            })
            span_index.add((start, end))

        # verb/adj/adv
        for token in doc:
            if token.pos_ in token_types and self._good_for_masking(token.text):
                start = token.idx
                end = start + len(token.text)
                try_add_span(start, end, token.text, 'token/' + token.pos_)
            
        # named entities
        for ent in doc.ents:
            if self._good_for_masking(ent.text):
                try_add_span(ent.start_char, ent.end_char, ent.text, 'ne/' + ent.label_)

        # noun phrases
        for chunk in doc.noun_chunks:
            # prioritize NE than noun chunks
            if self._good_for_masking(chunk.text):
                try_add_span(chunk.start_char, chunk.end_char, chunk.text, 'chunk/NOUN')

        return spans

    def _get_old_distractors(self, img: str, mask: Dict) -> List[str]:
        """
        Returns a list of distractors for the question based on the OLD_QUESTIONS.
        """
        matched_qas = [
            qa
            for qa in OLD_QUESTIONS
            if qa['img'] == img and qa['specific_type'] == 'intent_completion' \
                and mask['text'] == qa['options'][qa['answer_key']]
        ]
        if len(matched_qas) == 0:
            # TODO: remove this behavior
            return ['foo', 'bar', 'baz']

        qa = matched_qas[0]
        distractors = [qa['options'][i] for i in range(4) if i != qa['answer_key']]
        return distractors

class CompGenerator(AFGenerator):
    MAX_NEW_TOKENS = 20
    DISTRACTOR_SEPARATOR = ','
    GENERAL_TYPE = GENERAL_TYPE.SINGLE.value

    @override
    def _prompt_template(self, image_caption, text, q: Question) -> str:
        avoid_previous_distractors = (
            "Also, don't use the following words or phrases: " + ', '.join(q.previous_distractors)
        )
        masked_sentence = q.question.split(': ')[1]

        # WARNING: This prompt is specific to the intent completion task
        user_prompt = ' '.join([
            "You are given a meme as follows.",
            f"\nThe meme is composed of the following images: {image_caption}",
            f"\nThe meme contains the following text: {text}",
            f"\nList {NUM_DISTRACTORS} words or phrases that are the most sensible to be filled in the blank of the following sentence: '{masked_sentence}'.",
            f"\nThe words or phrases must have OPPOSITE or IRRELEVANT meaning from '{q.options[q.answer_key]}'.",
            (avoid_previous_distractors if q.previous_distractors != [] else ""),
            "Answer by listing the words or phrases separated by commas, and write NOTHING ELSE.",
            f"Remember, write NOTHING ELSE but the {NUM_DISTRACTORS} things.",
        ]
        )

        return user_prompt

if __name__ == "__main__":

    parser = HfArgumentParser((MyArguments))
    args: MyArguments = parser.parse_args_into_dataclasses()[0]
    wandb.init(
        project='memeqa',
        group='af',
        entity='khoi-ml',
        name=args.run_name,
    )
    args.run_id = wandb.run.id
    
    generator = CompGenerator("meta-llama/Meta-Llama-3.1-8B-Instruct")
    discriminator = AFDiscriminator(MODEL_ID.QWEN.value)
    
    img_set = SEMEVAL_IMAGES
    af = IntentCompAF(
        generator, discriminator, 
        img_set=img_set, args=args
    )
    af.run()