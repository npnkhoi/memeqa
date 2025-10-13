"""
Adversarial Filtering for deriv completion questions.
"""

import os
import random
from typing import Dict, List
from typing_extensions import override
import warnings
from src.af.af_base import AFBase
from src.af.af_intent_comp import CompGenerator, IntentCompAF
from src.af.discriminator import AFDiscriminator
from src.af.utils import MyArguments, Question
from src.af.data import OLD_QUESTIONS
from src.af.data import IMG_FOLDER, SEMEVAL_IMAGES, img_fn_to_id, PARSED_INTERPRETATIONS
from transformers import HfArgumentParser
import wandb

from src.utils import MODEL_ID

class DerivCompAF(IntentCompAF):
    QUESTION_TEXT = 'Fill in the blank to complete a sentence that can be derived from the meme: {}'

    def _get_old_distractors(self, img: str, masked_sentence: str, mask: Dict) -> List[str]:
        """
        We try finding the old distractors for this question by matching
        (1) the masked sentence and (2) the masked phrase

        The identifier for a deriv-comp question:
        (1) the node id of the masked sentence
        (2) the boundary indices of the masked phrase
        """
        matched_qas = [
            qa
            for qa in OLD_QUESTIONS
            if qa['img'] == img and qa['specific_type'] == 'deriv_completion' \
                and mask['text'] == qa['options'][qa['answer_key']] \
                and qa['question'].split(': ')[1] == masked_sentence
        ]

        if len(matched_qas) == 0:
            return []

        if len(matched_qas) > 1:
            warnings.warn(f"Multiple matched questions found for img {img}, masked sentence {masked_sentence}, mask {mask}")
        
        qa = matched_qas[0]
        distractors = [qa['options'][i] for i in range(4) if i != qa['answer_key']]
        return distractors
    
    @override
    def _get_init_questions(self):
        # for each img in img_set, get all the derivation sentences
        # and locate the meaning spans. Then, first find old question for it.
        # if not, generate new question.

        stats = {
            'success': 0,
            'fail': 0,
        }
        all_init_questions = []
        for img in self.img_set:
            img_id = img_fn_to_id(img)
            if img_id not in PARSED_INTERPRETATIONS:
                warnings.warn(f"Graph not found for img {img}")
                continue
            
            has_multiple_meanings = len(PARSED_INTERPRETATIONS[img_id]) > 1
            if has_multiple_meanings:
                continue
            
            graph = list(PARSED_INTERPRETATIONS[img_id].values())[0]

            for node in graph['segments'][:-1]:
                if node['label'][-1] != 'D':
                    continue
                
                # Now we got the base sentence in node['content']
                # We need to create a question for each meaningful span
                base_sentence = node['content'].strip()
                masks = self._meaningful_spans(base_sentence)

                for mask in masks:
                    masked_sentence = base_sentence[:mask['start']] + '____' + base_sentence[mask['end']:]
                    distractors = self._get_old_distractors(img, masked_sentence, mask) # can be empty!
                    if len(distractors) == 0:
                        stats['fail'] += 1
                    else:
                        stats['success'] += 1
                    options = distractors + [mask['text']]
                    random.shuffle(options)
                    full_question = self.QUESTION_TEXT.format(masked_sentence)
                    
                    q = Question(
                        img=os.path.join(IMG_FOLDER, img),
                        question=full_question,
                        options=options,
                        general_type='single',
                        answer_key=options.index(mask['text']),
                        specific_type='intent_completion',
                        mask=mask,
                        method='symbolic_old',
                        masked_sentence=masked_sentence,
                        previous_distractors=[],
                        active=True, # whether this question is still actively being renewed
                    ) 
                    all_init_questions.append(q)

        return all_init_questions, [], stats
    
if __name__ == '__main__':
    parser = HfArgumentParser((MyArguments))
    args: MyArguments = parser.parse_args_into_dataclasses()[0]
    wandb.init(
        project='memeqa',
        group='af',
        entity='khoi-ml',
        name=args.run_name,
    )
    args.run_id = wandb.run.id

    img_set = SEMEVAL_IMAGES[:4]
    
    generator = CompGenerator("meta-llama/Meta-Llama-3.1-8B-Instruct")
    discriminator = AFDiscriminator(MODEL_ID.QWEN.value)
    af = DerivCompAF(
        generator, discriminator, 
        img_set=img_set, args=args
    )
    af.run()