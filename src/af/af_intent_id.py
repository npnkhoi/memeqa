import os
from typing import Dict, List, Tuple
from typing_extensions import override
from src.const import GENERAL_TYPE
from src.af.af_base import AFBase
from src.af.generator import NUM_DISTRACTORS, AFGenerator
from src.af.utils import MyArguments, Question
from src.af.discriminator import AFDiscriminator
from src.af.data import IMG_FOLDER, OLD_QUESTIONS, SEMEVAL_IMAGES, MEME_INTENTS
from transformers import HfArgumentParser
import wandb

from src.utils import MODEL_ID

class IntentIdAF(AFBase):
    def _standardize_intent(self, intent: str) -> str:
        return intent.capitalize()

    @override
    def _get_init_questions(self) -> Tuple[List[Question], List[Question], Dict]:
        """
        For each meme that has single intent, find the two corresponding questions.

        To see types of returned values, see the parent class.
        """
        success_questions = []
        failed_questions = []

        for img in self.img_set:
            intents = MEME_INTENTS[img]['intents']
            # ignore multiple meaning memes
            if len(intents) > 1:
                continue
            
            intent = intents[0]
            img_path = os.path.join(IMG_FOLDER, img)

            # match the old questions
            matched_old_questions = [
                q
                for q in OLD_QUESTIONS
                if (
                    q['img'] == img 
                    and q['specific_type'] in ['intent_identification1', 'intent_identification2']
                )
            ]

            # convert the old questions to new format
            for q in matched_old_questions:
                img_id = q['img'].split('.')[0]

                which_old_method = q['specific_type'][-1]
                q2 = Question(
                    id=f"intent_id/{img_id}",
                    img=img_path,
                    question=q['question'],
                    options=[self._standardize_intent(o) for o in q['options']],
                    general_type=GENERAL_TYPE.SINGLE.value,
                    answer_key=q['answer_key'],
                    specific_type='intent_id',
                    method=f'symbolic_{which_old_method}',
                    correct=None, # dont know yet
                    valid=None, # dont know yet
                )
                success_questions.append(q2)
            
            # if there is no old question, create a new one
            if len(matched_old_questions) == 0:
                q2 = Question(
                    id=f"intent_id/{img_id}",
                    img=img_path,
                    question=f"What is the intent of the meme?",
                    options=[self._standardize_intent(intent)],
                    general_type=GENERAL_TYPE.SINGLE.value,
                    answer_key=0,
                    specific_type='intent_id',
                    method='empty',
                    correct=None, # dont know yet
                    valid=None, # dont know yet
                )
                failed_questions.append(q2)
        
        return success_questions, failed_questions, {'success': len(success_questions), 'fail': len(failed_questions)}

def initialize():
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


class IntentIdGenerator(AFGenerator):
    DISTRACTOR_SEPARATOR = '\n'
    MAX_NEW_TOKENS = 128
    GENERAL_TYPE = GENERAL_TYPE.SINGLE.value

    @override
    def _prompt_template(
        self, image_caption, text, q: Question
    ) -> str:

        avoid_previous_distractors = '\n'.join(
            [
                "Also, don't repeat the following intents:"
            ]
            + [
                '- ' + distractor
                for distractor in q.previous_distractors
            ]
        )

        intent = q.options[q.answer_key]

        user_prompt = ' '.join([
            "You are given a meme as follows.",
            f"\nThe meme is composed of the following images: {image_caption}",
            f"\nThe meme contains the following text: {text}",
            f"\nSomeone thinks the meme's intent is that '{intent}'.",
            f"\nList {NUM_DISTRACTORS} other possible intents of the meme.",
            f"\nThe new intents must have OPPOSITE or IRRELEVANT meaning from the original intent.",
            (avoid_previous_distractors if q.previous_distractors != [] else ""),
            "\nAnswer by listing each intent on one line, and write NOTHING ELSE.",
            f"Remember, write NOTHING ELSE but the {NUM_DISTRACTORS} new sentences.",
        ])

        return user_prompt

    @override
    def _standardize_distractor(self, distractor: str, answer: str) -> str:
        """do nothing"""
        return distractor

if __name__ == '__main__':
    args = initialize()
    generator = IntentIdGenerator("meta-llama/Meta-Llama-3.1-8B-Instruct")
    discriminator = AFDiscriminator(MODEL_ID.QWEN.value)
    af = IntentIdAF(
        generator, discriminator, 
        img_set=SEMEVAL_IMAGES, 
        args=args
    )
    af.run()