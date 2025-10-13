import os
from typing import Dict, List, Tuple
from src.const import GENERAL_TYPE
from src.af.utils import Question, af_init
from src.af.af_intent_comp import IntentCompAF
from src.af.discriminator import AFDiscriminator, MODEL_ID
from src.af.generator import NUM_OPTIONS, AFGenerator
from src.af.data import PARSED_INTERPRETATIONS, IMG_FOLDER, OLD_QUESTIONS, SEMEVAL_IMAGES, MEME_INTENTS, img_fn_to_id
import random

class DerivIdAF(IntentCompAF):
    def _standardize_deriv(self, s: str) -> str:
        """
        Capitalize the first letter
        """
        return s.capitalize()
        
    def _get_init_questions(self) -> Tuple[List[Question] | Dict]:
        """
        For each meme that has single intent, find the two corresponding questions.
        """
        success_questions = []
        failed_questions = []

        for img in self.img_set:
            # ignore multiple meaning memes
            if len(MEME_INTENTS[img]['intents']) > 1:
                continue
            
            img_path = os.path.join(IMG_FOLDER, img)

            # match the old questions
            matched_old_questions = [
                q
                for q in OLD_QUESTIONS
                if (
                    q['img'] == img 
                    and q['specific_type'] == 'deriv_identification2'
                )
            ]

            assert len(matched_old_questions) <= 1, f"Got {len(matched_old_questions)} matched questions for {img}"

            # convert the old questions to new format
            if len(matched_old_questions) == 1:
                q = matched_old_questions[0]
                img_id = q['img'].split('.')[0]

                q2 = Question(
                    id=f"deriv_id/{img_id}",
                    img=img_path,
                    question=q['question'],
                    options=[self._standardize_deriv(o) for o in q['options']],
                    general_type=GENERAL_TYPE.MULTIPLE.value,
                    answer_key=q['answer_key'],
                    specific_type='deriv_id',
                    method=f'symbolic',
                    correct=None, # dont know yet
                    valid=None, # dont know yet
                )
                success_questions.append(q2)
            else:
                # there is no old question, create a new one
                if img_fn_to_id(img) in PARSED_INTERPRETATIONS:
                    graph = list(PARSED_INTERPRETATIONS[img_fn_to_id(img)].values())[0]
                    derivs = [
                        node['content']
                        for node in graph.nodes
                        if node['label'][-1] == 'D'
                    ]
                else:
                    derivs = []

                NUM_OPTIONS = 4
                if len(derivs) > NUM_OPTIONS:
                    derivs = random.sample(derivs, NUM_OPTIONS)
                
                assert len(derivs) <= NUM_OPTIONS, f"Got {len(derivs)} derivations when initializing quetions for {img}"


                q2 = Question(
                    id=f"deriv_id/{img_id}",
                    img=img_path,
                    question=f"Which of the following sentence(s) can be derived from the meme?",
                    options=[self._standardize_deriv(deriv) for deriv in derivs],
                    general_type=GENERAL_TYPE.SINGLE.value,
                    answer_key=[1] * len(derivs), # requires different way to update these options (compared to SINGLE type questions)
                    specific_type='deriv_id',
                    method='empty',
                    correct=None, # dont know yet
                    valid=None, # dont know yet
                )
                failed_questions.append(q2)
        
        return success_questions, failed_questions, {'success': len(success_questions), 'fail': len(failed_questions)}

class DerivIdGenerator(AFGenerator):
    DISTRACTOR_SEPARATOR = '\n'
    MAX_NEW_TOKENS = 128
    GENERAL_TYPE = GENERAL_TYPE.MULTIPLE.value

    def __init__(self, model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> None:
        super().__init__(model_path) # register for "multi" type

    def _prompt_template(
        self, image_caption, text, q: Question
    ) -> str:
        
        avoid_previous_distractors = '\n'.join(
            [
                "Also, don't repeat the following sentences:"
            ]
            + [
                '- ' + distractor
                for distractor in q.previous_distractors
            ]    
        )
        
        num_correct = sum(q.answer_key)
        if num_correct > 0:
            correct_derivs = '\n'.join([
                f"- {deriv}"
                for deriv, key in zip(q.options, q.answer_key)
                if key == 1
            ])
        else:
            correct_derivs = "(There are no correct derivable statements)"
        num_distractors = NUM_OPTIONS - num_correct

        user_prompt = ' '.join([
            "You are given a meme as follows.",
            f"\nThe meme is composed of the following images: '{image_caption}'",
            f"\nThe meme contains the following text: '{text}'",
            f"\nSomeone thinks the following statements can be derived from the meme:",
            f"\n{correct_derivs}",
            f"\nList {num_distractors} other statements that look derivable from the meme but are actually wrong.",
            f"The new statements must have OPPOSITE or IRRELEVANT meaning from the original statements.",
            (avoid_previous_distractors if q.previous_distractors != [] else ""),
            "\nAnswer by listing each statement as a sentence on one line, and write NOTHING ELSE.",
            f"Remember, write NOTHING ELSE but the {num_distractors} new statements.",
        ])

        return user_prompt

    def _standardize_distractor(self, distractor: str, answer: str) -> str:
        """Override the method in the parent class and do nothing"""
        return distractor

if __name__ == '__main__':
    args = af_init()
    generator = DerivIdGenerator("meta-llama/Meta-Llama-3.1-8B-Instruct")
    discriminator = AFDiscriminator(MODEL_ID.QWEN.value)
    af = DerivIdAF(
        generator, discriminator, 
        img_set=SEMEVAL_IMAGES, 
        args=args
    )
    af.run()