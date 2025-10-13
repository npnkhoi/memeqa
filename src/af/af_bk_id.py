"""
Script to generate Visual Identification Questions
"""

import os
from typing import Dict, List, Tuple
from src.const import GENERAL_TYPE
from src.af.af_base import AFBase
from src.af.data import IMG_FOLDER, MEME_INTENTS, OLD_QUESTIONS
from src.af.discriminator import AFDiscriminator
from src.af.generator import NUM_OPTIONS, AFGenerator
from src.af.utils import Question, af_init
from typing_extensions import override

class AFBackgroundKnowledgeId(AFBase):
    """
    Class for generating Background Knowledge Identification Questions
    """

    QUESTION_TEXT = "Which of the following statements are factual and relevant to the meaning of the meme? Relevance is defined as the statement being necessary to understand the meme."

    @override
    def _get_init_questions(self) -> Tuple[List[Question] | Dict]:
        new_list = []

        for img in self.img_set:
            matched_old_questions: List[Dict] = [
                q
                for q in OLD_QUESTIONS
                if (
                    q['img'] == img 
                    and q['specific_type'] == 'bk_identification'
                )
            ]

            if len(matched_old_questions) == 0:
                print(f"Got {len(matched_old_questions)} matched questions for {img}")
                continue

            """
            If there are more than one questions for a meme, two things are true:
            - That meme as multiple meanings
            - The set of correct answers in those questions are likely to be identical
            Therefore, we will only consider the first question
            """
            
            q = matched_old_questions[0]
            img_id = q['img'].split('.')[0]

            options = [
                o.capitalize()
                for o in q['options']
            ]

            q2 = Question(
                id=f"bk_id/{img_id}",
                img=os.path.join(IMG_FOLDER, img),
                question=self.QUESTION_TEXT,
                options=options,
                general_type=GENERAL_TYPE.MULTIPLE.value,
                answer_key=q['answer_key'],
                specific_type='bk_id',
                method='symbolic',
                correct=None, # dont know yet
                valid=None # dont know yet
            )

            new_list.append(q2)
        
        return new_list, [], {
            'num_success': len(new_list),
            'num_failed': 0
        }
    
class BackgroundKnowledgeIdGenerator(AFGenerator):
    """
    Generator class for BK Identification
    """

    DISTRACTOR_SEPARATOR = '\n'
    MAX_NEW_TOKENS = 128
    GENERAL_TYPE = GENERAL_TYPE.MULTIPLE.value

    @override
    def _prompt_template(self, image_caption, text, q: Question) -> str:
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
            correct_derivs = "(There are no relevant facts yet)"
        
        num_distractors = NUM_OPTIONS - num_correct

        user_prompt = ' '.join([
            "You are given a meme as follows.",
            f"\nThe meme is composed of the following images: '{image_caption}'",
            f"\nThe meme contains the following text: '{text}'",
            f"\nSomeone thinks the following are relevant facts that need to be known to understand the meme:",
            f"\n{correct_derivs}",
            f"\nList {num_distractors} other statements that seem to be both factual and relevant to the meme but are actually not.",
            f"The new statements must have OPPOSITE or IRRELEVANT meaning from the original statements.",
            f"It can be a non-factual statements, or a factual statement that is not relevant to the meme.",
            (avoid_previous_distractors if q.previous_distractors != [] else ""),
            "\nAnswer by listing each statement as a sentence on one line, and write NOTHING ELSE.",
            f"Remember, write NOTHING ELSE but the {num_distractors} new statements.",
        ])

        return user_prompt

if __name__ == '__main__':
    args = af_init()
    generator = BackgroundKnowledgeIdGenerator()
    disciminator = AFDiscriminator()
    af = AFBackgroundKnowledgeId(generator, disciminator, MEME_INTENTS, args)
    af.run()