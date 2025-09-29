from enum import Enum
from src.utils import Question


class PromptGenerator:
    LLAVA_SYSTEM_PROMPT = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

    @staticmethod
    def get_user_prompt(prompt_id: str, q: Question) -> str:
        task_instruction = "You are given a meme.\n"

        if q.general_type == 'single':
            task_instruction += (
                "Answer the following question by writing ONLY one letter A, B, C, or D. " +
                "DO NOT write anything else. ONLY write the letter of the correct answer."
            )
        else:
            if prompt_id == PromptId.QA_NONEABOVE.value:
                task_instruction += (
                    "Answer the following question by selecting ALL the correct options and write their letters consecutively in alphabetical order, such as 'ACD' or 'B'. " +
                    "DO NOT write anything thing else. " +
                    "ONLY write the letters of the correct answers. " +
                    "Remember that you can select multiple options."
                )
            else:
                task_instruction += (
                    "Answer the following question by selecting ALL the correct options and write their letters consecutively in alphabetical order, such as 'ACD' or 'B'. " +
                    "Write 'N' if none of the options are correct. " +
                    "DO NOT write anything thing else. " +
                    "ONLY write the letters of the correct answers or 'N'. " +
                    "Remember that you can select multiple options."
                )

        prompt = (
            task_instruction + "\n" +
            '## Question: ' + q.question + '\n' +
            '## Options:\n' +
            '(A) ' + q.options[0] + '\n' +
            '(B) ' + q.options[1] + '\n' +
            '(C) ' + q.options[2] + '\n' +
            ('(D) ' + q.options[3] + '\n' if len(q.options) == 4 else '') +
            '## Answer: '
        )
        return prompt


###################### Prompt ######################

class PromptId(Enum):
    # General syntax: {task}/{sub_prompt_id}
    QA = "QA/none"
    QA_NONEABOVE = "QA/none_noneabove"