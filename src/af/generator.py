from abc import abstractmethod
from src.const import GENERAL_TYPE
from src.af.data import MEME_INTENTS
from src.af.utils import Question, batch_iter
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM
import math
import random
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple


BATCH_SIZE = 4
NUM_DISTRACTORS = 3
NUM_OPTIONS = 4

class AFGenerator:
    """
    Generator class for Adversarial Filtering
    """
    
    # WARNING: must update these in subclasses!!!
    MAX_NEW_TOKENS = 20
    DISTRACTOR_SEPARATOR = ','
    GENERAL_TYPE = None

    def __init__(self, model_path: str="meta-llama/Meta-Llama-3.1-8B-Instruct") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = '<|finetune_right_pad_id|>'
        self.tokenizer.padding_side = 'left'
        
        # lazy loading
        self.model_path = model_path
        self.model_loaded = False
        self.model = None
        
        self.num_distractors = NUM_DISTRACTORS
    
    def _load_model(self):
        if self.model_loaded:
            return
        
        self.model_loaded = True
        self.model = LlamaForCausalLM.from_pretrained(self.model_path, device_map={"": 0}, load_in_8bit=True)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        

    @abstractmethod
    def _prompt_template(self, image_caption, text, q: Question) -> str:
        raise NotImplementedError

    def _get_prompt_for_generator(self, q: Question) -> List[Dict]:
        """
        Returns a prompt for the generator model to generate distractors for the question.
        
        For LLM case, hints will be retrieved based on the img file.
        """

        q = deepcopy(q)

        img_fn = q.img.split('/')[-1]

        image_caption = MEME_INTENTS[img_fn]['image_caption']
        text = MEME_INTENTS[img_fn]['text']
        # bk = memeintents[img_for_hints]['bks']

        user_prompt = self._prompt_template(image_caption, text, q)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ]

        return messages

    def _standardize_distractor(self, distractor: str, answer: str) -> str:
        """
        Standardize the distractor to look like the answer.
        """
        distractor = distractor.strip('., ')
        if answer[0].isupper():
            distractor = ' '.join([word.capitalize() for word in distractor.split(' ')])
        else:
            distractor = distractor.lower()
        return distractor
    
    def _clear_distractors(self, q: Question) -> Question:
        """
        Clear the distractors of the question
        """
        if self.GENERAL_TYPE == GENERAL_TYPE.SINGLE.value:
            q.previous_distractors += [opt for opt in q.options if opt != q.options[q.answer_key]]
            q.options = [q.options[q.answer_key]]
            q.answer_key = 0
        else:
            q.previous_distractors += [opt for opt, key in zip(q.options, q.answer_key) if key == 0]
            q.options = [opt for opt, key in zip(q.options, q.answer_key) if key == 1]
            q.answer_key = [1] * len(q.options)
        
        return q
    
    def _add_distractors(self, q: Question, new_distractors: List[str]) -> Question:
        """
        Add new distractors by updating the options and answer_key of the question.
        Do a bit of cleaning on the distractors.
        """
        new_distractors = [d for d in new_distractors if d.strip() != '']
        if self.GENERAL_TYPE == GENERAL_TYPE.SINGLE.value:
            # standardize the distractors
            new_distractors = [
                self._standardize_distractor(d, q.options[q.answer_key])
                for d in new_distractors
            ]

            # update the question
            assert len(q.options) == 1
            answer = q.options[0]
            q.options += new_distractors
            random.shuffle(q.options) # REMEMBER TO ALWAYS SHUFFLE THE OPTIONS
            q.answer_key = q.options.index(answer)
            q.method = 'llm'
        else:
            assert all([key == 1 for key in q.answer_key])

            q.options += new_distractors
            random.shuffle(q.options) # REMEMBER TO ALWAYS SHUFFLE THE OPTIONS
            q.answer_key = [
                0 if opt in new_distractors else 1
                for opt in q.options
            ]
            q.method = 'llm'
        
        return q

    def generate(self, questions: List[Question]) -> Tuple[List[Dict]]:
        """
        Generate new questions. Returns new_list, inactive_list
        """
        self._load_model()
        new_list = []
        inactive_list = []
        for batch in tqdm(batch_iter(questions, BATCH_SIZE), total=math.ceil(len(questions) / BATCH_SIZE)):
            questions = batch
            
            # move the old distractors to the previous distractors list
            for q in questions:
                q = self._clear_distractors(q)
            
            prompts = [self._get_prompt_for_generator(q) for q in questions]

            inputs = self.tokenizer.apply_chat_template(prompts, return_tensors='pt', padding=True, truncation=True, return_dict=True).to(self.model.device)
            textual_inputs = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKENS)

            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [text[len(input):] for text, input in zip(generated_texts, textual_inputs)] # removes prompt string

            # Removes assistant tag from string
            for q, gen_text in zip(questions, generated_texts):
                if 'assistant' not in gen_text:
                    warnings.warn(f'Assistant not found in the generated text: {gen_text}')
                else:
                    cleaned_gen_text = gen_text.replace('assistant\n\n', '')
                new_distractors = cleaned_gen_text.split(self.DISTRACTOR_SEPARATOR)


                q = self._add_distractors(q, new_distractors)

                # This may be too strict. > 4 options may be okay, we can just ignore the extra distractors.
                if len(q.options) != NUM_OPTIONS:
                    inactive_list.append(q)
                else:
                    new_list.append(q)
        
        # inactive lists, means the LLM didn't produce more distractors
        return new_list, inactive_list

    
