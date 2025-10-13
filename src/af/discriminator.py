from enum import Enum
import warnings
from src.const import GENERAL_TYPE, MODEL_FAMILY
from src.af.data import MEME_INTENTS, MODEL_TYPE
from src.af.utils import Color, Question, batch_iter
from src.utils import MODEL_ID, get_model_family
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoProcessor, InstructBlipForConditionalGeneration, 
    InstructBlipProcessor, Blip2ForConditionalGeneration,
    LlavaNextForConditionalGeneration, Qwen2VLForConditionalGeneration,
    AutoModelForImageTextToText, BlipForQuestionAnswering,
    LlavaForConditionalGeneration
)
import math
from typing import List, Tuple

class AFDiscriminator:
    """
    Discriminator class for Adversarial Filtering
    """
    def __init__(self, model_path: str=MODEL_ID.QWEN.value, batch_size: int=4, gpu_id: int=1) -> None:
        """
        """
        self.max_new_tokens = 4 # max number of new tokens to generate, for multiple choice questions
        self.batch_size = batch_size
        self.gpu_id = gpu_id

        # model lazy loading
        self.model = None
        self.model_loaded = False
        self.model_path = model_path

        # load processor  
        if get_model_family(self.model_path) == MODEL_FAMILY.QWEN:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels = 256*28*28,
                max_pixels = 480*28*28
            )
        else:
            # BLIP, LLAVA, INSTRUCT_BLIP share a common processor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            if get_model_family(self.model_path) == MODEL_FAMILY.LLAVA:
                self.processor.tokenizer.padding_side = "left"
        
        
    
    def _load_model(self):
        if self.model_loaded:
            return
        
        self.model_loaded = True

        device_map = {"": self.gpu_id} if self.gpu_id != -1 else 'auto'
        if get_model_family(self.model_path) == MODEL_FAMILY.INSTRUCT_BLIP:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                self.model_path, load_in_8bit=True, device_map=device_map
            )
        elif get_model_family(self.model_path) == MODEL_FAMILY.BLIP2:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_path, load_in_8bit=True, device_map=device_map
            )
        
        elif get_model_family(self.model_path) == MODEL_FAMILY.LLAVA:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_path, load_in_8bit=True, device_map=device_map
            )
        elif get_model_family(self.model_path) == MODEL_FAMILY.QWEN:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path, load_in_8bit=True, device_map=device_map,
            )
        else:
            warnings.warn('Using a generic model classs')
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, device_map=device_map)
        
        # elif self.model_path == MODEL_ID.BLIP_VQA.value:
        #     warnings.warn('Using a generic model classs')
        #     self.model = BlipForQuestionAnswering.from_pretrained(self.model_path, device_map=device_map)
        #     self.processor = AutoProcessor.from_pretrained(self.model_path)
        #     self.processor.tokenizer.padding_side = "left"
        #     self.max_new_tokens = 10 # HACK?
        # elif self.model_path == MODEL_ID.COGVLM.value:
        #     warnings.warn('Not working yet!')
        #     print(f"{Color.RED}loading Cog{Color.END}")
        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         self.model_path, device_map=device_map,
        #         trust_remote_code=True
        #     )
        #     self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        self.model.eval()
        
        first_param = next(self.model.parameters())
        self.device = first_param.device

    def _get_discriminator_prompt(self, q: Question, model_path: str=MODEL_ID.LLAVA_v16.value, model_type: MODEL_TYPE=MODEL_TYPE.VLM) -> str:
        """
        q is a dictionary with the following keys: 'img', 'question', 'options', 'general_type'.
        For LLM case, hints will be retrieved based on the img file.
        """
        q.options = q.options[:4]
        if q.specific_type != "condiional_sentiment_id":
            assert len(q.options) == 4


        task_instruction = "You are given a meme.\n"
        
        if model_type == MODEL_TYPE.LLM:
            raise NotImplementedError
            img_fn = q.img.split('/')[-1]
            image_caption = MEME_INTENTS[img_fn]['image_caption']
            text = MEME_INTENTS[img_fn]['text']
            bk = MEME_INTENTS[img_fn]['bks']
            task_instruction += (
                "The meme is composed of the following images: " + image_caption + '\n' +
                "The meme contains the following text: " + text + '\n' +
                "The meme involves the following background knowledge:\n" +
                bk + '\n'
            )
        
        if q.general_type == 'single':
            task_instruction += (
                "Answer the following question by writing ONLY one letter A, B, C, or D. " +
                "DO NOT write anything else. ONLY write the letter of the correct answer."
            )
        else:
            task_instruction += (
                "Answer the following question by selecting ALL the correct options and write their letters consecutively, such as 'ACD' or 'B'. " +
                "Write 'N' if none of the options are correct. " +
                "DO NOT write anything thing else. " +
                "ONLY write the letters of the correct answers or 'N'. " +
                "Remember that you can select multiple options."
            )

        OPTION_LETTERS = [
            '(A) ',
            '(B) ',
            '(C) ',
            '(D) '
        ]
        prompt = (
            task_instruction + "\n" +
            '## Question: ' + q.question + '\n' +
            '## Options:\n'
        )

        for OL,_option in zip(OPTION_LETTERS,q.options):
            prompt+= OL + _option + '\n'
        prompt+='## Answer: '
        
        if model_path in [MODEL_ID.LLAVA_v16.value, MODEL_ID.LLAVA_v15.value, MODEL_ID.QWEN.value]:
            # LLAVA_SYSTEM_PROMPT = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""
            # prompt = (
            #     f"{LLAVA_SYSTEM_PROMPT} " +
            #     f"USER: <image>\n{prompt}\n"
            #     f"ASSISTANT: "
            # )
            conv = [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image'
                        },
                        {
                            'type': 'text', 'text': prompt
                        }
                    ]
                }
            ]
            prompt = self.processor.apply_chat_template(
                conv,
                add_generation_prompt = True
            )

        # print(f"{Color.GREEN}{q}{Color.END}")
        # print(f"{Color.YELLOW}{prompt}{Color.END}")
        # input('press enter to continue')

        return prompt

    def _format_output(self, out, q_type: str):
        # Format the answer
        POSSIBLE_ANSWERS = ['A', 'B', 'C', 'D']
        assert 'single' == GENERAL_TYPE.SINGLE
        if q_type == GENERAL_TYPE.SINGLE:
            # Single answer
            # strict logic: only take the first letter, and that letter must be ABCD
            if out[0] not in POSSIBLE_ANSWERS:
                warnings.warn(f"Invalid output: {out}")
                return None
            else:
                return ord(out[0]) - ord('A')
        else:
            # Multiple answers
            if out == 'N':
                return [0, 0, 0, 0]
            else:
                formatted_output = [0, 0, 0, 0]
                for a in out:
                    # very strict logic here
                    if a not in POSSIBLE_ANSWERS:
                        return None
                    formatted_output[ord(a) - ord('A')] = 1
                return formatted_output
    
    def _answer_batch(self, question_batch: List[Question]) -> List[Question]:
        """
        Update or create the `output` field of each Question
        """
        prompts = [self._get_discriminator_prompt(q, self.model_path) for q in question_batch]
        images = [Image.open(q.img).convert('RGB') for q in question_batch]

        inputs = self.processor(images=images, text=prompts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # for k,v in inputs.items():
        #     print(f"{k} : {v.shape}")
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        
        generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)

        # trim the prefix for certain models
        if self.model_path in [MODEL_ID.LLAVA_v16.value, MODEL_ID.LLAVA_v15.value, MODEL_ID.QWEN.value, MODEL_ID.BLIP_VQA.value]:
            textual_inputs = self.processor.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            generated_texts = [text[len(input):] for text, input in zip(generated_texts, textual_inputs)]
        # remove '(' for qwen
        if self.model_path == MODEL_ID.QWEN.value:
            generated_texts = [text.replace('(', '') for text in generated_texts]
        elif self.model_path == MODEL_ID.LLAVA_v16.value:
            generated_texts = [text.replace(' ', '') for text in generated_texts]
        elif self.model_path == MODEL_ID.LLAVA_v15.value:
            generated_texts = [text.replace('ASSISTANT:', '').strip() for text in generated_texts]
        
        # print(generated_texts)
        # breakpoint()

        for i in range(len(question_batch)):
            # weird bug in an earlier state: https://github.com/npnkhoi/cmu/commit/38886a00f3437f67dc5773c01207ab3858e784f8#diff-e1eb90f5cb135ecfd01ad1bb6533e1e05abfd39101aa4051e23c9b97ea9baa97R214
            # it may affect the validity of deriv-comp questions and evaluation results
            question_batch[i].output = self._format_output(generated_texts[i], question_batch[i].general_type)
        
        return question_batch

    def discriminate(self, questions: List[Question]) -> Tuple[List[Question], List[Question], List[Question]]:
        """
        Answer the questions. Returns new_success_list, new_failed_list, new_invalid_discrimination_list
        

        Args:
            questions (List[Question]): _description_

        Returns:
            Tuple[List[Question], List[Question], List[Question]]:
                 new_success_list: list of questions the model guessed wrong (in other words, they are good questions)
                 new_failed_list: list of questions the model guessed right (in other words, they are too easy)
                 new_invalid_discrimination_list: list of questions where the model failed to produce output (toss these out)
        """
        self._load_model()

        new_wrong_list = []
        new_correct_list = []
        new_invalid_discrimination_list = []
        for question_batch in tqdm(batch_iter(questions, self.batch_size), total=math.ceil(len(questions) / self.batch_size)):
            question_batch = self._answer_batch(question_batch)
            for q in question_batch:
                if q.output is None:
                    new_invalid_discrimination_list.append(q)
                elif q.output == q.answer_key:
                    new_correct_list.append(q)
                else:
                    new_wrong_list.append(q)

        # invalid list: model didn't produce output
        # correct list: model guess the correct answer
        # wrong list: model guess wrong. the question is good 
        return new_wrong_list, new_correct_list, new_invalid_discrimination_list
