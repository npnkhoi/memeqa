from src.const import GENERAL_TYPE, MODEL_FAMILY
from src.prompt import PromptGenerator
from src.utils import Question
from src.utils import get_model_family, get_is_enc_dec
import torch
from PIL import Image
from typing import Dict, List, Tuple


class Collator:
    def __init__(self, processor, tokenizer, prompt_id: str, model_id: str) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.prompt_id = prompt_id
        self.task = prompt_id.split('/')[0]
        self.model_enc_dec = get_is_enc_dec(model_id)

        if get_model_family(model_id) == MODEL_FAMILY.LLAMA:
            # llama does not have pad_token, so we set it to eos_token, should be fine
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_str_answer(self, example: Question) -> str:
        """
        For 'single' type, return something like "A", "C"
        For multi type, return the string of max 4.
        """
        if example.general_type == GENERAL_TYPE.SINGLE.value:
            # NOTE: for single type, there can be only three options (cond-sent-id)
            assert isinstance(example.answer_key, int)
            return "ABCD"[example.answer_key]
        elif example.general_type == GENERAL_TYPE.MULTIPLE.value:
            assert isinstance(example.answer_key, list)
            tmp = ''.join(["ABCD"[i] for i in range(4) if example.answer_key[i]])
            ans = tmp if len(tmp) > 0 else "N"
            return ans

    def get_ground_truth(self, task, example):
        if task == 'QA':
            return self._get_str_answer(example)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _get_basic_batch(self, examples, is_train: bool) -> Tuple[Dict, List[str]]:
        """
        Returns a basic batch (Dict) and a list of ground truths.

        The basic batch is a dictionary with keys:
        - input_ids: if training, containing the ground truth
        - attention_mask
        - pixel_values (only for llava)
        """
        images = []
        texts = []
        ground_truths = []
        for example in examples:
            ground_truth = self.get_ground_truth(self.task, example)

            # Get the prompt depending on the model
            if get_model_family(self.model_id) == MODEL_FAMILY.LLAMA:
                conv = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": PromptGenerator.get_user_prompt(self.prompt_id, example)},
                    {'role': 'assistant', 'content': ground_truth + self.tokenizer.eos_token if is_train else ''}
                ]
                prompt = self.processor.apply_chat_template(
                    conv,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                # VLM
                image = Image.open(example.img).convert('RGB')
                images.append(image)

                if get_model_family(self.model_id) in [MODEL_FAMILY.BLIP2, MODEL_FAMILY.INSTRUCT_BLIP]:
                    prompt = (
                        f"{PromptGenerator.get_user_prompt(self.prompt_id, example)}\n" +
                        f"{ground_truth + self.tokenizer.eos_token if (is_train and not self.model_enc_dec) else ''}"
                    )
                elif get_model_family(self.model_id) == MODEL_FAMILY.QWEN:
                    prompt = (
                        f"{PromptGenerator.get_user_prompt(self.prompt_id, example)}\n" +
                        f"{ground_truth + self.tokenizer.eos_token if is_train else ''}"
                    )
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
                else:
                    # llava
                    prompt = (
                        f"{PromptGenerator.LLAVA_SYSTEM_PROMPT} " +
                        f"USER: <image>\n{PromptGenerator.get_user_prompt(self.prompt_id, example)}\n" +
                        f"ASSISTANT: {ground_truth + self.tokenizer.eos_token if is_train else ''}"
                    )

            texts.append(prompt)
            ground_truths.append(ground_truth)


        # Tokenize
        if get_model_family(self.model_id) == MODEL_FAMILY.LLAMA:
            batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        elif get_model_family(self.model_id) == MODEL_FAMILY.BLIP2:
            if is_train:
                # Following https://github.com/huggingface/notebooks/blob/main/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb
                pixel_values = torch.stack([
                    self.processor(
                        images=[image],
                        padding="max_length",
                        return_tensors="pt",
                    )['pixel_values'].squeeze()
                    for image in images
                ])
                text_inputs = self.processor.tokenizer(
                    texts,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = text_inputs["input_ids"]
                attention_mask = text_inputs["attention_mask"]
                batch = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                }
            else:
                # the tutorial is about image captioning which doesn't have text input
                # for us, we just need to to this
                batch = self.processor(text=texts, images=images, padding=True, truncation=True, return_tensors="pt")
        elif get_model_family(self.model_id) == MODEL_FAMILY.INSTRUCT_BLIP:
            batch = self.processor(text=texts, images=images, padding=True, truncation=True, return_tensors="pt")
            # breakpoint()
        else:
            # probably llava
            batch = self.processor(text=texts, images=images, padding=True, truncation=True, return_tensors="pt")

        return batch, ground_truths

    def train_collate_fn(self, examples):
        batch, gts = self._get_basic_batch(examples, is_train=True)

        # Add `labels` field to the batch
        # copied from tutorials, but not sure why inputs and and labels are the same
        # maybe internally they do shifting to train on next tok predicion already?
        # => Yes
        if not self.model_enc_dec:
            labels = batch["input_ids"].clone()
        else:
            labels = self.processor.tokenizer(
                    gts,
                    padding=True,
                    return_tensors="pt",
                )['input_ids']

        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

    def eval_collate_fn(self, examples: List[Question]):
        batch, answers = self._get_basic_batch(examples, is_train=False)

        # Add `answers` and `image_paths` fields to the batch
        q_ids = [example.id for example in examples]
        batch['answers'] = answers
        batch['example_ids'] = q_ids

        return batch
