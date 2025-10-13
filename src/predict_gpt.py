import argparse
import json
import os
import random
from dotenv import load_dotenv
import openai
from tqdm import tqdm
from src.collator import Collator
from src.dataset import IMG_DIR
from src.qset import QuestionSet
from src.prompt import PromptGenerator, PromptId
import base64
import wandb

MAX_COMPLETION_TOKENS = 750
SAVING_INTERVAL = 1

class CollatorForChatGPT(Collator):
    # override
    def _get_basic_batch(self, examples, is_train: bool) -> dict:
        """
        Returns a basic batch (Dict) and a list of ground truths.

        The basic batch is a dictionary with keys:
        - texts
        - images
        - ground truths
        """
        images = []
        texts = []
        ground_truths = []
        qids = []
        for example in examples:
            ground_truth = self.get_ground_truth(self.task, example)
            
            # taken from formatting code for QWEN
            prompt = (
                f"{PromptGenerator.get_user_prompt(self.prompt_id, example)}\n" +
                f"{ground_truth + self.tokenizer.eos_token if is_train else ''}"
            )
            img_path = os.path.join(IMG_DIR, example.img.split('/')[-1])
            
            qids.append(example.id)
            images.append(img_path)
            texts.append(prompt)
            ground_truths.append(ground_truth)


        return {
            'qids': qids,
            'image_paths': images,
            'texts': texts,
            'ground_truths': ground_truths
        }

def predict(model, dataset_version, output_fn):
    wandb.init(
        project='memeqa-inference',
    )
    
    # Example usage:
    # questions = [
    #     {"image_path": "question1.png", "question_text": "Which option best describes the image? (Choose one: A, B, C, or D)"},
    #     {"image_path": "question2.png", "question_text": "Which of the following are present in the image? (Choose multiple: A, B, C, D, or N for none)"},
    # ]

    assert dataset_version in ['none_plus', 'none_minus'], f"Invalid version: {dataset_version}"
    prompt_id = PromptId.QA_NONEABOVE.value if dataset_version == 'none_plus' else PromptId.QA.value

    qset = QuestionSet(fn=f'data/{dataset_version}/test.json')
    random.shuffle(qset.questions)
    
    collator = CollatorForChatGPT(
        processor=None,
        tokenizer=None,
        prompt_id=prompt_id,
        model_id='gpt'
    )
    questions = collator._get_basic_batch(qset.questions, is_train=False)

    if not os.path.exists(output_fn):
        ret = {
            'overall': {},
            'per_example': {}
        }
        print(f"Saving answers to {output_fn}")
    else:
        with open(output_fn, 'r') as f:
            ret = json.load(f)
        print(f"Loading existing answers from {output_fn}")
    
    load_dotenv()
    if model in ['gpt-4o', 'o1']:
        api_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
        base_url = os.getenv("AZURE_ENDPOINT")
        api_version = "2024-12-01-preview"
        client = openai.AzureOpenAI(api_key=api_key, azure_endpoint=base_url, api_version=api_version)
    else:
        api_key = os.getenv("NEBIUS_API_KEY")
        base_url = os.getenv("NEBIUS_API_BASE_URL")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    cnt = 0
    
    for qid, image_path, question_text, ground_truth in \
        tqdm(
            zip(questions['qids'], questions['image_paths'], questions['texts'], questions['ground_truths']), 
            total=len(questions['qids'])):
        cnt += 1
        
        if qid in ret['per_example']:
            print(f"Already answered {qid}. Skipping.")
            continue
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an AI that answers image-based multiple-choice questions. Answer with a single letter (A, B, C, or D) if the question asks for one choice, or multiple letters (e.g., ABC, CD, or N for none) if it asks for multiple choices."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ],
                    },
                ],
            )
            answer = response.choices[0].message.content.strip()
        except openai.BadRequestError as e:
            print(f"Error: {e}")
            answer = str(e)
        print(f"Answer: {answer}. Gt: {ground_truth}. tokens: {response.usage.prompt_tokens} prompt / {response.usage.completion_tokens} completion / {response.usage.total_tokens} total")
        ret['per_example'][qid] = {
            'trimmed_output': answer,
            'answer': ground_truth,
        }

        if cnt % SAVING_INTERVAL == 0 or cnt == len(questions['qids']):
            with open(output_fn, 'w') as f:
                json.dump(ret, f, indent=4)
    print(f"Answers saved to {output_fn}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using AI models via API.")
    parser.add_argument("data", type=str, choices=['none_minus', 'none_plus'], help="Version of the dataset to use (none_plus or none_minus)")
    parser.add_argument("model", type=str, choices=['gpt-4o', 'o1', 'Qwen/QVQ-72B-preview'], help="Model to use")
    parser.add_argument("output_fn", type=str, default=None, help="Cache output file (.json) for the answers.")
    args = parser.parse_args()
    
    predict(args.model, args.data, args.output_fn)