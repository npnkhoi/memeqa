"""
Do both inference and evaluation on the test set
"""

import json
import os
import random
import re
import numpy as np
from typing import Dict
from src.collator import Collator
from src.const import DEFAULT_DATA_DIR
from src.data import data_factory
from src.prompt import PromptId
from src.utils import log
from src.utils import model_factory, processor_n_tokenizer_factory
from torch.utils.data import DataLoader, Subset
import argparse
import wandb
from tqdm import tqdm
import torch


def predict(model, processor, dataloader, max_new_tokens: int, evaluate=True) -> Dict[str, float]:
    # show the first batch
    batch = next(iter(dataloader))

    model.eval()
    with torch.no_grad():
        # Get all the texts first
        all_generated_texts = []
        all_answers = []
        all_example_ids = []
        for batch in tqdm(dataloader, desc="Generating texts"):
            # note: answers is just a list[str]
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            answers = batch.pop('answers')
            example_ids = batch.pop('example_ids')
            if 'pixel_values' in batch: # VLM
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens
                )
            else: # LLM
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens
                )
            generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            all_generated_texts.extend(generated_texts)
            all_answers.extend(answers)
            all_example_ids.extend(example_ids)

        # Parse and save
        ret = {
            'overall': {},
            'per_example': {}
        }
        for example_id, generated_text, answer in tqdm(zip(all_example_ids, all_generated_texts, all_answers), total=len(all_example_ids)):
            trimmed_generated_text = generated_text

            # Post-process
            HEADS = ['ASSISTANT:', '\n\nassistant\n', '## Answer:']
            for head in HEADS:
                if head in trimmed_generated_text:
                    trimmed_generated_text = trimmed_generated_text.split(head)[1].strip()
            
            TAILS = [
                'Y', # for blip2
                '<left>', # for qwen
            ]
            if hasattr(processor, 'tokenizer'): # for iblip
                TAILS.append(processor.tokenizer.eos_token)

            for tail in TAILS:
                if tail in trimmed_generated_text:
                    trimmed_generated_text = trimmed_generated_text.split(tail)[0].strip()
                        
            
            pattern = r"\(([ABCD])\)"
            res = re.search(pattern, trimmed_generated_text)
            if res:
                # for iblip, it usually outputs "(A) xyz" in zero-shot
                trimmed_generated_text = res.group(1)
            
            trimmed_generated_text = trimmed_generated_text.strip()
            
            ret['per_example'][example_id] = {
                'full_output': generated_text,
                'trimmed_output': trimmed_generated_text,
                'answer': answer,
            }
    return ret

def evaluate(ret: Dict, data_dict: Dict) -> Dict:
    # Get accuracy for each question and save accuracies by type
    acc_by_type = {}
    for example_id in ret['per_example']:
        acc = int( ret['per_example'][example_id]['trimmed_output'] ==  ret['per_example'][example_id]['answer'])
        ret['per_example'][example_id]['acc'] = acc
        q_type = data_dict[example_id].specific_type
        acc_by_type.setdefault(q_type, [])
        acc_by_type[q_type].append(acc)

    averages = []
    for qtype, accs in acc_by_type.items():
        avg = np.mean(accs)
        ret['overall']["type/" + qtype] = avg
        averages.append(avg)

    # Micro-average
    ret['overall']['micro_avg'] = np.mean([x['acc'] for x in ret['per_example'].values()])
    
    # Macro-average
    ret['overall']['macro_avg'] = np.mean(averages)
    
    wandb.log(ret['overall'])
    return ret


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, default=None, help="Path to the predictions in JSON. If specified, the predictions won't be generated anymore.")
    parser.add_argument('--base_model_id', type=str, default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--peft_id', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--no_evaluate', action='store_true')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'all', 'dev'])
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--prompt_id', type=str, default=PromptId.QA_NONEABOVE.value, choices=[memeber.value for memeber in PromptId])
    args = parser.parse_args()
    print(args)

    wandb.init(
        project='memeqa-inference',
        name=args.run_name,
        config=vars(args),
    )

    all_data = data_factory(args.data_dir)
    data_dict = all_data['data_dict'] # data_dict is needed regardless of the availability of predictions

    if args.pred_file is None:
        # Data
        task = args.prompt_id.split('/')[0]
        processor, tokenizer = processor_n_tokenizer_factory(args.base_model_id)
        collator = Collator(processor, tokenizer, args.prompt_id, args.base_model_id)    
        
        dataset = all_data[args.split]

        if args.pilot:
            print('Pilot mode')
            dataset = Subset(dataset, random.sample(range(len(dataset)), 5)) # for testing
        
        print('Dataset size:', len(dataset))
        
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, # no need to shuffle for testing
            collate_fn=collator.eval_collate_fn
        )
        
        # Model
        model = model_factory(args.base_model_id)
        if args.peft_id is not None:
            print('Loading adapter...')
            model.load_adapter(args.peft_id)
        
        # Inference 
        predictions = predict(model, processor, dataloader, args.max_new_tokens)
    else:
        predictions = json.load(open(args.pred_file))
    
    results = evaluate(predictions, data_dict)

    os.makedirs('out', exist_ok=True)
    output_file = f'out/{args.run_name}_{wandb.run.id}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f'Output saved to {output_file}')
    log({"output_file": output_file})
    
    # Finish
    wandb.finish()
