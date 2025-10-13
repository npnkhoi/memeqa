"""

Training script for llava, using PyTorch

Template:
```
CUDA_VISIBLE_DEVICES=0,1 python -m src.train \
    --model_name llava-hf/llava-1.5-7b-hf \
    --num_epochs 3 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --logging_steps 50 \
    --eval_freq 700 \
    --max_new_tokens 100 \
    --prompt_id MC/autoic_text \
    --output_dir weights/1011_autoic_text
```
"""

import json
import math
import os
from src.collator import Collator
from src.const import DEFAULT_DATA_DIR, Color
from src.evaluate import evaluate, predict
from src.prompt import PromptId
from src.utils import get_metric_names, model_factory, processor_n_tokenizer_factory
from src.dataset import data_factory
import torch
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import wandb
import argparse
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--eval_freq', type=float, default=0.2)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--prompt_id', type=str, default=PromptId.QA_NONEABOVE.value, 
                        choices=[memeber.value for memeber in PromptId])
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default="weights/tmp")
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--confirm', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--eval_first', action='store_true')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--train_subset', type=str, default='train')
    parser.add_argument('--dev_subset', type=str, default='dev')
    parser.add_argument('--run_name', type=str, default=None)
    return parser.parse_args()

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def log(r):
    global global_step
    wandb.log(r, step=global_step)
    print(r)

if __name__ == '__main__':
    args = parse_args()
    print(args)

    # check the integrity of args
    if args.data_dir == 'data/none_plus':
        assert args.prompt_id == PromptId.QA_NONEABOVE.value
    elif args.data_dir == 'data/none_minus':
        assert args.prompt_id == PromptId.QA.value
    else:
        raise ValueError(f"Unknown data_dir: {args.data_dir}")


    wandb.init(
        project='memeqa-train',
        config=vars(args),
        name=args.run_name,
    )

    ###################### Data ######################
    processor, tokenizer = processor_n_tokenizer_factory(args.model_name)

    # during training, one always uses padding on the right
    tokenizer.padding_side = "left" # maybe left (warning from model) 
    
    collator = Collator(processor, tokenizer, args.prompt_id, args.model_name) 
    task = collator.task

    all_data = data_factory(args.data_dir)
    

    train_dataset = all_data[args.train_subset]
    dev_dataset = all_data[args.dev_subset]
    data_dict = all_data['data_dict']
    # don't need test
    
    if args.pilot:
        print(f'{Color.RED}Running in pilot mode. Cropping the data!{Color.END}')
        PILOT_TRAIN_SIZE = 10
        PILOT_DEV_SIZE = 10
        train_dataset = Subset(train_dataset, range(PILOT_TRAIN_SIZE))
        dev_dataset = Subset(dev_dataset, range(PILOT_DEV_SIZE))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        collate_fn=collator.train_collate_fn, 
        batch_size=args.train_batch_size, 
        shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset, 
        collate_fn=collator.eval_collate_fn, 
        batch_size=args.eval_batch_size, 
        shuffle=True
    )
    
    # Print some info for confirmation
    print(f"{Color.RED} TRAIN BATCH {Color.END}")
    train_batch = next(iter(train_loader))
    print(f"{Color.GREEN}[decoded input_ids]{Color.END}:\n```\n{tokenizer.decode(train_batch['input_ids'][0], skip_special_tokens=True)}\n```")
    clean_labels = train_batch['labels'][0]
    clean_labels[clean_labels == -100] = tokenizer.pad_token_id
    print(f"{Color.GREEN}[decoded labels]{Color.END}:\n```\n{tokenizer.decode(clean_labels, skip_special_tokens=True)}\n```")

    
    print(f"{Color.RED} DEV BATCH {Color.END}")
    dev_batch = next(iter(dev_loader))
    print(f"{Color.GREEN}[decoded input_ids]{Color.END}:\n```\n{tokenizer.decode(dev_batch['input_ids'][0], skip_special_tokens=True)}\n```")
    print(f'{Color.GREEN}[answer]{Color.END}:', dev_batch['answers'][0])

    # get metrics
    train_metrics = get_metric_names(task, 'train')
    dev_metrics = get_metric_names(task, 'eval')
    print('Train metrics:', train_metrics)
    print('Dev metrics:', dev_metrics)

    if args.confirm:
        input('Press Enter to continue...')

    ###################### Model ######################
    
    base_model = model_factory(args.model_name)

    # Create or load Lora
    if args.checkpoint_path is not None:
        peft_model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
        for name, param in peft_model.named_parameters():
            if 'lora' in name or 'Lora' in name:
                param.requires_grad = True
        print('Loaded pre-trained Lora')
    else:
        # Brand new Lora
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=find_all_linear_names(base_model),
            init_lora_weights="gaussian",
        )
        peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    # Prepare for training
    save_dir = f"{args.output_dir}/{wandb.run.id}"
    os.makedirs(save_dir, exist_ok=True)
    json.dump(vars(args), open(f"{save_dir}/training_args.json", "w"), indent=4)
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.lr)
    max_dev_score = -float('inf')
    global_step = 0
    total_steps = len(train_loader) * args.num_epochs
    eval_steps = int(math.ceil(args.eval_freq * len(train_loader)))

    # Train!
    for i_epoch in range(args.num_epochs):
        losses, temp_losses = [], []
        for i_batch, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {i_epoch}"):
            global_step += 1
            torch.cuda.empty_cache()
            peft_model.train()
            optimizer.zero_grad()
            
            device = peft_model.device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = peft_model(**batch)
            
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            temp_losses.append(loss.item())
            
            if global_step % args.logging_steps == 0:
                log({"train_loss": np.mean(temp_losses)})
                losses.extend(temp_losses)
                temp_losses = []
            

            if global_step % eval_steps == 0 or global_step == total_steps or (args.eval_first and global_step == 1):
                # Eval!
                predictions = predict(peft_model, processor, dev_loader, args.max_new_tokens)
                eval_result = evaluate(predictions, data_dict)
                
                scores = eval_result['overall']
                dev_score = np.mean(scores['macro_avg'])
                log({"dev_score": dev_score})
                
                if max_dev_score < dev_score:
                    max_dev_score = dev_score
                    peft_model.save_pretrained(save_dir + "_best")
                    log({'when_model_saved': wandb.run.step})
                    json.dump(eval_result, open(f"{save_dir}/best_eval_result.json", "w"), indent=4)
                
                log({"max_score": max_dev_score})

                # save the last checkpoint anyway
                peft_model.save_pretrained(save_dir + "_last")
                json.dump(eval_result, open(f"{save_dir}/last_eval_result.json", "w"), indent=4)

        print(f"Epoch {i_epoch} is done")
    
    print('Training is done!')
    print('Max dev score:', max_dev_score)
    print('Best model is saved at:', save_dir + "_best")
    print('Last model is saved at:', save_dir + "_last")
    log({'best_model': save_dir + "_best"})
    log({'last_model': save_dir + "_last"})
    wandb.finish()
