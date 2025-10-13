from typing import Dict, List, Tuple
from src.const import GENERAL_TYPE
from src.af.af_base import AFBase
import json, os
from src.af.utils import Color, MyArguments, Question, DataclassJSONEncoder, log, batch_iter
from src.af.generator import AFGenerator
from src.af.discriminator import AFDiscriminator
from src.af.data import SEMEVAL_IMAGES, TARGETS_SENTIMENTS_ACTIONS, img_fn_to_id, IMG_FOLDER, \
                            LEGAL_SAME_SENTIMENT_DISTRACTORS, get_group, ACTION_SENTIMENT
from typing_extensions import override
import random, math
from tqdm import tqdm
from transformers import HfArgumentParser
import wandb

from src.utils import MODEL_ID

NUM_DISTRACTORS = 3
NUM_OPTIONS = 4
BATCH_SIZE = 4

def get_action_distractors(action:str,old_distractors:List[str]=[]):
    """return up to three distractors for action, excluding old_distractors"""
    
    distractors = []
    sentiment = ACTION_SENTIMENT[action]
    # same_sentiment_distractors = [d for d in LEGAL_SAME_SENTIMENT_DISTRACTORS[action] 
    #                               if d not in old_distractors]
    
    # for i in range(min(1,len(same_sentiment_distractors))):
    #     distractors.append(same_sentiment_distractors[i])
    
    opposite_sentiment_distractors = [d for d,s in ACTION_SENTIMENT.items() 
                                      if s != sentiment and d not in old_distractors]
    neutral_sentiment_distractors = [d for d,s in ACTION_SENTIMENT.items()
                                     if s == 1 and d not in old_distractors]
    
    distractors = opposite_sentiment_distractors + neutral_sentiment_distractors
    random.shuffle(distractors)
    
    return distractors[:3]

class ConditionalAction(AFBase):
    QUESTION_TEXT = 'Fill in the blank: The meme ____ {}'
    
    
    def _get_similar_words(self,actions:List[str]) -> List[str]:
        """get the list of actions that are similar to the input list of actions

        Returns:
            List[str]: a list of the similar words
        """
        similar_words = []
        
        for a in actions:
            group = get_group(a)
            for g in group:
                if g not in similar_words:
                    similar_words.append(g)
                
        return similar_words
    
    def _get_init_questions(self) -> Tuple[List[Question] | Dict]:
        stats = {
            'success': 0,
            'fail': 0,
        }

        success_questions = []
        failed_questions = []
        for img in self.img_set:
            img_id = img_fn_to_id(img)
            intents = TARGETS_SENTIMENTS_ACTIONS[img]
        
            # for two differing annotations
            target_sentiments:Dict[str,str] = {}
            target_actions:Dict[str,List[str]] = {}
            bad_targets = []
            for i in intents:
                for target in i["targets"]:
                    name = target["name"]
                    sentiment = target["sentiment"]
                    action = target["action"]
                    
                    if name in target_sentiments.keys() and target_sentiments[name] != sentiment: # contrasting intents
                        bad_targets.append(name)
                        continue
                    elif sentiment == 1 or action not in ACTION_SENTIMENT.keys()\
                        or ACTION_SENTIMENT[action] == 1: # labeled as neutral, accounting for strange annotations as well
                        continue
                    else:
                        target_sentiments[name] = sentiment
                        if name in target_actions.keys() and action not in target_actions[name]:
                            target_actions[name].append(action)
                        else:
                            target_actions[name] = [action]
                        
            for target in bad_targets:
                target_sentiments.pop(target)
                target_actions.pop(target)
            
            # get the list of actions that people used, these will not be added to the list of distractors
            
            for target,actions in target_actions.items():
                answer_key_action = actions[0]
                similar_words:List[str] = [] if len(actions) <= 1 else self._get_similar_words(actions[1:])
                options:List[str] = [answer_key_action] + get_action_distractors(answer_key_action,similar_words)
                random.shuffle(options)
                
                full_question = self.QUESTION_TEXT.format(target)
                
                q = Question(
                    id = img_id,
                    img=os.path.join(IMG_FOLDER, img),
                    question=full_question,
                    options=options,
                    general_type='single',
                    answer_key=options.index(answer_key_action),
                    specific_type='conditional_action_id',
                    method='coding :)',
                    previous_distractors=[],
                )
                if len(options) == 4: 
                    success_questions.append(q)
                    stats['success'] += 1
                else:
                    failed_questions.append(q)
                    stats['fail'] += 1
                
        return success_questions, failed_questions, stats
            
    @override
    def run(self):
        for i in range(self.args.num_iter):
            print(f'AF iteration {i}')

            # Generate
            if i == 0:
                print(f'{Color.RED}``` discriminator prompt')
                print(self.discriminator._get_discriminator_prompt(self.new_list[0], self.discriminator.model_path))
                print(f'```{Color.END}')
                if self.generator is None:
                    print('No generator provided')
            
            # Discriminate the new_list
            new_wrong_list, new_correct_list, new_invalid_discrimination_list = self.discriminator.discriminate(self.new_list)

            # Update
            self.correct_list = new_correct_list
            self.wrong_list += new_wrong_list
            self.invalid_discrimination_list += new_invalid_discrimination_list

            # Save
            total = len(self.wrong_list) + len(self.correct_list) + len(self.invalid_generation_list) + len(self.invalid_discrimination_list)
            log({
                'wrong_rate': len(self.wrong_list) / total,
                'correct_rate': len(self.correct_list) / total,
                'invalid_generation_rate': len(self.invalid_generation_list) / total,
                'invalid_discrimination_rate': len(self.invalid_discrimination_list) / total,
            })

            os.makedirs('out', exist_ok=True)
            fn = f'out/{self.args.run_name}_{self.args.run_id}.json'
            json.dump({
                'wrong': self.wrong_list,
                'correct': self.correct_list,
                'invalid_generation': self.invalid_generation_list,
                'invalid_discrimination': self.invalid_discrimination_list,
                'iter': i,
            }, open(fn, 'w'), cls=DataclassJSONEncoder, indent=4)
            print(f'Saved iteration {i} to {fn}')
            
            if len(new_wrong_list) == 0:
                print('AF has converged')
                break

            # Generate: turn correct_list into new questions in new_list
            if i < self.args.num_iter - 1:
                self.new_list, new_invalid_generation_list = self.generator.generate(self.correct_list)
                self.invalid_generation_list += new_invalid_generation_list
                self.correct_list = []
    
class NoModelGenerator(AFGenerator):
    GENERAL_TYPE = GENERAL_TYPE.SINGLE.value

    @override
    def __init__(self) -> None:
        self.num_distractors = NUM_DISTRACTORS
        self._get_distractors = get_action_distractors
    
    @override
    def generate(self, questions: List[Question]) -> Tuple[List[Dict]]:
        """
        Generate new questions. Returns new_list, inactive_list
        """
        new_list = []
        inactive_list = []
        for batch in tqdm(batch_iter(questions, BATCH_SIZE), total=math.ceil(len(questions) / BATCH_SIZE)):
            questions = batch
            
            # move the old distractors to the previous distractors list
            for q in questions:
                q = self._clear_distractors(q)
                new_distractors = self._get_distractors(q.options[q.answer_key],old_distractors=q.previous_distractors)
                q = self._add_distractors(q, new_distractors)
                q.method = 'algorithmic'

                if len(q.options) < NUM_OPTIONS:
                    inactive_list.append(q)
                else:
                    new_list.append(q)
        
        # inactive lists, meaning there aren't enough distractors to add, and the discriminator has guessed correctly.
        return new_list, inactive_list
    
if __name__ == "__main__":
    parser = HfArgumentParser((MyArguments))
    args: MyArguments = parser.parse_args_into_dataclasses()[0]
    wandb.init(
        project='memeqa',
        group='af',
        entity='khoi-ml',
        name=args.run_name,
    )
    args.run_id = wandb.run.id
    af = ConditionalAction(
        NoModelGenerator(),
        AFDiscriminator(MODEL_ID.QWEN.value),
        SEMEVAL_IMAGES,
        args
    )
    af.run()