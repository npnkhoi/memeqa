from typing import Dict, List, Tuple
from src.const import GENERAL_TYPE
from src.af.af_base import AFBase
import json, os
from src.af.utils import Color, MyArguments, Question, DataclassJSONEncoder, log, batch_iter
from src.af.generator import AFGenerator
from src.af.discriminator import AFDiscriminator
from src.af.data import SEMEVAL_IMAGES, TARGETS_SENTIMENTS_ACTIONS, img_fn_to_id, IMG_FOLDER, \
                            LEGAL_SAME_SENTIMENT_DISTRACTORS, get_group, ACTION_SENTIMENT, SIMILAR_TARGETS
from src.af.af_cond_act_id import NoModelGenerator
from typing_extensions import override
import random, math
from tqdm import tqdm
from transformers import HfArgumentParser
import wandb
import itertools

from src.utils import MODEL_ID

NUM_DISTRACTORS = 3
NUM_OPTIONS = 4
BATCH_SIZE = 4
ANSWER_TEXT = 'The meme {} {}'

def get_action_target_from_option(option:str) -> tuple[str,str]:
    """
    Option is a full sentence with format "The meme <action> <target>"
    """
    for a in ACTION_SENTIMENT.keys():
        if option[len('The meme '):].find(a) == 0:
            action = a
            break
    target = option[option.index(action)+len(action)+1:]
    return action,target

def get_target_action_distractors(action_target:str,
                                  old_distractors:List[str]=[]) -> List[str]:
    """return up to three distractors, excluding old_distractors"""
    action,target = get_action_target_from_option(action_target)
    
    alternate_targets = [t for t in SIMILAR_TARGETS.keys() if t not in SIMILAR_TARGETS[target.strip().lower()]]
    random.shuffle(alternate_targets)
    
    distractors = []
    sentiment = ACTION_SENTIMENT[action]
    same_sentiment_actions = LEGAL_SAME_SENTIMENT_DISTRACTORS[action]
    
    action_target_list = list(itertools.product(same_sentiment_actions + [action],alternate_targets))
    random.shuffle(action_target_list)
    for a,t in action_target_list:
        answer = ANSWER_TEXT.format(a,t)
        if len(distractors) >= 1:
            break
        
        if answer not in old_distractors:
            distractors.append(answer)
        
    
    opposite_sentiment_actions = [d for d,s in ACTION_SENTIMENT.items() 
                                      if s != sentiment]
    for a in opposite_sentiment_actions:
        if len(distractors) >= 2:
            break 
        
        _distractor = ANSWER_TEXT.format(a,target)
        if _distractor not in old_distractors:
            distractors.append(_distractor)
    
    action_target_list = list(itertools.product(opposite_sentiment_actions,alternate_targets))
    random.shuffle(action_target_list)
    for a,t in itertools.product(opposite_sentiment_actions,alternate_targets):
        answer = ANSWER_TEXT.format(a,t)
        if len(distractors) >= 3:
            break
        
        if answer not in old_distractors:
            distractors.append(answer)
    
    return distractors[:3]

class TargetActionId(AFBase):
    QUESTION_TEXT = 'Select all correct statements about this meme.'
    
    
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
            correct_answers = []
            potential_distractors = []
            for target,actions in target_actions.items():
                target = target.strip()
                correct_option = ANSWER_TEXT.format(actions[0],target)
                correct_answers.append(correct_option)
                
                similar_words:List[str] = [] if len(actions) <= 1 else self._get_similar_words(actions[1:])
                similar_action_targets = [ANSWER_TEXT.format(w,target) for w in similar_words]
                potential_distractors.extend( get_target_action_distractors(correct_option,similar_action_targets) ) 
                
            options = correct_answers + potential_distractors
            options = options[:4]
            random.shuffle(options)
            answer_key = [1 if o in correct_answers else 0 for o in options ]
            
            full_question = self.QUESTION_TEXT.format(target)
            q = Question(
                id = img_id,
                img=os.path.join(IMG_FOLDER, img),
                question=full_question,
                options=options,
                general_type='multi',
                answer_key=answer_key,
                specific_type='conditional_action_id',
                method='algorithmic',
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
                print(self.discriminator._get_discriminator_prompt(self.new_list[0]))
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

class TargetActionGenerator(NoModelGenerator):
    GENERAL_TYPE = GENERAL_TYPE.MULTIPLE.value
    
    @override
    def _standardize_distractor(self, distractor, answer):
        distractor = distractor.strip('., ')
        # _,target = get_action_target_from_option(answer)
        # if len(target.split()) < 2 and target[0].isupper():
        #     _,distractor_target = get_action_target_from_option(distractor)
        #     new_target = ' '.join(w.capitalize() for w in distractor_target.split())
        #     distractor.replace(distractor_target,new_target)
        return distractor
    
    @override
    def _add_distractors(self, q: Question, new_distractors: List[str]) -> Question:
        """
        Add new distractors by updating the options and answer_key of the question.
        Do a bit of cleaning on the distractors.
        """
        new_distractors = [d for d in new_distractors if d.strip() != '']
        if self.question_type == GENERAL_TYPE.SINGLE.value:
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
            q.method = 'algorithmic'
        else:
            assert all([key == 1 for key in q.answer_key])

            q.options += new_distractors
            q.options = q.options[:4]
            random.shuffle(q.options) # REMEMBER TO ALWAYS SHUFFLE THE OPTIONS
            q.answer_key = [
                0 if opt in new_distractors else 1
                for opt in q.options
            ]
            q.method = 'algorithmic'
        
        return q
    
    @override
    def generate(self, questions: List[Question]) -> Tuple[List[Dict]]:
        """
        Generate new questions. Returns new_list, inactive_list
        """
        new_list = []
        inactive_list = []
        for batch in tqdm(batch_iter(questions, BATCH_SIZE), total=math.ceil(len(questions) / BATCH_SIZE)):
            questions = batch
            
            for q in questions:
                # move the old distractors to the previous distractors list
                q = self._clear_distractors(q)
                new_distractors = []
                for i in range(len(q.answer_key)):
                    if q.answer_key[i] == 1: 
                        new_distractors.extend(self._get_distractors(q.options[i],old_distractors=q.previous_distractors))
                random.shuffle(new_distractors)
                q = self._add_distractors(q, new_distractors[:3])

                if len(q.options) < NUM_OPTIONS:
                    inactive_list.append(q)
                else:
                    new_list.append(q)
        
        # inactive lists, meaning there aren't enough distractors to add, and the discriminator has guessed correctly.
        return new_list, inactive_list
    
    def __init__(self):
        super().__init__()
        self._get_distractors = get_target_action_distractors
    
if __name__ == "__main__":
    parser = HfArgumentParser((MyArguments))
    args: MyArguments = parser.parse_args_into_dataclasses()[0]
    wandb.init(
        project='memeqa',
        group='af',
        entity='khoi-ml',
        name=args.run_name
    )
    args.run_id = wandb.run.id
    generator = TargetActionGenerator()
    af = TargetActionId(
        generator,
        AFDiscriminator(MODEL_ID.QWEN.value),
        SEMEVAL_IMAGES,
        args
    )
    af.run()