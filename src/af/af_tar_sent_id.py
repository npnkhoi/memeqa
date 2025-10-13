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
ANSWER_TEXT = 'The meme is {} towards {}'

POSITIVE = 'positive'
NEGATIVE = 'negative'
NEUTRAL = 'neutral'

SENTIMENTS = [POSITIVE,NEUTRAL,NEGATIVE]

def get_opposite_sentiment(sentiment:str):
    if sentiment == POSITIVE:
        return NEGATIVE
    if sentiment == NEGATIVE:
        return POSITIVE
    return NEUTRAL


def get_other_two_sentiments(sentiment:str):
    if sentiment == POSITIVE:
        return [NEGATIVE,NEUTRAL]
    if sentiment == NEGATIVE:
        return [POSITIVE,NEUTRAL]
    return [POSITIVE,NEGATIVE]

def get_sentiment_target_from_options(options:List[str]) -> tuple[str,str]:
    sentiments = []
    targets = []
    for o in options:
        _sentiment = o.split()[3]
        target_index = len('The meme is ') + len(_sentiment) + len(' towards ')
        _target = o[target_index:]
        
        sentiments.append(_sentiment)
        targets.append(_target)
    return sentiments,targets

def get_target_sentiment_distractors(correct_options:List[str],
                                  old_distractors:List[str]=[]) -> List[str]:
    """return up to three distractors, excluding old_distractors"""
    if len(correct_options) >= 4:
        return []
    
    sentiments, targets = get_sentiment_target_from_options(correct_options)
    
    all_similar_targets = []
    for t in targets:
        all_similar_targets.extend(SIMILAR_TARGETS[t.strip().lower()])
    alternate_targets = [t for t in SIMILAR_TARGETS.keys() if t not in all_similar_targets]
    
    distractors = []
    for sentiment,target in zip(sentiments,targets):
        # positive
        random.shuffle(alternate_targets)
        for at in alternate_targets:
            answer = ANSWER_TEXT.format(POSITIVE,at)

            if answer not in old_distractors:
                distractors.append(answer)
                break
        
        # negative
        random.shuffle(alternate_targets)
        for at in alternate_targets:
            answer = ANSWER_TEXT.format(NEGATIVE,at)

            if answer not in old_distractors:
                distractors.append(answer)
                break
        
        
        # neutral
        random.shuffle(alternate_targets)
        for at in alternate_targets:
            answer = ANSWER_TEXT.format(NEUTRAL,at)

            if answer not in old_distractors:
                distractors.append(answer)
                break
            
        # in the case that most options are exhausted
        random.shuffle(alternate_targets)
        for at in alternate_targets:
            if len(distractors) >= 3:
                break
            
            s = random.choice(SENTIMENTS)
            answer = ANSWER_TEXT.format(s,at)
            if answer not in old_distractors:
                distractors.append(answer)
    
    return distractors[:3]

class TargetSentimentId(AFBase):
    QUESTION_TEXT = 'Select all correct statements about this meme.'
    
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
            bad_targets = []
            for i in intents:
                for target in i["targets"]:
                    name = target["name"]
                    sentiment_id = target["sentiment"]
                    
                    if name in target_sentiments.keys() and target_sentiments[name] != sentiment_id: # contrasting intents
                        bad_targets.append(name)
                        continue
                    else:
                        target_sentiments[name] = sentiment_id
                        
            for target in bad_targets:
                target_sentiments.pop(target)
            
            # get the list of actions that people used, these will not be added to the list of distractors
            correct_options = []
            for target,sentiment_id in target_sentiments.items():
                target = target.strip()
                _correct_option = ANSWER_TEXT.format(SENTIMENTS[int(sentiment_id)],target)
                correct_options.append(_correct_option)
                
            potential_distractors = get_target_sentiment_distractors(correct_options)
                
            options = correct_options + potential_distractors
            options = options[:4]
            random.shuffle(options)
            answer_key = [1 if o in correct_options else 0 for o in options ]
            
            q = Question(
                id = img_id,
                img=os.path.join(IMG_FOLDER, img),
                question=self.QUESTION_TEXT,
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

class TargetSentimentGenerator(NoModelGenerator):
    @override
    def _standardize_distractor(self, distractor, answer):
        return distractor.strip('., ')
    
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
            q.options = q.options[:4] # grrr...
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
                
                correct_options = q.options # after _clear_distractors, this list only contains correct answers
                new_distractors = self._get_distractors(correct_options,q.previous_distractors)
                random.shuffle(new_distractors)
                
                q = self._add_distractors(q, new_distractors[:3])
                q.method = 'algorithmic'

                if len(q.options) < NUM_OPTIONS:
                    inactive_list.append(q)
                else:
                    new_list.append(q)
        
        # inactive lists, meaning there aren't enough distractors to add, and the discriminator has guessed correctly.
        return new_list, inactive_list

    def __init__(self):
        self.GENERAL_TYPE = self.question_type = GENERAL_TYPE.MULTIPLE.value
        super().__init__()
        self._get_distractors = get_target_sentiment_distractors
    
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
    generator = TargetSentimentGenerator()
    af = TargetSentimentId(
        generator,
        AFDiscriminator(MODEL_ID.QWEN.value),
        SEMEVAL_IMAGES,
        args
    )
    af.run()