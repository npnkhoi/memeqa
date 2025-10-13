from typing import Dict, List, Tuple
from src.af.af_base import AFBase
import json, os
from src.af.utils import Color, MyArguments, Question, DataclassJSONEncoder, log
from src.af.discriminator import AFDiscriminator, MODEL_ID
from src.af.data import SEMEVAL_IMAGES, TARGETS_SENTIMENTS_ACTIONS, img_fn_to_id, IMG_FOLDER, MEME_INTENTS, MODEL_TYPE, ACTION_SENTIMENT
from typing_extensions import override
from transformers import HfArgumentParser
import wandb

class ConditionalSentiment:
    QUESTION_TEXT = 'What is the meme\'s sentiment towards {}?'
    
    def _get_init_questions(self) -> Tuple[List[Question] | Dict]:
        stats = {}

        all_init_questions = []
        for img in self.img_set:
            img_id = img_fn_to_id(img)
            intents = TARGETS_SENTIMENTS_ACTIONS[img]
        
            # for two differing annotations
            target_sentiments:Dict[str,str] = {}
            bad_targets = []
            for i in intents:
                for target in i["targets"]:
                    target_name = target["name"]
                    action = target["action"]
                    sentiment:int
                    if action in ACTION_SENTIMENT.keys():
                        sentiment = ACTION_SENTIMENT[action]
                    else:
                        sentiment = int(target["sentiment"])
                    if target_name in target_sentiments.keys() and target_sentiments[target_name] != sentiment:
                        bad_targets.append(target_name)
                    else:
                        target_sentiments[target_name] = sentiment
                        
            for target in bad_targets:
                target_sentiments.pop(target)
            
            for target,sentiment in target_sentiments.items():
                options = ["Positive","Neutral","Negative"]
                full_question = self.QUESTION_TEXT.format(target)
                
                q = Question(
                    id=os.path.join('cond_sent_id',img.split('.')[0],target),
                    img=os.path.join(IMG_FOLDER, img),
                    question=full_question,
                    options=options,
                    general_type='single',
                    answer_key=sentiment,
                    specific_type='condiional_sentiment_id',
                    method='symbolic',
                    previous_distractors=[],
                ) 
                all_init_questions.append(q)
                
        return all_init_questions, [], stats
            
    def __init__(self, discriminator: AFDiscriminator, 
        img_set: List[str], args: MyArguments) -> None:
        
        self.discriminator = discriminator
        self.img_set = img_set
        self.args = args

        self.new_list, self.correct_list, stats = self._get_init_questions()
        print(f'Init stats: {stats}')
        self.wrong_list = []
        self.invalid_discrimination_list = []
    
    @override
    def run(self):
        print(f'{Color.RED}``` discriminator prompt')
        print(self.discriminator._get_discriminator_prompt(self.new_list[0]))
        print(f'```{Color.END}')
        
        # Discriminate the new_list
        new_wrong_list, new_correct_list, new_invalid_discrimination_list = self.discriminator.discriminate(self.new_list)

        # Update
        self.correct_list = new_correct_list
        self.wrong_list += new_wrong_list
        self.invalid_discrimination_list += new_invalid_discrimination_list

        # Save
        total = len(self.wrong_list) + len(self.correct_list) + len(self.invalid_discrimination_list)
        log({
            'wrong_rate': len(self.wrong_list) / total,
            'correct_rate': len(self.correct_list) / total,
            'invalid_discrimination_rate': len(self.invalid_discrimination_list) / total,
        })

        os.makedirs('out', exist_ok=True)
        fn = f'out/{self.args.run_name}.json'
        json.dump({
            'wrong': self.wrong_list,
            'correct': self.correct_list,
            'invalid_discrimination': self.invalid_discrimination_list,
        }, open(fn, 'w'), cls=DataclassJSONEncoder, indent=4)
        print(f'Saved to {fn}')
   
   
class ConditionalSentimentDiscriminator(AFDiscriminator):
    @override
    def _get_discriminator_prompt(self, q: Question, model_type: MODEL_TYPE=MODEL_TYPE.VLM) -> str:
        """
        q is a dictionary with the following keys: 'img', 'question', 'options', 'general_type'.
        For LLM case, hints will be retrieved based on the img file.
        """
        assert len(q.options) == 3


        task_instruction = "You are given a meme.\n"
        img_fn = q.img.split('/')[-1]
        if model_type == MODEL_TYPE.LLM:
            raise NotImplementedError
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
                "Answer the following question by writing ONLY one letter: A or B. " +
                "DO NOT write anything else. ONLY write the letter of the correct answer."
            )
        else: # this code isnt accessed
            task_instruction += (
                "Answer the following question by selecting ALL the correct options and write their letters consecutively, such as 'ACD' or 'B'. " +
                "Write 'N' if none of the options are correct. " +
                "DO NOT write anything thing else. " +
                "ONLY write the letters of the correct answers or 'N'. " +
                "Remember that you can select multiple options."
            )

        prompt = (
            task_instruction + "\n" +
            '## Question: ' + q.question + '\n' +
            '## Options:\n' +
            '(A) ' + q.options[0] + '\n'
            '(B) ' + q.options[1] + '\n'
            '(C) ' + q.options[2] + '\n'
            '## Answer: '
        )

        if self.model_path in [MODEL_ID.LLAVA_v15.value, MODEL_ID.QWEN.value]:
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
    cs = ConditionalSentiment(
        ConditionalSentimentDiscriminator(MODEL_ID.QWEN.value),
        SEMEVAL_IMAGES,
        args
    )
    cs.run()