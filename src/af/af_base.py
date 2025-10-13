from abc import ABC, abstractmethod
import json
import os
from typing import List, Dict, Tuple
from src.af.utils import Color, MyArguments, Question, DataclassJSONEncoder, log
from src.af.generator import AFGenerator
from src.af.discriminator import AFDiscriminator
import spacy

class AFBase(ABC):
    """
    Base class for all the AF classes.
    """
    
    def __init__(self, generator: AFGenerator, discriminator: AFDiscriminator, 
        img_set: List[str], args: MyArguments) -> None:
        
        self.generator = generator
        self.discriminator = discriminator
        self.img_set = img_set
        self.args = args

        if self.args.num_iter > 1 and self.generator is None:
            raise ValueError('Generator is required for more than 1 iteration')

        self.nlp = spacy.load("en_core_web_lg")
        self.new_list, self.correct_list, stats = self._get_init_questions()
        print(f'Init stats: {stats}')
        self.wrong_list = []
        self.invalid_generation_list = []
        self.invalid_discrimination_list = []
    
    @abstractmethod
    def _get_init_questions(self) -> Tuple[List[Question] | List[Question] | Dict]:
        """
        Returns a tuple of two lists and a dict:
        - List of questions that are successfully initialized with four options
        - List of questions that are failed to be initialized (only have correct answers)
        - Dict of stats
        """
        raise NotImplementedError

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
                else:
                    print(f'{Color.GREEN}``` generator prompt')
                    print(self.generator._get_prompt_for_generator(self.new_list[0]))
                    print(f'```{Color.END}')
            
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
            
            if i > 0 and len(new_wrong_list) == 0:
                print('AF has converged')
                break
            
            # Generate: turn correct_list into new questions in new_list
            if i < self.args.num_iter - 1:
                self.new_list, new_invalid_generation_list = self.generator.generate(self.correct_list)
                self.invalid_generation_list += new_invalid_generation_list
                self.correct_list = []