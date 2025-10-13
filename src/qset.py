import json
from src.dataset import Question

class QuestionSet:
    """
    A set of questions
    """

    def __init__(self, questions: list[Question]=None, fn: str=None):
        if fn is not None:
            self.questions = self.load_questions(fn)
        else:
            assert questions is not None, "Either questions or fn must be provided"
            self.questions = questions
        
        self.group_by_type()
    
    def load_questions(self, fn: str):
        """
        Load questions from a JSON file
        """

        return [Question(**q) for q in json.load(open(fn))]

        
    def group_by_type(self):
        self.type2pointers = {}
        for i, q in enumerate(self.questions):
            if q.specific_type not in self.type2pointers:
                self.type2pointers[q.specific_type] = []
            self.type2pointers[q.specific_type].append(i)
