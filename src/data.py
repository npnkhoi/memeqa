import json
import os
from typing import Dict
from torch.utils.data import Dataset
from src.utils import Question

IMG_DIR = "../../literature/semeval_img" # FIXME!

class MemeQADataset(Dataset):
    """
    Legacy
    """
    def __init__(self, filepath: str):
        """
        Load all the questions into memory
        """
        raw_data = json.load(open(filepath))
        self.questions = [Question(**q) for q in raw_data]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # mutating the `img` field of the question
        img_fn = self.questions[idx].img.split('/')[-1]
        self.questions[idx].img = os.path.join(IMG_DIR, img_fn)
        return self.questions[idx]


def data_factory(dir: str) -> Dict:
    ret = {}
    train = MemeQADataset(os.path.join(dir, "train.json"))
    dev = MemeQADataset(os.path.join(dir, "dev.json"))
    test = MemeQADataset(os.path.join(dir, "test.json"))
    ret['train'] = train
    ret['dev'] = dev
    ret['test'] = test
    data_dict = {q.id: q for q in train + dev + test}
    ret['data_dict'] = data_dict
    return ret