# MemeQA: Holistic Evaluation for Meme Understanding

[Paper](https://aclanthology.org/2025.acl-long.927/)

MemeQA is a dataset of over 9000 questions on meme understanding abilities, hosted under `data/`.

The two dataset versions are stored in `data/none-minus/` and `data/none-plus/`. (There are two versions: $None^-$ doesn't have "None of the above" options, while $$None^+$$ has "None of the above" options.)

The original meme images can be found in the repository of [SemEval 2021 Task 6](https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus/tree/main/data), and should be downloaded into `data/semeval_img/`.

The paragraphs that preceed question generation is at `paragraphs.json`. 

## Reproducing the benchmarking experiments

### Setup

Requires CUDA 11.8 and at least 48GB of GPU RAM. In our scripts, we assume two 24GB GPUs when setting `DEVICES=0,1`.

Run:
```bash
pip install poetry
poetry shell
poetry install
export DEVICES=0,1
cp .env.example .env
```

For big-models, we need to set up API services:
- GPT4o: In Azure AI Foundry, create a GPT4o deployment. Then copy the subscription key and azure endpoint into `.env`.
- QvQ: At the time of our experiments, we used the available service on Nebius AI. However, they recently ended supporting the model. We recommend finding alternative vendors by checking "Inference Providers" on [HuggingFace](https://huggingface.co/Qwen/QVQ-72B-Preview).

> **NOTE:** Students can get $100 of free Azure credits via the [GitHub Student Developer Pack](https://education.github.com/pack).


### Experiments

Zero-shot results on the two versions of MemeQA (Table 2):
```bash
# zero-shot evaluation of local models on none-minus
bash scripts/eval_zero_minus.sh llava $DEVICES # 8.5 mins
bash scripts/eval_zero_minus.sh blip $DEVICES
bash scripts/eval_zero_minus.sh iblip $DEVICES
bash scripts/eval_zero_minus.sh qwen $DEVICES


# zero-shot evaluation of local models on none-plus
bash scripts/eval_zero_plus.sh llava $DEVICES
bash scripts/eval_zero_plus.sh blip $DEVICES # 2.5 mins
bash scripts/eval_zero_plus.sh iblip $DEVICES
bash scripts/eval_zero_plus.sh qwen $DEVICES

# zero-shot evaluation of external models
# GPT-4o
python -m src.predict_gpt none_minus gpt-4o out/4o_minus.json & python -m src.evaluate_gpt out/4o_minus.json
python -m src.predict_gpt none_plus gpt-4o out/4o_plus.json & python -m src.evaluate_gpt out/4o_plus.json
# QvQ: The two following commands are broken due to the end of support for QvQ on Nebius.
# python -m src.predict_gpt none_minus qvq out/qvq_minus.json & python -m src.evaluate_gpt out/qvq_minus.json
# python -m src.predict_gpt none_plus qvq out/qvq_plus.json & python -m src.evaluate_gpt out/qvq_plus.json
```

Fine-tuned results on the two versions of MemeQA (Table 3):
```bash
# train
# NOTE: Note the model weight directory
bash scripts/train_minus.sh llava $DEVICES 
bash scripts/train_minus.sh blip $DEVICES # 2 hours 10 mins
bash scripts/train_minus.sh iblip $DEVICES
bash scripts/train_minus.sh qwen $DEVICES
bash scripts/train_plus.sh llava $DEVICES
bash scripts/train_plus.sh blip $DEVICES
bash scripts/train_plus.sh iblip $DEVICES
bash scripts/train_plus.sh qwen $DEVICES

# evaluate
# NOTE: replace ... with the path to the model weights printed by the training script
bash scripts/eval_ft_minus.sh llava $DEVICES ...
bash scripts/eval_ft_minus.sh blip $DEVICES ...
bash scripts/eval_ft_minus.sh iblip $DEVICES ...
bash scripts/eval_ft_minus.sh qwen $DEVICES ...
bash scripts/eval_ft_plus.sh llava $DEVICES ...
bash scripts/eval_ft_plus.sh blip $DEVICES ...
bash scripts/eval_ft_plus.sh iblip $DEVICES ...
bash scripts/eval_ft_plus.sh qwen $DEVICES ...
```

All results will be stored in `out/`, while model weights are in `weights/`.

## Reproducing Adversarial Filtering

Refer to `src/af/README.md`

## Cite

Please cite our paper if you find our resources useful for your work.

```
@inproceedings{nguyen-etal-2025-memeqa,
    title = "{M}eme{QA}: Holistic Evaluation for Meme Understanding",
    author = "Nguyen, Khoi P. N.  and
      Li, Terrence  and
      Zhou, Derek Lou  and
      Xiong, Gabriel  and
      Balu, Pranav  and
      Alahari, Nandhan  and
      Huang, Alan  and
      Chauhan, Tanush  and
      Bala, Harshavardhan  and
      Guzelordu, Emre  and
      Kashfi, Affan  and
      Xu, Aaron  and
      Shrestha, Suyesh  and
      Vu, Megan  and
      Wang, Jerry  and
      Ng, Vincent",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.927/",
    doi = "10.18653/v1/2025.acl-long.927",
    pages = "18926--18946",
    ISBN = "979-8-89176-251-0",
    abstract = "Automated meme understanding requires systems to demonstrate fine-grained visual recognition, commonsense reasoning, and extensive cultural knowledge. However, existing benchmarks for meme understanding only concern narrow aspects of meme semantics. To fill this gap, we present MemeQA, a dataset of over 9,000 multiple-choice questions designed to holistically evaluate meme comprehension across seven cognitive aspects. Experiments show that state-of-the-art Large Multimodal Models perform much worse than humans on MemeQA. While fine-tuning improves their performance, they still make many errors on memes wherein proper understanding requires going beyond surface-level sentiment. Moreover, injecting ``None of the above'' into the available options makes the questions more challenging for the models. Our dataset is publicly available at https://github.com/npnkhoi/memeqa."
}
```

## Contacts
Feel free to reach out to the following people in case you have any questions or comments:
* Khoi Nguyen (khoi.nguyen6@utdallas.edu)
* Terrence Li (terrence.li@utdallas.edu)
* Vincent Ng (vince@utdallas.edu)