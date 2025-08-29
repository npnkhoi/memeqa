# MemeQA: Holistic Evaluation for Meme Understanding

MemeQA is a dataset of over 9000 questions on meme understanding abilities. There are two versions -- $None^-$ (without "None of the above" option) and $$None^+$$ (with "None of the above" option). 

The dataset was publised in a paper with the same name at ACL 2025: https://aclanthology.org/2025.acl-long.927/


## Data

Relevant data files are in `data/`:
- The two dataset versions are stored in `none-minus/` and `none-plus/`. There, the JSON files show the questions, options, answer key, and other fields. 
- The paragraphs written that preceed question generation is at `paragraphs.json`.

The original meme images can be found in the repository of the [SemEval 2021 Task 6](https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus/tree/main/data).


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

