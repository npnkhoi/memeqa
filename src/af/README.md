# Adversarial Filtering

## Install

Prerequisite: installed everything in the main README.md

```bash
spacy download en_core_web_lg
```

## Run

``` bash
python -m src.af.af_bk_id
python -m src.af.af_cond_act_id
python -m src.af.af_cond_sent_id
python -m src.af.af_cond_deriv_id
python -m src.af.af_intent_comp
python -m src.af.af_intent_id
python -m src.af.af_tar_act_id
python -m src.af.af_tar_id
python -m src.af.af_tar_sent_id
python -m src.af.af_tar_viz_id
python -m src.af.af_deriv_comp # broken
```

Each of the commands above conduct adversarial filtering for a question type. The results will be in `out/af_run_*.json` files.