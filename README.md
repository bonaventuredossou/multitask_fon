Toward Building a Multi-Task Learning Model for Fon Language, accepted at WiNLP@EMNLP2023

- Build Multi-task Learning Model

    - For the shared layers (encoder) we can use any existing pretrained Afro-Centric LM

        - AfroLM-Large: https://huggingface.co/bonadossou/afrolm_active_learning
            * `@inproceedings{dossou-etal-2022-afrolm, title = "{A}fro{LM}: A Self-Active Learning-based Multilingual Pretrained Language Model for 23 {A}frican Languages", author = "Dossou, Bonaventure F. P. and Tonja, Atnafu Lambebo and Yousuf, Oreen and Osei, Salomey and Oppong, Abigail and Shode, Iyanuoluwa and Awoyomi, Oluwabusayo Olufunke and Emezue, Chris", booktitle = "Proceedings of The Third Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)", month = dec, year = "2022", address = "Abu Dhabi, United Arab Emirates (Hybrid)", publisher = "Association for Computational Linguistics", url = "https://aclanthology.org/2022.sustainlp-1.11", pages = "52--64"}`
        
        - XLMR-Large: https://huggingface.co/xlm-roberta-large
            * `@inproceedings{conneau-etal-2020-unsupervised, title = "Unsupervised Cross-lingual Representation Learning at Scale", author = "Conneau, Alexis  and Khandelwal, Kartikay and Goyal, Naman  and Chaudhary, Vishrav and Wenzek, Guillaume  and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin", booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics", month = jul, year = "2020", address = "Online", publisher = "Association for Computational Linguistics", url = "https://aclanthology.org/2020.acl-main.747", doi = "10.18653/v1/2020.acl-main.747", pages = "8440--8451"}`

- Evaluation:

    - The goal primarily is to detect whether multitask learning improves performance on downstream tasks for Fon. We try two settings: (a) training only on Fon and evaluating on Fon, (b) training on all languages and evaluating on Fon. We evaluate the multi-task learning model on NER and POS tasks, and compare it with baselines (models finetuned and evaluated on single tasks)

# How to get started

- Run the training: `sbatch run.sh`

This command will:

- Set up the environement
- Install required libraries: `pip install -r requirements.txt -q`
- Move to the code folder: `cd code`
- Run the training: `python run_train.py`
