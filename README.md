Toward Building a Multi-Task Learning Model for Fon Language

# Structure
- Get Task and Data
    - NER and POS already added
    - We can add other tasks (there is QA dataset for Fon in AfriQA but I am not familiar with QA downstream task)

- Build Multi-task Learning Model
    - I found [this](https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855?gi=362f8e810585 ) to be a good tutorial and walk-through
    - For the shared layers (encoder) we can use any existing pretrained Afro-Centric LM
        - AfroLM (by me): https://huggingface.co/bonadossou/afrolm_active_learning
        - Afriberta: https://huggingface.co/castorini/afriberta_large (to be integrated if we decide to include it too)

    - For now the code is built using AfroLM:
        - `@inproceedings{dossou-etal-2022-afrolm, title = "{A}fro{LM}: A Self-Active Learning-based Multilingual Pretrained Language Model for 23 {A}frican Languages", author = "Dossou, Bonaventure F. P. and Tonja, Atnafu Lambebo and Yousuf, Oreen and Osei, Salomey and Oppong, Abigail and Shode, Iyanuoluwa and Awoyomi, Oluwabusayo Olufunke and Emezue, Chris", booktitle = "Proceedings of The Third Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)", month = dec, year = "2022", address = "Abu Dhabi, United Arab Emirates (Hybrid)", publisher = "Association for Computational Linguistics", url = "https://aclanthology.org/2022.sustainlp-1.11", pages = "52--64"}`

- Evaluation:
    - The goal primarily is to detect whether multitask learning improves performance on downstream tasks for Fon. We will evaluate the multi-task learning model on the task defined above, and a baseline would be to see how the models would perform when created for individual tasks (I have results for NER (for previous works) for Afriberta and AfroLM)

# How to get started

- Install required libraries: `pip install -r requirements.txt -q`
- Move to the code folder: `cd code`
- Run the training: `python run_train.py`

# To be done

- Testing code (very similar to validation) - easily done if training successful
- ...