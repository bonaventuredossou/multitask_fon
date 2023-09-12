Toward Building a Multi-Task Learning Model for Fon Language, accepted at WiNLP@EMNLP2023

- Build Multi-task Learning Model: For the shared layers (encoders) we used the following language model heads:

    - [AfroLM-Large](https://huggingface.co/bonadossou/afrolm_active_learning)
        - [AfroLM: A Self-Active Learning-based Multilingual Pretrained Language Model for 23 African Languages](https://aclanthology.org/2022.sustainlp-1.11/) (Dossou et.al., EMNLP 2022)        
    
    - [XLMR-Large](https://huggingface.co/xlm-roberta-large):
        - [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747) (Conneau et.al., ACL 2020)

- Evaluation:

    - The goal primarily is to explore whether multitask learning improves performance on downstream tasks for Fon. We try two settings: (a) training only on Fon and evaluating on Fon, (b) training on all languages and evaluating on Fon. We evaluate the multi-task learning model on NER and POS tasks, and compare it with baselines (models finetuned and evaluated on single tasks)

# How to get started

- Run the training: `sbatch run.sh`

This command will:

- Set up the environement
- Install required libraries: `pip install -r requirements.txt -q`
- Move to the code folder: `cd code`
- Run the training & evaluate: `python run_train.py`

# NER Results
Model | Task | Pretraining/Finetuning Dataset | Pretraining/Finetuning Language(s) | Evaluation Dataset | Metric | Metric's Value |
|:---: |:---: |:---: | :---: |:---: | :---: | :---: |
`AfroLM-Large` | Single Task | MasakhaNER 2.0 | All | FON NER | F1-Score | 80.48 |
`AfriBERTa-Large` | Single Task | MasakhaNER 2.0 | All | FON NER | F1-Score | 79.90 |
`XLMR-Base` | Single Task | MasakhaNER 2.0 | All | FON NER | F1-Score | 81.90 |
`XLMR-Large` | Single Task | MasakhaNER 2.0 | All | FON NER | F1-Score | 81.60 |
`AfroXLMR-Base` | Single Task | MasakhaNER 2.0 | All | FON NER | F1-Score | 82.30 |
`AfroXLMR-Large` | Single Task | MasakhaNER 2.0 | All | FON NER | F1-Score | 82.70 |
|:---: |:---: |:---: | :---: |:---: | :---: |
`MTL Sum (ours)` | Multi-Task | MasakhaNER 2.0 & MasakhaPOS | All | FON NER | F1-Score | 79.87 |
`MTL Weighted (ours)` | Multi-Task | MasakhaNER 2.0 & MasakhaPOS | All | FON NER | F1-Score | 81.92 |
`MTL Weighted (ours)` | Multi-Task | MasakhaNER 2.0 & MasakhaPOS | Fon Data | FON NER | F1-Score | 64.43 |


# POS Results
Model | Task | Pretraining/Finetuning Dataset | Pretraining/Finetuning Language(s) | Evaluation Dataset | Metric | Metric's Value |
|:---: |:---: |:---: | :---: |:---: | :---: | :---: |
`AfroLM-Large` | Single Task | MasakhaPOS | All | FON POS | Accuracy | 82.40 |
`AfriBERTa-Large` | Single Task | MasakhaPOS | All | FON POS | Accuracy | 88.40 |
`XLMR-Base` | Single Task | MasakhaPOS | All | FON POS | Accuracy | 90.10 |
`XLMR-Large` | Single Task | MasakhaPOS | All | FON POS | Accuracy | 90.20 |
`AfroXLMR-Base` | Single Task | MasakhaPOS | All | FON POS | Accuracy | 90.10 |
`AfroXLMR-Large` | Single Task | MasakhaPOS | All | FON POS | Accuracy | 90.40 |
|:---: |:---: |:---: | :---: |:---: | :---: |
`MTL Sum (ours)` | Multi-Task | MasakhaNER 2.0 & MasakhaPOS | All | FON POS | Accuracy | 82.45 |
`MTL Weighted (ours)` | Multi-Task | MasakhaNER 2.0 & MasakhaPOS | All | FON POS | Accuracy | 89.20 |
`MTL Weighted (ours)` | Multi-Task | MasakhaNER 2.0 & MasakhaPOS | Fon Data | FON POS | Accuracy | 80.85 |

# Importance of Merging Representation Type

Merging Type | Models | Task | Metric | Metric's Value |
| :---: | :---: | :---: | :---: | :---: |
Multiplicative | MTL Weighted (multi-task; ours; *) | NER | F1-Score | **81.92** |
Multiplicative | MTL Weighted (multi-task; ours; +) | NER | F1-Score | 64.43 |
| :---: | :---: | :---: | :---: | :---:|
Multiplicative | MTL Weighted (multi-task; ours; *) | POS | Accuracy | **89.20** |
Multiplicative & MTL Weighted (multi-task; ours; +) | POS | Accuracy | 80.85 | 
| :---: | :---: | :---: | :---: | :---: |
Additive | MTL Weighted (multi-task; ours; *) | NER | F1-Score | 78.91 |
Additive | MTL Weighted (multi-task; ours; +) | NER | F1-Score | 60.93 |
| :---: | :---: | :---: | :---: | :---: |
Additive | MTL Weighted (multi-task; ours; *) | POS | Accuracy | 86.99 |
Additive | MTL Weighted (multi-task; ours; +) | POS | Accuracy | 78.25 |

# Model End-Points

- [`multitask_model_fon_False_multiplicative.bin`](https://huggingface.co/bonadossou/multitask_model_fon_False_multiplicative) is the MTL Fon Model which has been pre-trained on all MasakhaNER 2.0 and MasakhaPOS datasets, and merging representations in a multiplicative way.

- [`multitask_model_fon_True_multiplicative.bin`](https://huggingface.co/bonadossou/multitask-learning-fon-true-multiplicative) is the MTL Fon Model which has been pre-trained only on Fon data from the MasakhaNER 2.0 and MasakhaPOS datasets, and merging representations in a multiplicative way.

# How to run inference when you have the model
 To run inference with the model(s), you can use the [testing block](https://github.com/bonaventuredossou/multitask_fon/blob/main/code/run_train.py#L209) defined in our MultitaskFON class.

 # TODO

 - leverage the impact of merging representations in an `additive way` (currently in exploration/running)
 - leverage the impact of `the dynamic weighted average loss`