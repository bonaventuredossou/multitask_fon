# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task.
"""
import logging
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for token classification.
    """

    def __init__(self, guid, words, labels):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """
    A single set of features of data.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_ner_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if len(line) < 2 or line == "\n":
                print(line, words)
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index), words=words, labels=labels
                        )
                    )
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    labels.append("O")
        if words:
            examples.append(
                InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels)
            )
    return examples

def read_pos_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if len(line) < 2  or line == "\n":
                if len(words) <1:
                    print(guid_index)
                    print(line, words)
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    labels.append("X")
        if words:
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
    return examples

def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of `InputBatch`s `cls_token_at_end` define the location of the CLS
    token:

        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        try:
            assert len(label_ids) == max_seq_length
        except:
            continue

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
            )
        )
    return features


def get_ner_labels(path=None):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-DATE", "I-DATE", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

def get_pos_labels(path=None):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "X" not in labels:
            labels = ["X"] + labels
        return labels
    else:
        return ["X", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB"]

class FonDataset(Dataset):
    def __init__(self, input_ids, input_labels, input_masks) -> None:
        self.examples = np.array(
            [
                {"input_ids": ids, "labels": labels, "attention_mask": masks}
                for ids, labels, masks in zip(input_ids, input_labels, input_masks)
            ]
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.examples[index]

# A simple MultiTask Model that uses a shared encoder (here an LM)
class MultiTaskModel(torch.nn.Module):
    def __init__(self, encoder, num_labels, sequence_lengths):
        super(MultiTaskModel,self).__init__()
        self.encoder = encoder
        self.n_labels_task1, self.n_labels_task2 = num_labels
        self.seq_len_1, self.seq_len_2 = sequence_lengths
        
        self.fc1 = torch.nn.Linear(768, self.seq_len_1)
        self.fc2 = torch.nn.Linear(768, self.seq_len_2)
        
        self.classifier_task1 = torch.nn.Linear(self.seq_len_1, self.n_labels_task1)
        self.classifier_task2 = torch.nn.Linear(self.seq_len_2, self.n_labels_task2)

        
    def forward(self, x1, x2):
        # Inputs of each task
        # same models but with different configurations i.e. number of labels
        representation_x1 = self.encoder(**x1)['last_hidden_state']
        representation_x2 = self.encoder(**x2)['last_hidden_state']
        
        representation_x1 = self.fc1(representation_x1)
        representation_x2 = self.fc2(representation_x2)
        
        task1_pred = self.classifier_task1(representation_x1)
        task2_pred = self.classifier_task2(representation_x2)
        
        task1_pred = task1_pred.transpose(1, 2)
        task2_pred = task2_pred.transpose(1, 2)
        
        return task1_pred, task2_pred



def load_ner_examples(path, tokenizer, labels, pad_token_label_id, mode):

    examples = read_ner_examples_from_file(path, mode)
    features = convert_examples_to_features(
        examples,
        labels,
        256,
        tokenizer,
        cls_token_at_end=False,
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=False,
        pad_on_left=False,
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        pad_token_label_id=pad_token_label_id)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def load_pos_examples(path, tokenizer, labels, pad_token_label_id, mode):
    
    examples = read_pos_examples_from_file(path, mode)
    features = convert_examples_to_features(
        examples,
        labels,
        256,
        tokenizer,
        cls_token_at_end=False,
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=False,
        pad_on_left=False,
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        pad_token_label_id=pad_token_label_id)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def to_device(model, num_gpus, device):
    if len(num_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=num_gpus)
        model = model.module
    model = model.to(device)
    return model