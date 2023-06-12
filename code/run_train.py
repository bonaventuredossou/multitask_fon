import os
from utils import *
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaForMaskedLM
import wandb

wandb.init(project="multitask_fon")

num_gpus = [i for i in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(num_gpus) > 1:
    print("Let's use", len(num_gpus), "GPUs!")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in num_gpus)

labels_ner_path, labels_pos_path = "", ""
ner_data, pos_data = "../data/ner/fon/", "../data/pos/fon/"
labels_ner, labels_pos = get_ner_labels(labels_ner_path), get_pos_labels(labels_pos_path)
num_labels_ner, num_labels_pos = len(labels_ner), len(labels_pos)
# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
pad_token_label_id = CrossEntropyLoss().ignore_index

hf_model_path2 = "xlm-roberta-large"
hf_tokenizer_path2 = "xlm-roberta-large"

hf_model_path = "bonadossou/afrolm_active_learning"
hf_tokenizer_path = "bonadossou/afrolm_active_learning"

encoder = XLMRobertaForMaskedLM.from_pretrained(hf_model_path)
tokenizer = XLMRobertaTokenizer.from_pretrained(hf_tokenizer_path)

tokenizer2 = XLMRobertaForMaskedLM.from_pretrained(hf_tokenizer_path2)
encoder2 = XLMRobertaModel.from_pretrained(hf_model_path2)

encoders = [encoder, encoder2]

train_dataset_ner = load_ner_examples(
            ner_data, tokenizer, labels_ner, pad_token_label_id, mode="train")
        
train_dataset_pos = load_pos_examples(
            pos_data, tokenizer, labels_pos, pad_token_label_id, mode="train")

dev_dataset_ner = load_ner_examples(
            ner_data, tokenizer, labels_ner, pad_token_label_id, mode="dev")
        
dev_dataset_pos = load_pos_examples(
            pos_data, tokenizer, labels_pos, pad_token_label_id, mode="dev")

train_dataset = [train_dataset_ner, train_dataset_pos]
dev_dataset = [dev_dataset_ner, dev_dataset_pos]

labels = [labels_ner, labels_pos]

train_batch_size = 8
dev_batch_size = 8

train_dataloader_ner = DataLoader(train_dataset[0], batch_size=train_batch_size*5, shuffle=True)
train_dataloader_pos = DataLoader(train_dataset[1], shuffle=True, batch_size=train_batch_size)

dev_dataloader_ner = DataLoader(dev_dataset[0], batch_size=dev_batch_size*4, shuffle=False)
dev_dataloader_pos = DataLoader(dev_dataset[1], shuffle=False, batch_size=dev_batch_size)

# define the model
seq_length_ner, seq_length_pos = 0, 0
for batches in zip(train_dataloader_ner, train_dataloader_pos):
    ner_batch, pos_batch = batches
    seq_length_ner = ner_batch[0].shape[-1]
    seq_length_pos = pos_batch[0].shape[-1]
    break

model = MultiTaskModel(encoders, [num_labels_ner, num_labels_pos],
                       [seq_length_ner, seq_length_pos])
model = to_device(model, num_gpus, device)

num_train_epochs = 50
learning_rate = 3e-5

print("***** Running training *****")
print("Num examples NER = %d", len(train_dataset[0]))
print("Num examples POS = %d", len(train_dataset[1]))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
best_dev_loss = 1000

for epoch in range(num_train_epochs):
    model.train()
    epoch_train_loss, epoch_dev_loss  = 0, 0

    for batches in zip(train_dataloader_ner, train_dataloader_pos):

        ner_batch, pos_batch = batches

        total_data = len(ner_batch) + len(pos_batch)
        
        ner_batch = tuple(t.to(device) for t in ner_batch)
        pos_batch = tuple(t.to(device) for t in pos_batch)
        
        # ner_inputs = {"input_ids": ner_batch[0], "attention_mask": ner_batch[1], "labels": ner_batch[3]}
        # pos_inputs = {"input_ids": pos_batch[0], "attention_mask": pos_batch[1], "labels": pos_batch[3]}

        ner_inputs = {"input_ids": ner_batch[0], "attention_mask": ner_batch[1]}
        pos_inputs = {"input_ids": pos_batch[0], "attention_mask": pos_batch[1]}

        ner_inputs["token_type_ids"] = ner_batch[2]
        pos_inputs["token_type_ids"] = pos_batch[2]
        
        outputs_ner, outputs_pos = model(ner_inputs, pos_inputs)
        loss_t1 = criterion(outputs_ner, ner_batch[3])
        loss_t2 = criterion(outputs_pos, pos_batch[3])
        
        ner_ratio = 1 # len(ner_batch)/total_data
        pos_ratio = 1 # len(pos_batch)/total_data

        loss = ner_ratio*loss_t1 + pos_ratio*loss_t2

        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
    
    epoch_train_loss = epoch_train_loss / total_data
    wandb.log({"train_loss": epoch_train_loss, "epoch": epoch + 1})
    print("Epoch {}'s training loss: {}".format(epoch +1, epoch_train_loss))
    
    # evaluation
    # set the model to eval mode
    model.eval()
        
    for dev_batches in zip(dev_dataloader_ner, dev_dataloader_pos):

        dev_ner_batch, dev_pos_batch = dev_batches
        dev_total_data = len(dev_ner_batch) + len(dev_pos_batch)
        
        dev_ner_batch = tuple(t.to(device) for t in dev_ner_batch)
        dev_pos_batch = tuple(t.to(device) for t in dev_pos_batch)
        
        # dev_ner_inputs = {"input_ids": dev_ner_batch[0], "attention_mask": dev_ner_batch[1], "labels": dev_ner_batch[3]}
        # dev_pos_inputs = {"input_ids": dev_pos_batch[0], "attention_mask": dev_pos_batch[1], "labels": dev_pos_batch[3]}

        dev_ner_inputs = {"input_ids": dev_ner_batch[0], "attention_mask": dev_ner_batch[1]}
        dev_pos_inputs = {"input_ids": dev_pos_batch[0], "attention_mask": dev_pos_batch[1]}

        dev_ner_inputs["token_type_ids"] = dev_ner_batch[2]
        dev_pos_inputs["token_type_ids"] = dev_pos_batch[2]

        with torch.no_grad():
            dev_outputs_ner, dev_outputs_pos = model(dev_ner_inputs, dev_pos_inputs)

        dev_loss_t1 = criterion(dev_outputs_ner, dev_ner_batch[3])
        dev_loss_t2 = criterion(dev_outputs_pos, dev_pos_batch[3])
        
        # batches typically have != length so we weight the final loss accordingly
        dev_ner_ratio = 0.5 # len(dev_ner_batch)/dev_total_data
        dev_pos_ratio = 0.5 # len(dev_pos_batch)/dev_total_data
        
        dev_loss = dev_ner_ratio*dev_loss_t1 + dev_pos_ratio*dev_loss_t2
        epoch_dev_loss += dev_loss.item()
    
    epoch_dev_loss = epoch_dev_loss / dev_total_data
    wandb.log({"dev_loss": epoch_dev_loss, "epoch": epoch + 1})
    
    if epoch_dev_loss < best_dev_loss:
        best_dev_loss = epoch_dev_loss
        path = os.path.join('../models/multitask_model_fon.bin')
        torch.save(model.state_dict(), path)  
    print("Epoch {}'s validation loss: {}".format(epoch + 1, epoch_dev_loss))

print('Best validation loss: {}'.format(best_dev_loss))
