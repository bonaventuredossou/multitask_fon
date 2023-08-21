import os
from utils import *
import torch
from torch.nn import CrossEntropyLoss
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaForMaskedLM, AutoTokenizer, AutoModel

import wandb
import argparse

class MultitaskFON:
    def __init__(self, args):
        self.args = args
        wandb.init(project="multitask_fon")
        self.num_gpus = [i for i in range(torch.cuda.device_count())]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        
        self.merging_type = args.merging_type

        if len(self.num_gpus) > 1:
            print("Let's use", len(self.num_gpus), "GPUs!")
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self.num_gpus)

        self.labels_ner_path = args.labels_ner_path
        self.labels_pos_path = args.labels_pos_path
        
        if args.fon_only:
            self.ner_data = "../data/ner/fon/"
            self.pos_data = "../data/pos/fon/"
        else:
            self.ner_data = "../data/ner/all/"
            self.pos_data = "../data/pos/all/"
        
        self.labels_ner = get_ner_labels(self.labels_ner_path)
        self.labels_pos = get_pos_labels(self.labels_pos_path)

        self.num_labels_ner = len(self.labels_ner)
        self.num_labels_pos = len(self.labels_pos)

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        # Load the model
        self.encoder_1 = XLMRobertaForMaskedLM.from_pretrained(args.hf_encoder_model_1_path)
        self.encoder_2 = XLMRobertaForMaskedLM.from_pretrained(args.hf_encoder_model_2_path)
        # self.tokenizer_2 = XLMRobertaTokenizer.from_pretrained(args.hf_model_2_tokenizer_path)
        
        # Using a sole tokenizer (AfroLM): it was pretrained on all languages of the datasets, while XLMR was
        # so intuitively we think it would provide a better representation
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(args.hf_tokenizer_path)
        self.encoders = [self.encoder_1, self.encoder_2]

        self.train_dataset_ner = load_ner_examples(self.ner_data, self.tokenizer, self.labels_ner, self.pad_token_label_id, mode="train")
        self.train_dataset_pos = load_pos_examples(self.pos_data, self.tokenizer, self.labels_pos, self.pad_token_label_id, mode="train")

        self.dev_dataset_ner = load_ner_examples(self.ner_data, self.tokenizer, self.labels_ner, self.pad_token_label_id, mode="dev")
        self.dev_dataset_pos = load_pos_examples(self.pos_data, self.tokenizer, self.labels_pos, self.pad_token_label_id, mode="dev")

        self.test_dataset_ner = load_ner_examples(self.ner_data, self.tokenizer, self.labels_ner, self.pad_token_label_id, mode="test")
        self.test_dataset_pos = load_pos_examples(self.pos_data, self.tokenizer, self.labels_pos, self.pad_token_label_id, mode="test")

        self.train_dataset = [self.train_dataset_ner, self.train_dataset_pos]
        self.dev_dataset = [self.dev_dataset_ner, self.dev_dataset_pos]
        self.test_dataset = [self.test_dataset_ner, self.test_dataset_pos]

        self.labels = [self.labels_ner, self.labels_pos]

        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.test_batch_size = args.test_batch_size

        self.train_dataloader_ner = DataLoader(self.train_dataset[0], batch_size=self.train_batch_size, shuffle=True)
        self.train_dataloader_pos = DataLoader(self.train_dataset[1], shuffle=True, batch_size=self.train_batch_size)

        self.dev_dataloader_ner = DataLoader(self.dev_dataset[0], batch_size=self.dev_batch_size, shuffle=False)
        self.dev_dataloader_pos = DataLoader(self.dev_dataset[1], shuffle=False, batch_size=self.dev_batch_size)

        self.test_dataloader_ner = DataLoader(self.test_dataset[0], batch_size=self.test_batch_size, shuffle=False)
        self.test_dataloader_pos = DataLoader(self.test_dataset[1], shuffle=False, batch_size=self.test_batch_size)

        # define the model
        seq_length_ner, seq_length_pos = 0, 0
        for batches in zip(self.train_dataloader_ner, self.train_dataloader_pos):
            ner_batch, pos_batch = batches
            seq_length_ner = ner_batch[0].shape[-1]
            seq_length_pos = pos_batch[0].shape[-1]
            break#####################################
        
        self.seq_length_ner = seq_length_ner
        self.seq_length_pos = seq_length_pos

        self.model = MultiTaskModel(self.encoders, [self.num_labels_ner, self.num_labels_pos], [seq_length_ner, seq_length_pos], self.merging_type)
        self.model = to_device(self.model, self.num_gpus, self.device)

        self.num_train_epochs = args.epochs
        self.learning_rate = args.learning_rate


    def train(self):         
        print("***** Running training *****")
        print("Num examples NER = %d", len(self.train_dataset[0]))
        print("Num examples POS = %d", len(self.train_dataset[1]))

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        best_dev_loss = 1000

        #Initialize the weights for each task
        if args.dynamic_weighting:
            ner_weight = torch.tensor(0.5, requires_grad=True).to(self.device) #initialize to 0.5
            pos_weight = 1 - ner_weight
        else:
            ner_weight = 0.5
            pos_weight = 0.5

        for epoch in range(self.num_train_epochs):
            self.model.train()
            epoch_train_loss, epoch_dev_loss = 0, 0

            for batches in zip(self.train_dataloader_ner, self.train_dataloader_pos):
                ner_batch, pos_batch = batches
                total_data = len(ner_batch) + len(pos_batch)

                ner_batch = tuple(t.to(self.device) for t in ner_batch)
                pos_batch = tuple(t.to(self.device) for t in pos_batch)

                ner_inputs = {"input_ids": ner_batch[0], "attention_mask": ner_batch[1]}
                pos_inputs = {"input_ids": pos_batch[0], "attention_mask": pos_batch[1]}

                ner_inputs["token_type_ids"] = ner_batch[2]
                pos_inputs["token_type_ids"] = pos_batch[2]

                outputs_ner, outputs_pos = self.model(ner_inputs, pos_inputs)

                loss_t1 = criterion(outputs_ner, ner_batch[3])
                loss_t2 = criterion(outputs_pos, pos_batch[3])

                # Calculate the final loss using the weighted sum of the losses for each task
                loss = ner_weight * loss_t1 + pos_weight * loss_t2
                epoch_train_loss += loss.item()
                epoch_train_loss = epoch_train_loss / total_data

                loss.backward()

                if args.dynamic_weighting:
                    ner_weight.backward()
                    ner_weight = ner_weight - self.learning_rate * ner_weight.grad
                    ner_weight = torch.tensor(ner_weight, requires_grad=True).to(self.device)
                    pos_weight = 1 - ner_weight
                    wandb.log({"ner_weight": ner_weight, "epoch": epoch + 1})
                    wandb.log({"pos_weight": pos_weight, "epoch": epoch + 1})

                optimizer.step()
                optimizer.zero_grad()

            wandb.log({"train_loss": epoch_train_loss, "epoch": epoch + 1})
            print("Epoch {}'s training loss: {}".format(epoch +1, epoch_train_loss))

            self.model.eval()

            for dev_batches in zip(self.dev_dataloader_ner, self.dev_dataloader_pos):
                dev_ner_batch, dev_pos_batch = dev_batches
                dev_total_data = len(dev_ner_batch) + len(dev_pos_batch)

                dev_ner_batch = tuple(t.to(self.device) for t in dev_ner_batch)
                dev_pos_batch = tuple(t.to(self.device) for t in dev_pos_batch)

                dev_ner_inputs = {"input_ids": dev_ner_batch[0], "attention_mask": dev_ner_batch[1]}
                dev_pos_inputs = {"input_ids": dev_pos_batch[0], "attention_mask": dev_pos_batch[1]}

                dev_ner_inputs["token_type_ids"] = dev_ner_batch[2]
                dev_pos_inputs["token_type_ids"] = dev_pos_batch[2]

                with torch.no_grad():
                    dev_outputs_ner, dev_outputs_pos = self.model(dev_ner_inputs, dev_pos_inputs)

                dev_loss_t1 = criterion(dev_outputs_ner, dev_ner_batch[3])
                dev_loss_t2 = criterion(dev_outputs_pos, dev_pos_batch[3])

                dev_ner_ratio = 0.5
                dev_pos_ratio = 0.5

                dev_loss = dev_ner_ratio*dev_loss_t1 + dev_pos_ratio*dev_loss_t2
                epoch_dev_loss += dev_loss.item()

                epoch_dev_loss = epoch_dev_loss / dev_total_data

            wandb.log({"dev_loss": epoch_dev_loss, "epoch": epoch + 1})

            if epoch_dev_loss < best_dev_loss:
                best_dev_loss = epoch_dev_loss
                path = 'multitask_model_fon.bin'
                torch.save(self.model.state_dict(), path)

            print("Epoch {}'s validation loss: {}".format(epoch + 1, epoch_dev_loss))
        print('Best validation loss: {}'.format(best_dev_loss))

    def test(self):         
        print("***** Running testing *****")
        print("Num examples NER = %d", len(self.test_dataset[0]))
        print("Num examples POS = %d", len(self.test_dataset[1]))

        model = MultiTaskModel(self.encoders, [self.num_labels_ner, self.num_labels_pos],
                               [self.seq_length_ner, self.seq_length_pos])
        model = to_device(model, self.num_gpus, self.device)
        model.load_state_dict(torch.load('multitask_model_fon.bin'))
        model.eval()

        ner_preds = None
        ner_label_ids = None

        pos_preds = None
        pos_label_ids = None

        for test_batches in zip(self.test_dataloader_ner, self.test_dataloader_pos):
            test_ner_batch, test_pos_batch = test_batches

            test_ner_batch = tuple(t.to(self.device) for t in test_ner_batch)
            test_pos_batch = tuple(t.to(self.device) for t in test_pos_batch)

            test_ner_batch = {"input_ids": test_ner_batch[0], "attention_mask": test_ner_batch[1]}
            test_pos_batch = {"input_ids": test_pos_batch[0], "attention_mask": test_pos_batch[1]}

            test_ner_batch["token_type_ids"] = test_ner_batch[2]
            test_pos_batch["token_type_ids"] = test_pos_batch[2]

            with torch.no_grad():
                test_outputs_ner, test_outputs_pos = model(test_ner_batch, test_pos_batch)

            if ner_preds is None:
                ner_preds = test_outputs_ner.detach().cpu().numpy()
                ner_label_ids = test_ner_batch[3].detach().cpu().numpy()
            else:
                ner_preds = np.append(ner_preds, test_outputs_ner.detach().cpu().numpy(), axis=0)
                ner_label_ids = np.append(ner_label_ids, test_ner_batch[3].detach().cpu().numpy(), axis=0)

            if pos_preds is None:
                pos_preds = test_outputs_pos.detach().cpu().numpy()
                pos_label_ids = test_pos_batch[3].detach().cpu().numpy()
            else:
                pos_preds = np.append(pos_preds, test_outputs_pos.detach().cpu().numpy(), axis=0)
                pos_label_ids = np.append(pos_label_ids, test_pos_batch[3].detach().cpu().numpy(), axis=0)

        ner_preds = np.argmax(ner_preds, axis=2)
        pos_preds = np.argmax(pos_preds, axis=2)

        ner_label_map = {i: label for i, label in enumerate(self.labels_ner)}
        pos_label_map = {i: label for i, label in enumerate(self.labels_pos)}
        
        ner_label_list = [[] for _ in range(ner_label_ids.shape[0])]
        pos_label_list = [[] for _ in range(pos_label_ids.shape[0])]

        ner_preds_list = [[] for _ in range(ner_label_ids.shape[0])]
        pos_preds_list = [[] for _ in range(pos_label_ids.shape[0])]

        for i in range(ner_label_ids.shape[0]):
            for j in range(ner_label_ids.shape[1]):
                if ner_label_ids[i, j] != self.pad_token_label_id:
                    ner_label_list[i].append(ner_label_map[ner_label_ids[i][j]])
                    ner_preds_list[i].append(ner_label_map[ner_preds[i][j]])

        for i in range(pos_label_ids.shape[0]):
            for j in range(pos_label_ids.shape[1]):
                if pos_label_ids[i, j] != self.pad_token_label_id:
                    pos_label_list[i].append(pos_label_map[pos_label_ids[i][j]])
                    pos_preds_list[i].append(pos_label_map[pos_preds[i][j]])

        ner_results = {
        "precision": precision_score(ner_label_list, ner_preds_list),
        "recall": recall_score(ner_label_list, ner_preds_list),
        "f1": f1_score(ner_label_list, ner_preds_list),
        'report': classification_report(ner_label_list, ner_preds_list)
        }

        pos_results = {
        "precision": precision_score(pos_label_list, pos_preds_list),
        "recall": recall_score(pos_label_list, pos_preds_list),
        "f1": f1_score(pos_label_list, pos_preds_list),
        'report': classification_report(pos_label_list, pos_preds_list)
        }

        with open('mtl_fon_ner_results.txt', "w") as ner_writer:
            for key in sorted(ner_results.keys()):
                ner_writer.write("{} = {}\n".format(key, str(ner_results[key])))

        ner_writer.close()

        with open('mtl_fon_pos_results.txt', "w") as pos_writer:
            for key in sorted(pos_results.keys()):
                pos_writer.write("{} = {}\n".format(key, str(pos_results[key])))
        
        pos_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multitask FON')
    parser.add_argument('--labels_ner_path', type=str, help='file path to label file for NER task')
    parser.add_argument('--labels_pos_path', type=str, help='file path to label file for POS task')
    parser.add_argument('--train_batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--dev_batch_size', type=int, default=4, help='dev batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--hf_encoder_model_1_path', type=str, default="bonadossou/afrolm_active_learning", help='Hugging Face encoder model path')
    parser.add_argument('--hf_encoder_model_2_path', type=str, default="xlm-roberta-large", help='Hugging Face encoder model path')
    parser.add_argument('--hf_tokenizer_path', type=str, default="bonadossou/afrolm_active_learning", help='Hugging Face tokenizer path')  
    # parser.add_argument('--hf_model_2_tokenizer_path', type=str, default="xlm-roberta-large", help='Hugging Face tokenizer path')
    parser.add_argument('--dynamic_weighting', action='store_true', help='dynamic weighting')
    parser.add_argument('--fon_only', type=bool, default=False, help='train only on fon or on all languages data')
    parser.add_argument('--merging_type', type=str, default='multiplicative', help='parameter deciding on how to merge the representations from both shared encoder')

    args = parser.parse_args()
    mt_fon = MultitaskFON(args)

    mt_fon.train()
    mt_fon.test()
