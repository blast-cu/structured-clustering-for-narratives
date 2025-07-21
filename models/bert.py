import argparse
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
from pyhocon import ConfigFactory, HOCONConverter
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from utils.early_stopper import EarlyStopper

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the schemas module and create an alias for pickle compatibility
import annotation.schemas as schemas
sys.modules['schemas'] = schemas


class Model(torch.nn.Module):
    def __init__(self, config, device):
        super(Model, self).__init__()
        self.device = device
        model_id = config['pretrained_model']
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        assert self.tokenizer.is_fast is True
        llm_config = AutoConfig.from_pretrained(model_id)

        self.model = AutoModel.from_pretrained(model_id, config=llm_config).to(self.device)
        
        # # Freeze BERT parameters
        # for param in self.model.parameters():
        #     param.requires_grad = False
            
        input_dims = llm_config.hidden_size

        if config['use_cluster_feats']:
            input_dims += config['num_clusters']
        elif config['use_all_feats']:
            input_dims += config['num_clusters'] + config['num_char_feats']

        self.dropout = torch.nn.Dropout(0.3).to(self.device)
        self.classifier = torch.nn.Linear(input_dims, config['num_classes']).to(self.device)

    def forward(self, inputs, feats):
        text_embs = self.get_embs(inputs)
        if feats is not None:
            x = torch.cat((text_embs, feats), dim=1)
        else:
            x = text_embs
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def get_embs(self, inputs):
        encoded_batch = self.tokenizer(
            inputs, 
            return_tensors='pt', 
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        output = self.model(**encoded_batch)
        cls_embs = output.last_hidden_state[:, 0, :]  # Extract CLS tokens for all inputs
        return cls_embs


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, feats, labels):
        self.inputs = inputs
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.feats is not None:
            return self.inputs[index], self.feats[index], self.labels[index]
        else:
            return self.inputs[index], self.labels[index]


class Trainer:
    def __init__(self, config):
        super().__init__()
        self.config = config

        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(self.config, self.device)
        torch.set_default_dtype(torch.float32)

    def create_dataset(self, clustering_data, processed_chains, corpus):
        print("Creating dataset...", flush=True)

        immig_roles_stance = {
            "Immigrants:Hero": 0,
            "Immigrants:Threat": 0,
            "Immigrants:Victim": 0,
            "Immigration Advocates:Hero": 0,
            "Immigration Advocates:Threat": 0,
            "Immigration Advocates:Victim": 0,
            "Government:Hero": 0,
            "Government:Threat": 0,
            "Government:Victim": 0,
            "Judiciary:Hero": 0,
            "Judiciary:Threat": 0,
            "Judiciary:Victim": 0,
            "Law Enforcement:Hero": 0,
            "Law Enforcement:Threat": 0,
            "Law Enforcement:Victim": 0,
            "Politicians:Hero": 0,
            "Politicians:Threat": 0,
            "Politicians:Victim": 0,
            "Stance:Pro": 0,
            "Stance:Anti": 0
        }

        doc_to_clusters = {}

        for chain_idx, chain in processed_chains['processed_chains'].items():
            doc_id = chain['doc_id']
            if doc_id not in doc_to_clusters:
                doc_to_clusters[doc_id] = {
                    'chains': [],
                    'clusters': set(),
                    'cluster_freq': [0] * config['num_clusters'],
                    'role_stance_freq': immig_roles_stance.copy(),
                    'chain_to_cluster': {},
                    'text': corpus[doc_id]['text'],
                    'frame_label': corpus[doc_id]['primary_frame']
                }
            doc_to_clusters[doc_id]['chains'].append([chain_idx])
            doc_to_clusters[doc_id]['clusters'].add(clustering_data['labels'][chain_idx])
            doc_to_clusters[doc_id]['cluster_freq'][clustering_data['labels'][chain_idx]] += 1
            doc_to_clusters[doc_id]['chain_to_cluster'][chain_idx] = clustering_data['labels'][chain_idx]

            chain_group_roles = processed_chains['chain_group_roles'][chain_idx]
            for char, role in chain_group_roles.items():
                key = "{}:{}".format(char, role)
                if key in doc_to_clusters[doc_id]['role_stance_freq']:
                    doc_to_clusters[doc_id]['role_stance_freq'][key] += 1

            # generate a list using values of the role_stance_freq dictionary
            doc_to_clusters[doc_id]['role_stance_freq_list'] = list(doc_to_clusters[doc_id]['role_stance_freq'].values())

        data = []
        for doc_id, doc in doc_to_clusters.items():
            data.append({
                'doc_id': doc_id,
                'text': doc['text'],
                'cluster_feats': doc['cluster_freq'],
                'role_stance_feats': doc['role_stance_freq_list'],
                'frame_label': doc['frame_label']
            })
        
        df = pd.DataFrame(data)
        
        # Normalize cluster_feats and role_stance_feats to [0,1] using MinMaxScaler
        cluster_scaler = MinMaxScaler()
        role_stance_scaler = MinMaxScaler()
        
        cluster_feats_normalized = cluster_scaler.fit_transform(df['cluster_feats'].tolist())
        role_stance_feats_normalized = role_stance_scaler.fit_transform(df['role_stance_feats'].tolist())
        
        df['cluster_feats'] = cluster_feats_normalized.tolist()
        df['role_stance_feats'] = role_stance_feats_normalized.tolist()
        
        label_encoder = LabelEncoder()
        df['frame_label_encoded'] = label_encoder.fit_transform(df['frame_label'])
        
        # Split into train, dev, test (70%, 15%, 15%)
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=self.config["seed"], 
            stratify=df['frame_label_encoded']
        )
        dev_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=self.config["seed"],
            stratify=temp_df['frame_label_encoded']
        )
        
        # Reset indices to maintain order for doc_id recovery
        train_df = train_df.reset_index(drop=True)
        dev_df = dev_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        if self.config['use_cluster_feats']:
            train_feats = torch.tensor(train_df['cluster_feats'].tolist(), dtype=torch.float32)
            dev_feats = torch.tensor(dev_df['cluster_feats'].tolist(), dtype=torch.float32)
            test_feats = torch.tensor(test_df['cluster_feats'].tolist(), dtype=torch.float32)
        elif self.config['use_all_feats']:
            train_cluster_feats = torch.tensor(train_df['cluster_feats'].tolist(), dtype=torch.float32)
            dev__cluster_feats = torch.tensor(dev_df['cluster_feats'].tolist(), dtype=torch.float32)
            test__cluster_feats = torch.tensor(test_df['cluster_feats'].tolist(), dtype=torch.float32)

            train_role_stance_feats = torch.tensor(train_df['role_stance_feats'].tolist(), dtype=torch.float32)
            dev_role_stance_feats = torch.tensor(dev_df['role_stance_feats'].tolist(), dtype=torch.float32)
            test_role_stance_feats = torch.tensor(test_df['role_stance_feats'].tolist(), dtype=torch.float32)

            train_feats = torch.cat((train_cluster_feats, train_role_stance_feats), dim=1)
            dev_feats = torch.cat((dev__cluster_feats, dev_role_stance_feats), dim=1)
            test_feats = torch.cat((test__cluster_feats, test_role_stance_feats), dim=1)
        else:
            train_feats = None
            dev_feats = None
            test_feats = None

        train_dataloader = torch.utils.data.DataLoader(
            Dataset(train_df['text'].tolist(), train_feats, train_df['frame_label_encoded'].tolist()),
            batch_size=self.config.get('batch_size', config["batch_size"]),
            shuffle=True
        )
        dev_dataloader = torch.utils.data.DataLoader(
            Dataset(dev_df['text'].tolist(), dev_feats, dev_df['frame_label_encoded'].tolist()),
            batch_size=self.config.get('batch_size', config["batch_size"]),
            shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            Dataset(test_df['text'].tolist(), test_feats, test_df['frame_label_encoded'].tolist()),
            batch_size=self.config.get('batch_size', config["batch_size"]),
            shuffle=False
        )
        
        return train_df, dev_df, test_df, label_encoder, train_dataloader, dev_dataloader, test_dataloader

    def train(self, train_dataloader, val_dataloader, test_dataloader):
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config["weight_decay"])
        
        # Calculate total training steps for linear scheduler
        total_steps = len(train_dataloader) * self.config['epochs']
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        best_model = None
        best_val_f1, best_val_acc = -np.inf, -np.inf
        early_stopper = EarlyStopper(patience=self.config["patience"], min_delta=self.config["min_delta"])
        for epoch in range(self.config['epochs']):
            self.model.train()
            print("Epoch: ", epoch, flush=True)
            epoch_loss = 0
            if self.config['use_cluster_feats'] or self.config['use_all_feats']:
                print("Using cluster features for training...", flush=True)
                for inputs, feats, labels in tqdm(train_dataloader):
                    feats = feats.to(self.device)
                    labels = labels.to(self.device)
                    x = self.model(inputs, feats)
                    loss = loss_fn(x, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            else:
                print("Not using cluster features for training...", flush=True)
                for inputs, labels in tqdm(train_dataloader):
                    labels = labels.to(self.device)
                    x = self.model(inputs, None)
                    loss = loss_fn(x, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                epoch_loss += loss.item()

            print("Epoch Loss: ", epoch_loss / len(train_dataloader), flush=True)

            print("Evaluation on validation set...", flush=True)
            f1, acc = self.evaluate(self.model, val_dataloader)
            if early_stopper.early_stop_score(f1):
                print("Early Stopping...", flush=True)
                print("Best Validation Accuracy: ", best_val_acc, flush=True)
                print("Best Validation F1: ", best_val_f1, flush=True)
                break
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_val_acc = acc
                best_model = self.model
        
        print("Evaluation on training set...", flush=True)
        train_f1, train_acc = self.evaluate(best_model, train_dataloader)

        print("Evaluation on test set...", flush=True)
        test_f1, test_acc = self.evaluate(best_model, test_dataloader)

        print(f"{np.round(train_acc * 100, 2)}\t{np.round(train_f1 * 100, 2)}\t{np.round(best_val_acc * 100, 2)}\t{np.round(best_val_f1 * 100, 2)}\t{np.round(test_acc * 100, 2)}\t{np.round(test_f1 * 100, 2)}", flush=True)

    def evaluate(self, model, dataloader):
        print("Evaluating...", flush=True)
        with torch.no_grad():
            model.eval()
            predicted_labels = []
            true_labels = []
            if self.config['use_cluster_feats'] or self.config['use_all_feats']:
                for inputs, feats, labels in dataloader:
                    feats = feats.to(self.device)
                    labels = labels.to(self.device)
                    x = model(inputs, feats)
                    predicted = torch.argmax(x, dim=1)
                    predicted_labels.extend(predicted.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
            else:
                for inputs, labels in dataloader:
                    labels = labels.to(self.device)
                    x = model(inputs, None)
                    predicted = torch.argmax(x, dim=1)
                    predicted_labels.extend(predicted.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

            f1 = f1_score(true_labels, predicted_labels, average='weighted')
            acc = accuracy_score(true_labels, predicted_labels)

            print("Accuracy: " + str(np.round(acc * 100, 2)), flush=True)
            print("F1 Score: " + str(np.round(f1 * 100, 2)), flush=True)

        return np.round(f1, 3), np.round(acc, 3)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCKmeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print(HOCONConverter.convert(config, 'hocon'))

    print("Loading data from disk...", flush=True)

    with open(config["cluster_eval_path"], "rb") as f:
        clustering_data = pickle.load(f)

    with open(config["processed_chains_path"], "rb") as f:
        processed_chains = pickle.load(f)

    with open(config["char_event_chains_path"], "rb") as f:
        corpus = pickle.load(f)

    trainer = Trainer(config)
    train_df, dev_df, test_df, label_encoder, train_dataloader, dev_dataloader, test_dataloader = (
        trainer.create_dataset(clustering_data, processed_chains, corpus))
    print("Training model...", flush=True)
    trainer.train(train_dataloader, dev_dataloader, test_dataloader)