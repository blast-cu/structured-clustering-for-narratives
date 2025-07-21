import argparse
import pickle
import random

import numpy as np
import torch
from pyhocon import ConfigFactory, HOCONConverter
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from utils.early_stopper import EarlyStopper

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

        # Feature transformation layers for non-BERT features
        if config['use_cluster_feats']:
            self.cluster_transform = torch.nn.Sequential(
                torch.nn.Linear(config['num_clusters'], 32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(32, 16)
            ).to(self.device)
            input_dims += 16
            self.role_transform = None
            self.stance_transform = None
            self.fusion_layer = None
        elif config['use_all_feats']:
            # Separate transformation networks for each feature type
            self.cluster_transform = torch.nn.Sequential(
                torch.nn.Linear(config['num_clusters'], 32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(32, 16)
            ).to(self.device)
            
            self.role_transform = torch.nn.Sequential(
                torch.nn.Linear(18, 24),  # 18 role features
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(24, 12)
            ).to(self.device)
            
            self.stance_transform = torch.nn.Sequential(
                torch.nn.Linear(2, 8),   # 2 stance features
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(8, 4)
            ).to(self.device)
            
            # Fusion layer for combining all three
            self.fusion_layer = torch.nn.Sequential(
                torch.nn.Linear(16 + 12 + 4, 32),  # 32 combined dims
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(32, 16)            # Final 16 dims
            ).to(self.device)
            
            input_dims += 16
        else:
            self.cluster_transform = None
            self.role_transform = None
            self.stance_transform = None
            self.fusion_layer = None

        self.dropout = torch.nn.Dropout(0.3).to(self.device)
        self.classifier = torch.nn.Linear(input_dims, config['num_classes']).to(self.device)

    def forward(self, inputs, feats):
        text_embs = self.get_embs(inputs)
        if feats is not None:
            if self.fusion_layer is not None:
                # Split concatenated features back into components for use_all_feats
                num_clusters = feats.shape[1] - 18 - 2  # Total - role - stance
                cluster_feats = feats[:, :num_clusters]
                role_feats = feats[:, num_clusters:num_clusters+18]
                stance_feats = feats[:, -2:]
                
                # Transform each component separately
                cluster_repr = self.cluster_transform(cluster_feats)
                role_repr = self.role_transform(role_feats)
                stance_repr = self.stance_transform(stance_feats)
                
                # Combine representations
                combined_feats = torch.cat([cluster_repr, role_repr, stance_repr], dim=1)
                transformed_feats = self.fusion_layer(combined_feats)
            elif self.cluster_transform is not None:
                # For use_cluster_feats only
                transformed_feats = self.cluster_transform(feats)
            else:
                transformed_feats = feats
                
            x = torch.cat((text_embs, transformed_feats), dim=1)
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

    def load_dataset(self):
        """Load pre-created dataset from disk."""
        print(f"Loading dataset from disk...", flush=True)
        
        with open(config["frame_prediction_data_path"] + "frame_prediction_data.pickle", "rb") as f:
            dataset = pickle.load(f)
        
        train_df = dataset['train_df']
        dev_df = dataset['dev_df'] 
        test_df = dataset['test_df']
        label_encoder = dataset['label_encoder']

        if self.config['use_cluster_feats']:
            train_feats = torch.tensor(train_df['cluster_feats'].tolist(), dtype=torch.float32)
            dev_feats = torch.tensor(dev_df['cluster_feats'].tolist(), dtype=torch.float32)
            test_feats = torch.tensor(test_df['cluster_feats'].tolist(), dtype=torch.float32)
        elif self.config['use_all_feats']:
            train_cluster_feats = torch.tensor(train_df['cluster_feats'].tolist(), dtype=torch.float32)
            dev__cluster_feats = torch.tensor(dev_df['cluster_feats'].tolist(), dtype=torch.float32)
            test__cluster_feats = torch.tensor(test_df['cluster_feats'].tolist(), dtype=torch.float32)

            train_role_feats = torch.tensor(train_df['role_feats'].tolist(), dtype=torch.float32)
            dev_role_feats = torch.tensor(dev_df['role_feats'].tolist(), dtype=torch.float32)
            test_role_feats = torch.tensor(test_df['role_feats'].tolist(), dtype=torch.float32)

            train_stance_feats = torch.tensor(train_df['stance_feats'].tolist(), dtype=torch.float32)
            dev_stance_feats = torch.tensor(dev_df['stance_feats'].tolist(), dtype=torch.float32)
            test_stance_feats = torch.tensor(test_df['stance_feats'].tolist(), dtype=torch.float32)

            train_feats = torch.cat((train_cluster_feats, train_role_feats, train_stance_feats), dim=1)
            dev_feats = torch.cat((dev__cluster_feats, dev_role_feats, dev_stance_feats), dim=1)
            test_feats = torch.cat((test__cluster_feats, test_role_feats, test_stance_feats), dim=1)
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
                    epoch_loss += loss.item()
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

    def inference(self, model, train_dataloader, dev_dataloader, test_dataloader):
        """Run inference on full dataset and return predictions and probabilities."""
        print("Running inference on full dataset...", flush=True)
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        
        # Combine all dataloaders
        all_dataloaders = [
            ("train", train_dataloader),
            ("dev", dev_dataloader), 
            ("test", test_dataloader)
        ]
        
        with torch.no_grad():
            for split_name, dataloader in all_dataloaders:
                print(f"Running inference on {split_name} split...", flush=True)
                split_predictions = []
                split_probabilities = []
                
                if self.config['use_cluster_feats'] or self.config['use_all_feats']:
                    for inputs, feats, labels in dataloader:
                        feats = feats.to(self.device)
                        logits = model(inputs, feats)
                        
                        # Get predictions
                        predictions = torch.argmax(logits, dim=1)
                        split_predictions.extend(predictions.cpu().numpy())
                        
                        # Get probabilities using softmax
                        probabilities = torch.softmax(logits, dim=1)
                        split_probabilities.extend(probabilities.cpu().numpy())
                else:
                    for inputs, labels in dataloader:
                        logits = model(inputs, None)
                        
                        # Get predictions
                        predictions = torch.argmax(logits, dim=1)
                        split_predictions.extend(predictions.cpu().numpy())
                        
                        # Get probabilities using softmax
                        probabilities = torch.softmax(logits, dim=1)
                        split_probabilities.extend(probabilities.cpu().numpy())
                
                all_predictions.append({
                    'split': split_name,
                    'predictions': split_predictions
                })
                all_probabilities.append({
                    'split': split_name,
                    'probabilities': split_probabilities
                })

        with open(self.config["frame_prediction_data_path"] + "bert_predictions.pickle", "wb") as f:
            pickle.dump({
                'predictions': all_predictions,
                'probabilities': all_probabilities
            }, f)
        
        return all_predictions, all_probabilities



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCKmeans Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print(HOCONConverter.convert(config, 'hocon'))

    trainer = Trainer(config)
    train_df, dev_df, test_df, label_encoder, train_dataloader, dev_dataloader, test_dataloader = (
        trainer.load_dataset())
    print("Training model...", flush=True)
    trainer.train(train_dataloader, dev_dataloader, test_dataloader)
    
    # Run inference on full dataset after training
    predictions, probabilities = trainer.inference(trainer.model, train_dataloader, dev_dataloader, test_dataloader)