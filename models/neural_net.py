import argparse
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyhocon import ConfigFactory
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import shap

from utils.early_stopper import EarlyStopper


class NeuralNetModel(nn.Module):
    """Neural network model using only cluster, role, and stance frequency features (no BERT)."""
    
    def __init__(self, config, cluster_dim, role_dim, stance_dim, num_classes, device):
        super(NeuralNetModel, self).__init__()
        self.device = device
        
        # Feature transformation layers with higher dropout
        self.cluster_transform = nn.Sequential(
            nn.Linear(cluster_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # 0.3
            nn.Linear(64, 32)
        ).to(device)
        
        self.role_transform = nn.Sequential(
            nn.Linear(role_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),# 0.3
            nn.Linear(32, 16)
        ).to(device)
        
        self.stance_transform = nn.Sequential(
            nn.Linear(stance_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8)
        ).to(device)
        
        # Fusion layer with higher dropout
        fusion_input_dim = 32 + 16 + 8  # cluster + role + stance dims
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4), # 0.4
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3), # 0.3
            nn.Linear(32, 16)
        ).to(device)
        
        # Final classifier
        self.classifier = nn.Linear(16, num_classes).to(device)
        
    def forward(self, cluster_feats, role_feats, stance_feats):
        # Transform each feature type separately
        cluster_repr = self.cluster_transform(cluster_feats)
        role_repr = self.role_transform(role_feats)
        stance_repr = self.stance_transform(stance_feats)
        
        # Combine representations
        combined_feats = torch.cat([cluster_repr, role_repr, stance_repr], dim=1)
        fused_feats = self.fusion_layer(combined_feats)
        
        # Final classification
        output = self.classifier(fused_feats)
        return output


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cluster_feats, role_feats, stance_feats, labels):
        self.cluster_feats = cluster_feats
        self.role_feats = role_feats
        self.stance_feats = stance_feats
        self.labels = labels
        
    def __len__(self):
        return len(self.cluster_feats)
    
    def __getitem__(self, index):
        return (
            self.cluster_feats[index],
            self.role_feats[index], 
            self.stance_feats[index],
            self.labels[index]
        )


class NeuralNetTrainer:
    def __init__(self, config):
        self.config = config
        
        # Set random seeds
        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_dataset(self):
        """Load pre-created dataset from disk."""
        print("Loading dataset from disk...", flush=True)
        
        with open(self.config["frame_prediction_data_path"] + "frame_prediction_data.pickle", "rb") as f:
            dataset = pickle.load(f)
        
        train_df = dataset['train_df']
        dev_df = dataset['dev_df']
        test_df = dataset['test_df']
        label_encoder = dataset['label_encoder']
        
        # Extract features - using frequency features instead of structural
        train_cluster_feats = torch.tensor(train_df['cluster_feats_frequency'].tolist(), dtype=torch.float32)
        dev_cluster_feats = torch.tensor(dev_df['cluster_feats_frequency'].tolist(), dtype=torch.float32)
        test_cluster_feats = torch.tensor(test_df['cluster_feats_frequency'].tolist(), dtype=torch.float32)
        
        train_role_feats = torch.tensor(train_df['role_feats'].tolist(), dtype=torch.float32)
        dev_role_feats = torch.tensor(dev_df['role_feats'].tolist(), dtype=torch.float32)
        test_role_feats = torch.tensor(test_df['role_feats'].tolist(), dtype=torch.float32)
        
        train_stance_feats = torch.tensor(train_df['stance_feats'].tolist(), dtype=torch.float32)
        dev_stance_feats = torch.tensor(dev_df['stance_feats'].tolist(), dtype=torch.float32)
        test_stance_feats = torch.tensor(test_df['stance_feats'].tolist(), dtype=torch.float32)
        
        train_labels = torch.tensor(train_df['frame_label_encoded'].tolist(), dtype=torch.long)
        dev_labels = torch.tensor(dev_df['frame_label_encoded'].tolist(), dtype=torch.long)
        test_labels = torch.tensor(test_df['frame_label_encoded'].tolist(), dtype=torch.long)
        
        print(f"Feature dimensions - Cluster: {train_cluster_feats.shape[1]}, Role: {train_role_feats.shape[1]}, Stance: {train_stance_feats.shape[1]}")
        print(f"Number of classes: {len(label_encoder.classes_)}")
        
        # Initialize model with correct dimensions
        self.model = NeuralNetModel(
            self.config,
            cluster_dim=train_cluster_feats.shape[1],
            role_dim=train_role_feats.shape[1], 
            stance_dim=train_stance_feats.shape[1],
            num_classes=len(label_encoder.classes_),
            device=self.device
        )
        
        # Create dataloaders
        train_dataset = Dataset(train_cluster_feats, train_role_feats, train_stance_feats, train_labels)
        dev_dataset = Dataset(dev_cluster_feats, dev_role_feats, dev_stance_feats, dev_labels)
        test_dataset = Dataset(test_cluster_feats, test_role_feats, test_stance_feats, test_labels)
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.get('batch_size', 32), shuffle=True
        )
        dev_dataloader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=self.config.get('batch_size', 32), shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config.get('batch_size', 32), shuffle=False
        )
        
        return train_df, dev_df, test_df, label_encoder, train_dataloader, dev_dataloader, test_dataloader
    
    def train(self, train_dataloader, dev_dataloader, test_dataloader):
        """Train the neural network model with early stopping."""
        print("Training neural network model...", flush=True)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('lr', 0.001), 
                              weight_decay=self.config.get("weight_decay", 1e-4))  # Increased weight decay
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        best_model = None
        best_val_f1, best_val_acc = -np.inf, -np.inf
        early_stopper = EarlyStopper(patience=self.config.get("patience", 5),  # Reduced patience
                                   min_delta=self.config.get("min_delta", 0.01))  # Increased min_delta
        
        for epoch in range(self.config.get('epochs', 100)):
            # Training phase
            self.model.train()
            epoch_loss = 0
            print(f"Epoch: {epoch}", flush=True)
            
            for cluster_feats, role_feats, stance_feats, labels in tqdm(train_dataloader):
                cluster_feats = cluster_feats.to(self.device)
                role_feats = role_feats.to(self.device)
                stance_feats = stance_feats.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(cluster_feats, role_feats, stance_feats)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch Loss: {avg_loss:.4f}", flush=True)
            
            # Validation phase
            print("Evaluation on validation set...", flush=True)
            val_f1, val_acc = self.evaluate(self.model, dev_dataloader)
            
            scheduler.step(val_f1)
            
            # Early stopping check
            if early_stopper.early_stop_score(val_f1):
                print("Early Stopping...", flush=True)
                print(f"Best Validation Accuracy: {best_val_acc:.4f}", flush=True)
                print(f"Best Validation F1: {best_val_f1:.4f}", flush=True)
                break
                
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_model = self.model.state_dict().copy()
        
        # Load best model for final evaluation
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        # Final evaluation on all sets
        print("Final evaluation on training set...", flush=True)
        train_f1, train_acc = self.evaluate(self.model, train_dataloader)
        
        print("Final evaluation on validation set...", flush=True)
        final_val_f1, final_val_acc = self.evaluate(self.model, dev_dataloader)
        
        print("Final evaluation on test set...", flush=True)  
        test_f1, test_acc = self.evaluate(self.model, test_dataloader)
        
        # Print results in tabular format
        print("\n=== FINAL RESULTS ===")
        print("Split\t\tAccuracy\tF1 Score")
        print(f"Train\t\t{train_acc*100:.2f}\t\t{train_f1*100:.2f}")
        print(f"Validation\t{final_val_acc*100:.2f}\t\t{final_val_f1*100:.2f}")
        print(f"Test\t\t{test_acc*100:.2f}\t\t{test_f1*100:.2f}")
        print("====================\n")
        
        return train_f1, train_acc, final_val_f1, final_val_acc, test_f1, test_acc
    
    def evaluate(self, model, dataloader):
        """Evaluate model performance."""
        model.eval()
        predicted_labels = []
        true_labels = []
        
        with torch.no_grad():
            for cluster_feats, role_feats, stance_feats, labels in dataloader:
                cluster_feats = cluster_feats.to(self.device)
                role_feats = role_feats.to(self.device)
                stance_feats = stance_feats.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(cluster_feats, role_feats, stance_feats)
                predicted = torch.argmax(outputs, dim=1)
                
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        acc = accuracy_score(true_labels, predicted_labels)
        
        print(f"Accuracy: {acc*100:.2f}%", flush=True)
        print(f"F1 Score: {f1*100:.2f}%", flush=True)
        
        return f1, acc
    
    def analyze_feature_importance(self, train_dataloader, dev_dataloader, test_dataloader, label_encoder, num_samples=100):
        """Use SHAP to analyze feature importance on dev + test sets."""
        print("Analyzing feature importance with SHAP on dev + test sets...", flush=True)
        
        self.model.eval()
        
        # Collect training data for SHAP background
        train_cluster_feats = []
        train_role_feats = []
        train_stance_feats = []
        
        with torch.no_grad():
            for cluster_feats, role_feats, stance_feats, labels in train_dataloader:
                train_cluster_feats.append(cluster_feats)
                train_role_feats.append(role_feats)
                train_stance_feats.append(stance_feats)
        
        # Combine training data for background
        background_cluster = torch.cat(train_cluster_feats, dim=0).to(self.device)
        background_role = torch.cat(train_role_feats, dim=0).to(self.device)
        background_stance = torch.cat(train_stance_feats, dim=0).to(self.device)
        
        print(f"Using {background_cluster.shape[0]} training samples as SHAP background")
        
        # Collect dev + test data for SHAP analysis
        eval_cluster_feats = []
        eval_role_feats = []
        eval_stance_feats = []
        eval_labels = []
        
        # Use dev and test sets for analysis
        eval_dataloaders = [dev_dataloader, test_dataloader]
        
        with torch.no_grad():
            for dataloader in eval_dataloaders:
                for cluster_feats, role_feats, stance_feats, labels in dataloader:
                    eval_cluster_feats.append(cluster_feats)
                    eval_role_feats.append(role_feats)
                    eval_stance_feats.append(stance_feats)
                    eval_labels.extend(labels.numpy())
        
        print(f"Using {len(eval_labels)} samples from dev + test sets for SHAP analysis")
        
        # Combine eval data for SHAP analysis
        sample_cluster = torch.cat(eval_cluster_feats, dim=0).to(self.device)
        sample_role = torch.cat(eval_role_feats, dim=0).to(self.device)
        sample_stance = torch.cat(eval_stance_feats, dim=0).to(self.device)
        eval_labels = np.array(eval_labels)
        
        # Use entire training set as background
        background_data = [background_cluster, background_role, background_stance]
        
        # Define wrapper function for SHAP
        def model_wrapper(combined_features):
            batch_size = combined_features.shape[0]
            cluster_dim = background_data[0].shape[1]
            role_dim = background_data[1].shape[1]
            stance_dim = background_data[2].shape[1]
            
            # Split combined features back
            cluster_feats = combined_features[:, :cluster_dim]
            role_feats = combined_features[:, cluster_dim:cluster_dim+role_dim]
            stance_feats = combined_features[:, cluster_dim+role_dim:]
            
            # Convert to tensors
            cluster_tensor = torch.tensor(cluster_feats, dtype=torch.float32, device=self.device)
            role_tensor = torch.tensor(role_feats, dtype=torch.float32, device=self.device)
            stance_tensor = torch.tensor(stance_feats, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                outputs = self.model(cluster_tensor, role_tensor, stance_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            return probabilities.cpu().numpy()
        
        # Combine features for SHAP
        background_combined = torch.cat(background_data, dim=1).cpu().numpy()
        sample_combined = torch.cat([sample_cluster, sample_role, sample_stance], dim=1).cpu().numpy()
        
        # Create SHAP explainer using entire training set as background
        explainer = shap.KernelExplainer(model_wrapper, background_combined)
        
        # Calculate SHAP values for ALL samples
        print(f"Computing SHAP values for {sample_combined.shape[0]} samples (this may take several minutes)...", flush=True)
        print(f"Sample combined shape: {sample_combined.shape}")
        print(f"Background combined shape: {background_combined.shape}")
        shap_values = explainer.shap_values(sample_combined)  # Analyze ALL samples
        print(f"SHAP values shape: {shap_values.shape}")
        
        # Create feature names
        cluster_names = [f'cluster_freq_{i}' for i in range(background_data[0].shape[1])]
        role_names = [f'role_{i}' for i in range(background_data[1].shape[1])]
        stance_names = [f'stance_{i}' for i in range(background_data[2].shape[1])]
        feature_names = cluster_names + role_names + stance_names
        
        # Define feature boundaries
        cluster_end = len(cluster_names)
        role_end = cluster_end + len(role_names)
        total_features = len(feature_names)
        
        print("\n=== GLOBAL FEATURE IMPORTANCE ANALYSIS ===")
        print("Averaged across all instances and classes")
        
        # Global importance: average absolute SHAP values across all samples and classes
        global_importance = np.abs(shap_values).mean(axis=(0, 2))  # Average over samples and classes
        
        # Get top features globally
        global_top_indices = np.argsort(global_importance)[::-1]
        
        print(f"\nTop 15 Most Important Features Globally:")
        for i, idx in enumerate(global_top_indices[:15]):
            importance = global_importance[idx]
            feature_type = "cluster" if idx < cluster_end else ("role" if idx < role_end else "stance")
            print(f"  {i+1}. {feature_names[idx]} ({feature_type}): {importance:.4f}")
        
        # Global analysis by feature type
        cluster_importance = global_importance[:cluster_end]
        role_importance = global_importance[cluster_end:role_end] 
        stance_importance = global_importance[role_end:]
        
        print(f"\nGlobal Top 10 Cluster Features:")
        top_cluster_global = np.argsort(cluster_importance)[-10:][::-1]
        for i, idx in enumerate(top_cluster_global):
            importance = cluster_importance[idx]
            print(f"  {i+1}. {cluster_names[idx]}: {importance:.4f}")
        
        print(f"\nGlobal Top 5 Role Features:")
        top_role_global = np.argsort(role_importance)[-min(5, len(role_importance)):][::-1]
        for i, idx in enumerate(top_role_global):
            importance = role_importance[idx]
            print(f"  {i+1}. {role_names[idx]}: {importance:.4f}")
        
        print(f"\nGlobal Stance Features:")
        for i, importance in enumerate(stance_importance):
            print(f"  {i+1}. {stance_names[i]}: {importance:.4f}")
        
        print("\n" + "="*60)
        print("CLASS-SPECIFIC FEATURE IMPORTANCE ANALYSIS")
        print("Averaged across all instances for each class")
        
        for class_idx in range(len(label_encoder.classes_)):  # Show all classes
            class_name = label_encoder.classes_[class_idx]
            print(f"\nClass {class_idx} ({class_name}):")
            
            # Class-specific importance: average absolute SHAP values for this class across all samples
            class_shap_values = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
            
            # Get top 10 cluster features for this class
            cluster_shap = class_shap_values[:cluster_end]
            top_cluster_indices = np.argsort(cluster_shap)[-10:][::-1]
            print("  Top 10 Cluster Features:")
            for i, idx in enumerate(top_cluster_indices):
                importance = cluster_shap[idx]
                print(f"    {i+1}. {cluster_names[idx]}: {importance:.4f}")
            
            # Get top 5 role features for this class
            role_shap = class_shap_values[cluster_end:role_end]
            if len(role_shap) > 0:
                top_role_indices = np.argsort(role_shap)[-min(5, len(role_shap)):][::-1]
                print("  Top 5 Role Features:")
                for i, idx in enumerate(top_role_indices):
                    importance = role_shap[idx]
                    print(f"    {i+1}. {role_names[idx]}: {importance:.4f}")
            else:
                print("  Top 5 Role Features: No role features found")
            
            # Show both stance features for this class
            stance_shap = class_shap_values[role_end:]
            if len(stance_shap) > 0:
                print("  Both Stance Features:")
                for i, importance in enumerate(stance_shap):
                    print(f"    {i+1}. {stance_names[i]}: {importance:.4f}")
            else:
                print("  Both Stance Features: No stance features found")
        
        print("\n" + "="*60)
        print("SAMPLE LOCAL PREDICTIONS")
        print("Individual instance explanations for correct and incorrect predictions")
        
        # Get predictions for all samples to find correct and incorrect examples
        all_probs = model_wrapper(sample_combined)
        all_predictions = np.argmax(all_probs, axis=1)
        
        # Find correct and incorrect prediction indices
        correct_indices = np.where(all_predictions == eval_labels)[0]
        incorrect_indices = np.where(all_predictions != eval_labels)[0]
        
        print(f"\nDataset Summary:")
        print(f"  Total samples: {len(eval_labels)}")
        print(f"  Correct predictions: {len(correct_indices)} ({len(correct_indices)/len(eval_labels)*100:.1f}%)")
        print(f"  Incorrect predictions: {len(incorrect_indices)} ({len(incorrect_indices)/len(eval_labels)*100:.1f}%)")
        
        # Show 3 correct predictions
        print(f"\n--- CORRECT PREDICTIONS (3 examples) ---")
        for i, sample_idx in enumerate(correct_indices[:3]):
            true_class = eval_labels[sample_idx]
            predicted_class = all_predictions[sample_idx]
            true_class_name = label_encoder.classes_[true_class]
            confidence = all_probs[sample_idx, predicted_class]
            
            print(f"\nCorrect Example {i+1} (Sample #{sample_idx}):")
            print(f"  True Label: {true_class_name}")
            print(f"  Predicted: {true_class_name} (confidence: {confidence:.3f}) ✓")
            
            # Show top features for this correct prediction
            sample_shap = np.abs(shap_values[sample_idx, :, predicted_class])
            top_features_sample = np.argsort(sample_shap)[-10:][::-1]
            
            print(f"  Top 10 Features Supporting Correct Prediction:")
            for j, idx in enumerate(top_features_sample):
                importance = sample_shap[idx]
                feature_type = "cluster" if idx < cluster_end else ("role" if idx < role_end else "stance")
                print(f"    {j+1}. {feature_names[idx]} ({feature_type}): {importance:.4f}")
        
        # Show 3 incorrect predictions
        if len(incorrect_indices) > 0:
            print(f"\n--- INCORRECT PREDICTIONS (3 examples) ---")
            for i, sample_idx in enumerate(incorrect_indices[:3]):
                true_class = eval_labels[sample_idx]
                predicted_class = all_predictions[sample_idx]
                true_class_name = label_encoder.classes_[true_class]
                predicted_class_name = label_encoder.classes_[predicted_class]
                confidence = all_probs[sample_idx, predicted_class]
                true_confidence = all_probs[sample_idx, true_class]
                
                print(f"\nIncorrect Example {i+1} (Sample #{sample_idx}):")
                print(f"  True Label: {true_class_name}")
                print(f"  Predicted: {predicted_class_name} (confidence: {confidence:.3f}) ✗")
                print(f"  True class confidence: {true_confidence:.3f}")
                
                # Show top features for the incorrect prediction
                sample_shap_pred = np.abs(shap_values[sample_idx, :, predicted_class])
                top_features_pred = np.argsort(sample_shap_pred)[-10:][::-1]
                
                print(f"  Top 10 Features Supporting Incorrect Prediction ({predicted_class_name}):")
                for j, idx in enumerate(top_features_pred):
                    importance = sample_shap_pred[idx]
                    feature_type = "cluster" if idx < cluster_end else ("role" if idx < role_end else "stance")
                    print(f"    {j+1}. {feature_names[idx]} ({feature_type}): {importance:.4f}")
                
                # Show top features for the true class
                sample_shap_true = np.abs(shap_values[sample_idx, :, true_class])
                top_features_true = np.argsort(sample_shap_true)[-5:][::-1]
                
                print(f"  Top 5 Features Supporting True Class ({true_class_name}):")
                for j, idx in enumerate(top_features_true):
                    importance = sample_shap_true[idx]
                    feature_type = "cluster" if idx < cluster_end else ("role" if idx < role_end else "stance")
                    print(f"    {j+1}. {feature_names[idx]} ({feature_type}): {importance:.4f}")
        else:
            print(f"\n--- NO INCORRECT PREDICTIONS FOUND ---")
            print("The model achieved 100% accuracy on the training set!")
        
        print("="*60 + "\n")
        
        return shap_values, feature_names
    
    def save_predictions(self, dev_df, test_df, dev_dataloader, test_dataloader, label_encoder):
        """Save predictions and compare with BERT model directly."""
        print("Generating neural network predictions and comparing with BERT...")
        
        self.model.eval()
        
        # Load BERT predictions
        try:
            with open(self.config["frame_prediction_data_path"] + "bert_predictions.pickle", "rb") as f:
                bert_data = pickle.load(f)
            print("Loaded BERT predictions for comparison")
        except FileNotFoundError:
            print("BERT predictions not found - saving neural net predictions only")
            bert_data = None
        
        # Get neural net predictions for dev and test
        dataloaders = [('dev', dev_dataloader, dev_df), ('test', test_dataloader, test_df)]
        nn_predictions = {}
        matching_examples = []
        
        for split_name, dataloader, df in dataloaders:
            split_predictions = []
            
            with torch.no_grad():
                for cluster_feats, role_feats, stance_feats, labels in dataloader:
                    cluster_feats = cluster_feats.to(self.device)
                    role_feats = role_feats.to(self.device)
                    stance_feats = stance_feats.to(self.device)
                    
                    outputs = self.model(cluster_feats, role_feats, stance_feats)
                    predictions = torch.argmax(outputs, dim=1)
                    split_predictions.extend(predictions.cpu().numpy())
            
            nn_predictions[split_name] = split_predictions
            
            # Compare with BERT if available
            if bert_data:
                bert_split_preds = None
                for bert_split in bert_data['predictions']:
                    if bert_split['split'] == split_name:
                        bert_split_preds = bert_split['predictions']
                        break
                
                if bert_split_preds:
                    # Direct comparison since both use shuffle=False
                    for i, (nn_pred, bert_pred) in enumerate(zip(split_predictions, bert_split_preds)):
                        if nn_pred == bert_pred:
                            true_label = df['frame_label_encoded'].iloc[i]
                            matching_examples.append({
                                'split': split_name,
                                'index': i,
                                'text': df['text'].iloc[i],
                                'true_label': int(true_label),
                                'true_label_name': label_encoder.classes_[true_label],
                                'predicted_label': int(nn_pred),
                                'predicted_label_name': label_encoder.classes_[nn_pred],
                                'prediction_correct': bool(nn_pred == true_label)
                            })
        
        # Save results
        if bert_data and matching_examples:
            print(f"Found {len(matching_examples)} matching predictions between neural net and BERT")
            
            # Save matching examples
            import json
            with open(self.config["frame_prediction_data_path"] + "matching_predictions.json", "w") as f:
                json.dump(matching_examples, f, indent=2)
            
            # Print summary
            correct_matches = sum(1 for ex in matching_examples if ex['prediction_correct'])
            total_examples = sum(len(nn_predictions[split]) for split in ['dev', 'test'])
            
            print(f"Agreement summary:")
            print(f"  Total examples: {total_examples}")
            print(f"  Matching predictions: {len(matching_examples)} ({len(matching_examples)/total_examples*100:.1f}%)")
            print(f"  Correct matches: {correct_matches}")
            print(f"  Incorrect matches: {len(matching_examples) - correct_matches}")
        
        return nn_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Network Frame Prediction')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=0.001, help='Early stopping minimum delta')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]
    
    # Override config with command line arguments
    if args.lr:
        config['lr'] = args.lr
    if args.epochs:
        config['epochs'] = args.epochs  
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.patience:
        config['patience'] = args.patience
    if args.min_delta:
        config['min_delta'] = args.min_delta
    if args.weight_decay:
        config['weight_decay'] = args.weight_decay
    
    print("Configuration:")
    print(f"  Learning rate: {config.get('lr', 0.001)}")
    print(f"  Epochs: {config.get('epochs', 100)}")
    print(f"  Batch size: {config.get('batch_size', 32)}")
    print(f"  Early stopping patience: {config.get('patience', 10)}")
    print(f"  Weight decay: {config.get('weight_decay', 1e-4)}")
    print()
    
    trainer = NeuralNetTrainer(config)
    train_df, dev_df, test_df, label_encoder, train_dataloader, dev_dataloader, test_dataloader = trainer.load_dataset()
    
    print("Training neural network model using only cluster, role, and stance frequency features...", flush=True)
    results = trainer.train(train_dataloader, dev_dataloader, test_dataloader)
    
    print("Performing SHAP feature importance analysis...", flush=True)
    shap_values, feature_names = trainer.analyze_feature_importance(train_dataloader, dev_dataloader, test_dataloader, label_encoder)
    
    print("Saving predictions for model comparison...", flush=True)
    trainer.save_predictions(dev_df, test_df, dev_dataloader, test_dataloader, label_encoder)
    
    print("Neural network training and analysis complete!", flush=True)