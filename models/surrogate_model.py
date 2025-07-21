import argparse
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from pyhocon import ConfigFactory
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score
import optuna


class SurrogateModel:
    """LightGBM-based surrogate model that approximates BERT using handcrafted features."""
    
    def __init__(self, config):
        self.config = config
        self.models = []  # One model per class for probability prediction
        self.num_classes = config['num_classes']
        self.feature_names = None
        
        # Check GPU availability
        self.device_type = self._check_gpu_availability()
        print(f"Using device: {self.device_type}", flush=True)
        
        # Set random seed
        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
    
    def _check_gpu_availability(self):
        """Check if GPU is available for LightGBM."""
        try:
            # Try to create a simple LightGBM dataset and train with GPU
            test_data = lgb.Dataset(np.random.rand(10, 5), label=np.random.rand(10))
            test_params = {'device': 'gpu', 'objective': 'regression', 'verbosity': -1, 'num_leaves': 10}
            lgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
            return 'gpu'
        except Exception as e:
            print(f"GPU not available, falling back to CPU: {e}", flush=True)
            return 'cpu'
    
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        """Custom cross-entropy loss for LightGBM."""
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Return gradient and hessian for LightGBM
        grad = -(y_true / y_pred_clipped)
        hess = y_true / (y_pred_clipped ** 2)
        
        return grad, hess
    
    @staticmethod  
    def cross_entropy_eval(y_true, y_pred):
        """Custom cross-entropy evaluation metric for LightGBM."""
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true * np.log(y_pred_clipped))
        return 'cross_entropy', loss, False  # (eval_name, eval_result, is_higher_better)
        
    def load_data(self):
        """Load dataset and BERT predictions."""
        print("Loading dataset and BERT predictions...", flush=True)
        
        # Load dataset with features
        dataset_path = self.config["frame_prediction_data_path"] + "frame_prediction_data.pickle"
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        
        # Load BERT predictions and probabilities
        predictions_path = self.config["frame_prediction_data_path"] + "bert_predictions.pickle"
        with open(predictions_path, "rb") as f:
            bert_outputs = pickle.load(f)
        
        return dataset, bert_outputs
    
    def prepare_features_and_targets(self, dataset, bert_outputs):
        """Prepare feature matrix and probability targets from loaded data."""
        print("Preparing features and targets...", flush=True)
        
        # Combine all splits
        all_dfs = []
        all_probabilities = []
        
        splits = ['train', 'dev', 'test']
        for split in splits:
            df = dataset[f'{split}_df'].copy()
            df['split'] = split
            all_dfs.append(df)
            
            # Find corresponding probabilities
            for prob_data in bert_outputs['probabilities']:
                if prob_data['split'] == split:
                    all_probabilities.extend(prob_data['probabilities'])
                    break
        
        # Combine into single dataframe
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # Create feature matrix
        features = []
        feature_names = []
        
        # Cluster features
        if self.config.get('use_cluster_feats', True):
            cluster_feats = np.array(full_df['cluster_feats'].tolist())
            features.append(cluster_feats)
            feature_names.extend([f'cluster_{i}' for i in range(cluster_feats.shape[1])])
        
        # Role features  
        if self.config.get('use_role_feats', True):
            role_feats = np.array(full_df['role_feats'].tolist())
            features.append(role_feats)
            feature_names.extend([f'role_{i}' for i in range(role_feats.shape[1])])
            
        # Stance features
        if self.config.get('use_stance_feats', True):
            stance_feats = np.array(full_df['stance_feats'].tolist())
            features.append(stance_feats)
            feature_names.extend([f'stance_{i}' for i in range(stance_feats.shape[1])])
        
        # Combine all features
        X = np.hstack(features)
        y_probs = np.array(all_probabilities)
        
        self.feature_names = feature_names
        
        # Split back into train/dev/test
        train_mask = full_df['split'] == 'train'
        dev_mask = full_df['split'] == 'dev'  
        test_mask = full_df['split'] == 'test'
        
        return {
            'X_train': X[train_mask],
            'X_dev': X[dev_mask], 
            'X_test': X[test_mask],
            'y_train': y_probs[train_mask],
            'y_dev': y_probs[dev_mask],
            'y_test': y_probs[test_mask],
            'splits': full_df[['split']].reset_index(drop=True)
        }
    
    def optimize_hyperparameters(self, X_train, y_train, X_dev, y_dev):
        """Use Optuna to find optimal hyperparameters."""
        print("Optimizing hyperparameters with Optuna...", flush=True)
        
        def objective(trial):
            params = {
                'objective': 'None',  # Use custom objective
                'device': self.device_type,
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'verbosity': -1,
                'seed': self.config.get('seed', 42)
            }
            
            # Train models for each class and compute average RMSE
            # Use standard regression for hyperparameter optimization
            total_rmse = 0
            for class_idx in range(self.num_classes):
                train_data = lgb.Dataset(X_train, label=y_train[:, class_idx])
                valid_data = lgb.Dataset(X_dev, label=y_dev[:, class_idx], reference=train_data)
                
                # Use standard regression objective for optimization
                opt_params = params.copy()
                opt_params['objective'] = 'regression'
                opt_params['metric'] = 'rmse'
                
                model = lgb.train(
                    opt_params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_dev)
                rmse = np.sqrt(mean_squared_error(y_dev[:, class_idx], y_pred))
                total_rmse += rmse
            
            return total_rmse / self.num_classes
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        print(f"Best hyperparameters: {study.best_params}", flush=True)
        return study.best_params
    
    def train(self, data):
        """Train LightGBM models - one for each class probability."""
        print("Training surrogate models...", flush=True)
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_dev = data['X_dev']
        y_dev = data['y_dev']
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(X_train, y_train, X_dev, y_dev)
        
        # Train one model per class
        self.models = []
        for class_idx in range(self.num_classes):
            print(f"Training model for class {class_idx}...", flush=True)
            
            # Prepare data for this class
            train_data = lgb.Dataset(
                X_train, 
                label=y_train[:, class_idx],
                feature_name=self.feature_names
            )
            valid_data = lgb.Dataset(
                X_dev, 
                label=y_dev[:, class_idx], 
                reference=train_data,
                feature_name=self.feature_names
            )
            
            # Train model with best parameters using custom loss
            params = {
                'boosting_type': 'gbdt',
                'device': self.device_type,
                'verbosity': -1,
                'seed': self.config.get('seed', 42),
                **best_params
            }
            
            # Use custom cross-entropy loss function for final training
            try:
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(50),
                        lgb.log_evaluation(100)
                    ],
                    fobj=self.cross_entropy_loss,
                    feval=self.cross_entropy_eval
                )
            except TypeError:
                # Fallback to standard regression if fobj not supported
                print("Custom loss not supported, using standard regression", flush=True)
                params['objective'] = 'regression'
                params['metric'] = 'rmse'
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(50),
                        lgb.log_evaluation(100)
                    ]
                )
            
            self.models.append(model)
    
    def predict_probabilities(self, X):
        """Predict probability distributions."""
        if not self.models:
            raise ValueError("Models not trained yet")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Stack and normalize to ensure probabilities sum to 1
        prob_matrix = np.column_stack(predictions)
        
        # Apply softmax normalization
        exp_probs = np.exp(prob_matrix - np.max(prob_matrix, axis=1, keepdims=True))
        normalized_probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
        
        return normalized_probs
    
    def predict_classes(self, X):
        """Predict class labels."""
        probs = self.predict_probabilities(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, data):
        """Evaluate surrogate model performance."""
        print("Evaluating surrogate model...", flush=True)
        
        results = {}
        
        for split in ['train', 'dev', 'test']:
            X = data[f'X_{split}']
            y_true_probs = data[f'y_{split}']
            y_true_classes = np.argmax(y_true_probs, axis=1)
            
            # Predict with surrogate model
            y_pred_probs = self.predict_probabilities(X)
            y_pred_classes = self.predict_classes(X)
            
            # Classification metrics
            acc = accuracy_score(y_true_classes, y_pred_classes)
            f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
            
            # Probability distribution similarity metrics
            prob_mse = mean_squared_error(y_true_probs.flatten(), y_pred_probs.flatten())
            prob_r2 = r2_score(y_true_probs.flatten(), y_pred_probs.flatten())
            
            # Cross-entropy between distributions
            y_pred_clipped = np.clip(y_pred_probs, 1e-15, 1 - 1e-15)
            cross_entropy = -np.mean(y_true_probs * np.log(y_pred_clipped))
            
            results[split] = {
                'accuracy': acc,
                'f1_score': f1,
                'probability_mse': prob_mse,
                'probability_r2': prob_r2,
                'cross_entropy': cross_entropy
            }
            
            print(f"{split.upper()} - Accuracy: {acc:.4f}, F1: {f1:.4f}, Prob MSE: {prob_mse:.4f}, R²: {prob_r2:.4f}, CE: {cross_entropy:.4f}")
        
        return results
    
    def feature_importance(self):
        """Get feature importance from trained models."""
        if not self.models:
            raise ValueError("Models not trained yet")
        
        # Average importance across all class models
        importances = []
        for model in self.models:
            importance = model.feature_importance(importance_type='gain')
            importances.append(importance)
        
        avg_importance = np.mean(importances, axis=0)
        
        # Create DataFrame for easy viewing
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, path):
        """Save trained models."""
        save_data = {
            'models': self.models,
            'num_classes': self.num_classes,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Surrogate model saved to {path}")
    
    def load(self, path):
        """Load trained models."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.models = save_data['models']
        self.num_classes = save_data['num_classes']
        self.feature_names = save_data['feature_names']
        
        print(f"Surrogate model loaded from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Surrogate Model')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    
    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]
    
    # Initialize surrogate model
    surrogate = SurrogateModel(config)
    
    # Load data
    dataset, bert_outputs = surrogate.load_data()
    
    # Prepare features and targets
    data = surrogate.prepare_features_and_targets(dataset, bert_outputs)
    
    print(f"Feature matrix shape: {data['X_train'].shape}")
    print(f"Target probabilities shape: {data['y_train'].shape}")
    
    # Train surrogate model
    surrogate.train(data)
    
    # Evaluate performance
    results = surrogate.evaluate(data)
    
    # Show feature importance
    print("\nTop 10 Most Important Features:")
    importance_df = surrogate.feature_importance()
    print(importance_df.head(10))
    
    # Save model
    model_path = config.get("surrogate_model_path", "models/") + "surrogate_model.pkl"
    surrogate.save(model_path)
    
    print(f"\nSurrogate model training complete!")