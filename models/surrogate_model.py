import argparse
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from pyhocon import ConfigFactory
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
import optuna


class SurrogateModel:
    """LightGBM-based surrogate model that approximates BERT using handcrafted features."""
    
    def __init__(self, config):
        self.config = config
        self.model = None  # Single multi-class model
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
        all_predictions = []
        
        splits = ['train', 'dev', 'test']
        for split in splits:
            df = dataset[f'{split}_df'].copy()
            df['split'] = split
            all_dfs.append(df)
            
            # Find corresponding probabilities and predictions
            for prob_data in bert_outputs['probabilities']:
                if prob_data['split'] == split:
                    all_probabilities.extend(prob_data['probabilities'])
                    break
            
            for pred_data in bert_outputs['predictions']:
                if pred_data['split'] == split:
                    all_predictions.extend(pred_data['predictions'])
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
        y_classes = np.array(all_predictions)  # Use BERT predictions as class labels
        
        self.feature_names = feature_names
        
        # Split back into train/dev/test
        train_mask = full_df['split'] == 'train'
        dev_mask = full_df['split'] == 'dev'  
        test_mask = full_df['split'] == 'test'
        
        return {
            'X_train': X[train_mask],
            'X_dev': X[dev_mask], 
            'X_test': X[test_mask],
            'y_train_probs': y_probs[train_mask],
            'y_dev_probs': y_probs[dev_mask],
            'y_test_probs': y_probs[test_mask],
            'y_train_classes': y_classes[train_mask],
            'y_dev_classes': y_classes[dev_mask],
            'y_test_classes': y_classes[test_mask],
            'splits': full_df[['split']].reset_index(drop=True)
        }
    
    def optimize_hyperparameters(self, X_train, y_train_classes, X_dev, y_dev_classes):
        """Use Optuna to find optimal hyperparameters for multi-class model."""
        print("Optimizing hyperparameters with Optuna...", flush=True)
        
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': self.num_classes,
                'metric': 'multi_logloss',
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
            
            # Train single multi-class model
            train_data = lgb.Dataset(X_train, label=y_train_classes, feature_name=self.feature_names)
            valid_data = lgb.Dataset(X_dev, label=y_dev_classes, reference=train_data, feature_name=self.feature_names)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Return validation loss (lower is better)
            return model.best_score['valid_0']['multi_logloss']
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)  # Increased trials for better optimization
        
        print(f"Best hyperparameters: {study.best_params}", flush=True)
        return study.best_params
    
    def train(self, data):
        """Train single multi-class LightGBM model."""
        print("Training surrogate model...", flush=True)
        
        X_train = data['X_train']
        y_train_classes = data['y_train_classes']  # Use BERT class predictions
        X_dev = data['X_dev']
        y_dev_classes = data['y_dev_classes']
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(X_train, y_train_classes, X_dev, y_dev_classes)
        
        print("Training final multi-class model...", flush=True)
        
        # Prepare data
        train_data = lgb.Dataset(
            X_train, 
            label=y_train_classes,
            feature_name=self.feature_names
        )
        valid_data = lgb.Dataset(
            X_dev, 
            label=y_dev_classes, 
            reference=train_data,
            feature_name=self.feature_names
        )
        
        # Train model with best parameters
        params = {
            'objective': 'multiclass',
            'num_class': self.num_classes,
            'metric': 'multi_logloss',
            'device': self.device_type,
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': self.config.get('seed', 42),
            **best_params
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )
    
    def predict_probabilities(self, X):
        """Predict probability distributions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # LightGBM multi-class returns probabilities directly
        probabilities = self.model.predict(X)
        return probabilities
    
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
            y_true_probs = data[f'y_{split}_probs']  # BERT probabilities
            y_true_classes = data[f'y_{split}_classes']  # BERT predictions
            
            # Predict with surrogate model
            y_pred_probs = self.predict_probabilities(X)
            y_pred_classes = self.predict_classes(X)
            
            # Classification metrics (comparing predicted vs BERT classes)
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
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get feature importance from single multi-class model
        importance = self.model.feature_importance(importance_type='gain')
        
        # Create DataFrame for easy viewing
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, path):
        """Save trained model."""
        save_data = {
            'model': self.model,
            'num_classes': self.num_classes,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Surrogate model saved to {path}")
    
    def load(self, path):
        """Load trained model."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data['model']
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
    print(f"Target probabilities shape: {data['y_train_probs'].shape}")
    print(f"Target classes shape: {data['y_train_classes'].shape}")
    
    # Add diagnostics
    print("\n=== FEATURE DIAGNOSTICS ===")
    X_train = data['X_train']
    y_train = data['y_train_classes']
    
    print(f"Feature statistics:")
    print(f"  - Mean: {np.mean(X_train, axis=0)[:10]}...")  # First 10 features
    print(f"  - Std: {np.std(X_train, axis=0)[:10]}...") 
    print(f"  - Non-zero ratio: {np.mean(X_train != 0, axis=0)[:10]}...")  # Sparsity check
    
    print(f"\nClass distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, (cls, count) in enumerate(zip(unique, counts)):
        print(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.1f}%)")
        if i >= 5:  # Show first 6 classes
            print(f"  ... and {len(unique)-6} more classes")
            break
    
    # Check if features correlate with targets at all
    from sklearn.feature_selection import f_classif
    f_scores, p_values = f_classif(X_train, y_train)
    significant_features = np.sum(p_values < 0.05)
    print(f"\nFeature-target correlation:")
    print(f"  - Features with p<0.05: {significant_features}/{len(p_values)}")
    print(f"  - Top 5 F-scores: {np.sort(f_scores)[-5:][::-1]}")
    
    # Simple baseline check
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
    baseline_lr.fit(X_train, y_train)
    baseline_pred = baseline_lr.predict(data['X_dev'])
    baseline_f1 = f1_score(data['y_dev_classes'], baseline_pred, average='weighted')
    print(f"\nBaseline Logistic Regression F1: {baseline_f1:.4f}")
    print("==============================\n")
    
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