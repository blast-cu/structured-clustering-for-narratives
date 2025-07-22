import argparse
import pickle
import random
import warnings

import numpy as np
import pandas as pd
from pyhocon import ConfigFactory
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import optuna


class LinearSurrogateModel:
    """Linear surrogate model for interpreting BERT predictions using Elastic Net."""
    
    def __init__(self, config, num_classes=14, approach='one_vs_rest',
                 train_on_probs=True, use_gold_labels=False, use_cluster_feats=True, use_role_feats=True, 
                 use_stance_feats=True, use_frequency_features=False, seed=42, n_trials=50):
        self.config = config
        self.model = None
        self.approach = approach
        self.train_on_probs = train_on_probs
        self.use_gold_labels = use_gold_labels
        self.num_classes = num_classes
        self.use_cluster_feats = use_cluster_feats
        self.use_role_feats = use_role_feats
        self.use_stance_feats = use_stance_feats
        self.use_frequency_features = use_frequency_features
        self.n_trials = n_trials
        self.feature_names = None
        self.scaler = StandardScaler()
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Suppress sklearn warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
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
        """Prepare feature matrix and targets from loaded data."""
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
        if self.use_cluster_feats:
            if self.use_frequency_features:
                # Use original normalized frequency features
                cluster_feats_raw = np.array(full_df['cluster_feats_frequency'].tolist())
                # Normalize frequency features
                frequency_scaler = StandardScaler()
                cluster_feats = frequency_scaler.fit_transform(cluster_feats_raw)
                feature_names.extend([f'cluster_freq_{i}' for i in range(cluster_feats.shape[1])])
                print(f"Using frequency-based cluster features: {cluster_feats.shape[1]} clusters", flush=True)
            else:
                # Use structural features
                cluster_feats = np.array(full_df['cluster_feats'].tolist())
                if 'feature_names' in dataset:
                    feature_names.extend(dataset['feature_names'])
                else:
                    feature_names.extend([f'cluster_{i}' for i in range(cluster_feats.shape[1])])
                print(f"Using structural cluster features: {cluster_feats.shape[1]} features", flush=True)
            
            features.append(cluster_feats)
        
        # Role features  
        if self.use_role_feats:
            role_feats = np.array(full_df['role_feats'].tolist())
            features.append(role_feats)
            feature_names.extend([f'role_{i}' for i in range(role_feats.shape[1])])
            
        # Stance features
        if self.use_stance_feats:
            stance_feats = np.array(full_df['stance_feats'].tolist())
            features.append(stance_feats)
            feature_names.extend([f'stance_{i}' for i in range(stance_feats.shape[1])])
        
        # Combine all features
        X = np.hstack(features) if features else np.array([])
        y_probs = np.array(all_probabilities)
        y_classes = np.array(all_predictions)
        
        # Extract gold labels
        y_gold_labels = np.array(full_df['frame_label_encoded'].tolist())
        
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
            'y_train_gold': y_gold_labels[train_mask],
            'y_dev_gold': y_gold_labels[dev_mask],
            'y_test_gold': y_gold_labels[test_mask]
        }
    
    def optimize_hyperparameters(self, X_train, y_train, X_dev, y_dev):
        """Use Optuna to find optimal hyperparameters."""
        print("Optimizing hyperparameters with Optuna...", flush=True)
        
        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.001, 10.0, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.01, 0.99)
            
            if self.approach == 'multiclass':
                # Multi-class logistic regression with elastic net penalty
                model = LogisticRegression(
                    penalty='elasticnet',
                    C=1/alpha,
                    l1_ratio=l1_ratio,
                    solver='saga',
                    multi_class='multinomial',
                    max_iter=2000,
                    random_state=42
                )
                
                if self.use_gold_labels or not self.train_on_probs:
                    y_train_weighted = y_train
                    y_dev_weighted = y_dev
                else:
                    y_train_weighted = np.argmax(y_train, axis=1)
                    y_dev_weighted = np.argmax(y_dev, axis=1)
                
                model.fit(X_train, y_train_weighted)
                y_pred = model.predict(X_dev)
                score = f1_score(y_dev_weighted, y_pred, average='weighted')
                
            else:  # one_vs_rest
                if self.train_on_probs and not self.use_gold_labels:
                    # Train on probability values - use regression
                    scores = []
                    for class_idx in range(self.num_classes):
                        model = ElasticNet(
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                            max_iter=2000,
                            random_state=42
                        )
                        model.fit(X_train, y_train[:, class_idx])
                        y_pred = model.predict(X_dev)
                        # Use negative MSE as score (higher is better)
                        mse = mean_squared_error(y_dev[:, class_idx], y_pred)
                        scores.append(-mse)
                    score = np.mean(scores)
                else:
                    # Train on class labels (either gold or BERT) - use classification
                    model = OneVsRestClassifier(
                        LogisticRegression(
                            penalty='elasticnet',
                            C=1/alpha,
                            l1_ratio=l1_ratio,
                            solver='saga',
                            max_iter=2000,
                            random_state=42
                        )
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_dev)
                    score = f1_score(y_dev, y_pred, average='weighted')
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        print(f"Best hyperparameters: {study.best_params}", flush=True)
        return study.best_params
    
    def train(self, data):
        """Train linear surrogate model."""
        print(f"Training {self.approach} linear surrogate model...", flush=True)
        
        if self.use_gold_labels:
            print("Training on gold labels...", flush=True)
        elif self.train_on_probs:
            print("Training on BERT probabilities...", flush=True)  
        else:
            print("Training on BERT class labels...", flush=True)
        
        X_train = data['X_train']
        X_dev = data['X_dev']
        
        if self.use_gold_labels:
            y_train = data['y_train_gold']
            y_dev = data['y_dev_gold']
        elif self.train_on_probs:
            y_train = data['y_train_probs']
            y_dev = data['y_dev_probs']
        else:
            y_train = data['y_train_classes']
            y_dev = data['y_dev_classes']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_dev_scaled = self.scaler.transform(X_dev)
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(X_train_scaled, y_train, X_dev_scaled, y_dev)
        
        print("Training final model...", flush=True)
        
        if self.approach == 'multiclass':
            self.model = LogisticRegression(
                penalty='elasticnet',
                C=1/best_params['alpha'],
                l1_ratio=best_params['l1_ratio'],
                solver='saga',
                multi_class='multinomial',
                max_iter=2000,
                random_state=42
            )
            
            if self.use_gold_labels or not self.train_on_probs:
                y_train_final = y_train
            else:  # self.train_on_probs and not self.use_gold_labels
                y_train_final = np.argmax(y_train, axis=1)
                
            self.model.fit(X_train_scaled, y_train_final)
            
        else:  # one_vs_rest
            if self.train_on_probs and not self.use_gold_labels:
                # Train separate regression models for each class probability
                self.models = []
                for class_idx in range(self.num_classes):
                    print(f"Training model for class {class_idx} probability...", flush=True)
                    model = ElasticNet(
                        alpha=best_params['alpha'],
                        l1_ratio=best_params['l1_ratio'],
                        max_iter=2000,
                        random_state=42
                    )
                    model.fit(X_train_scaled, y_train[:, class_idx])
                    self.models.append(model)
            else:
                # Train one-vs-rest classification (gold or BERT class labels)
                self.model = OneVsRestClassifier(
                    LogisticRegression(
                        penalty='elasticnet',
                        C=1/best_params['alpha'],
                        l1_ratio=best_params['l1_ratio'],
                        solver='saga',
                        max_iter=2000,
                        random_state=42
                    )
                )
                self.model.fit(X_train_scaled, y_train)
    
    def predict_probabilities(self, X):
        """Predict probability distributions."""
        X_scaled = self.scaler.transform(X)
        
        if self.approach == 'multiclass':
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X_scaled)
            else:
                return self._predictions_to_probs(self.model.predict(X_scaled))
                
        else:  # one_vs_rest
            if self.train_on_probs and not self.use_gold_labels:
                # Get predictions from each regression model
                predictions = []
                for model in self.models:
                    pred = model.predict(X_scaled)
                    predictions.append(pred)
                
                # Stack and normalize to probabilities
                prob_matrix = np.column_stack(predictions)
                prob_matrix = np.maximum(prob_matrix, 0)
                prob_sums = np.sum(prob_matrix, axis=1, keepdims=True)
                prob_sums = np.maximum(prob_sums, 1e-10)
                normalized_probs = prob_matrix / prob_sums
                
                return normalized_probs
            else:
                # Either gold labels or BERT class labels - use classification model
                return self.model.predict_proba(X_scaled)
    
    def predict_classes(self, X):
        """Predict class labels."""
        if self.approach == 'one_vs_rest' and self.train_on_probs and not self.use_gold_labels:
            probs = self.predict_probabilities(X)
            return np.argmax(probs, axis=1)
        else:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
    
    def _predictions_to_probs(self, predictions):
        """Convert predictions to probability distributions using softmax."""
        if predictions.ndim == 1:
            probs = np.zeros((len(predictions), self.num_classes))
            probs[np.arange(len(predictions)), predictions.astype(int)] = 1.0
            return probs
        else:
            exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    
    def evaluate(self, data):
        """Evaluate linear surrogate model performance."""
        print("Evaluating linear surrogate model...", flush=True)
        
        results = {}
        
        for split in ['train', 'dev', 'test']:
            X = data[f'X_{split}']
            
            # Use appropriate ground truth based on training mode
            if self.use_gold_labels:
                y_true_classes = data[f'y_{split}_gold']
                # For gold labels, use BERT probs as reference for prob metrics
                y_true_probs = data[f'y_{split}_probs']
            else:
                y_true_probs = data[f'y_{split}_probs']
                y_true_classes = data[f'y_{split}_classes']
            
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
    
    def get_feature_importance(self):
        """Get feature importance/coefficients from trained model."""
        if self.approach == 'multiclass':
            if hasattr(self.model, 'coef_'):
                coef_df = pd.DataFrame(
                    self.model.coef_.T,
                    columns=[f'class_{i}' for i in range(self.num_classes)],
                    index=self.feature_names
                )
                return coef_df
                
        else:  # one_vs_rest
            if self.train_on_probs and hasattr(self, 'models'):
                coef_matrix = []
                for model in self.models:
                    if hasattr(model, 'coef_'):
                        coef_matrix.append(model.coef_)
                    else:
                        coef_matrix.append(np.zeros(len(self.feature_names)))
                
                coef_df = pd.DataFrame(
                    np.array(coef_matrix).T,
                    columns=[f'class_{i}' for i in range(self.num_classes)],
                    index=self.feature_names
                )
                return coef_df
                
            elif hasattr(self.model, 'estimators_'):
                coef_matrix = []
                for estimator in self.model.estimators_:
                    if hasattr(estimator, 'coef_'):
                        coef_matrix.append(estimator.coef_.flatten())
                    else:
                        coef_matrix.append(np.zeros(len(self.feature_names)))
                
                coef_df = pd.DataFrame(
                    np.array(coef_matrix).T,
                    columns=[f'class_{i}' for i in range(self.num_classes)],
                    index=self.feature_names
                )
                return coef_df
        
        return None
    
    def interpret_coefficients(self):
        """Provide human-readable interpretation of model coefficients."""
        coef_df = self.get_feature_importance()
        if coef_df is None:
            print("No coefficients available for interpretation.")
            return None
        
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        for class_idx in range(min(self.num_classes, 5)):  # Show first 5 classes
            class_col = f'class_{class_idx}'
            print(f"\nClass {class_idx} - Top Positive Predictors:")
            
            class_coefs = coef_df[class_col].sort_values(ascending=False)
            
            # Top positive predictors
            top_positive = class_coefs.head(5)
            for feature, coef in top_positive.items():
                if abs(coef) > 1e-6:
                    print(f"  {feature}: {coef:.4f}")
            
            # Top negative predictors
            print(f"\nClass {class_idx} - Top Negative Predictors:")
            top_negative = class_coefs.tail(5)
            for feature, coef in top_negative.items():
                if abs(coef) > 1e-6:
                    print(f"  {feature}: {coef:.4f}")
        
        print("\n===============================\n")
        return coef_df
    
    def save(self, path):
        """Save trained model."""
        save_data = {
            'model': self.model if hasattr(self, 'model') else None,
            'models': self.models if hasattr(self, 'models') else None,
            'scaler': self.scaler,
            'approach': self.approach,
            'train_on_probs': self.train_on_probs,
            'num_classes': self.num_classes,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Linear surrogate model saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Linear Surrogate Model')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--approach', choices=['multiclass', 'one_vs_rest'], 
                        default='one_vs_rest',
                        help='Training approach (default: one_vs_rest)')
    parser.add_argument('--train-on-probs', action='store_true',
                        help='Train on BERT probability distributions instead of class labels')
    parser.add_argument('--use-gold-labels', action='store_true',
                        help='Train on gold labels instead of BERT predictions (overrides --train-on-probs)')
    parser.add_argument('--num-classes', type=int, default=14,
                        help='Number of classes (default: 14)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials for hyperparameter optimization (default: 50)')
    parser.add_argument('--no-cluster-feats', action='store_true',
                        help='Disable cluster features')
    parser.add_argument('--no-role-feats', action='store_true', 
                        help='Disable role features')
    parser.add_argument('--no-stance-feats', action='store_true',
                        help='Disable stance features')
    parser.add_argument('--use-frequency-features', action='store_true',
                        help='Use original normalized frequency counts instead of structural features')
    parser.add_argument('--output-path', default='models/linear_surrogate_model.pkl',
                        help='Output path for saved model (default: models/linear_surrogate_model.pkl)')
    
    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]
    
    print(f"Training approach: {args.approach}")
    if args.use_gold_labels:
        print("Training target: gold labels")
    elif args.train_on_probs:
        print("Training target: BERT probabilities")
    else:
        print("Training target: BERT class labels")
    print(f"Feature type: {'frequency counts' if args.use_frequency_features else 'structural features'}")
    print(f"Features: cluster={not args.no_cluster_feats}, role={not args.no_role_feats}, stance={not args.no_stance_feats}")
    
    # Initialize linear surrogate model
    surrogate = LinearSurrogateModel(
        config=config,
        num_classes=args.num_classes,
        approach=args.approach,
        train_on_probs=args.train_on_probs,
        use_gold_labels=args.use_gold_labels,
        use_cluster_feats=not args.no_cluster_feats,
        use_role_feats=not args.no_role_feats,
        use_stance_feats=not args.no_stance_feats,
        use_frequency_features=args.use_frequency_features,
        seed=args.seed,
        n_trials=args.n_trials
    )
    
    # Load data
    dataset, bert_outputs = surrogate.load_data()
    
    # Prepare features and targets
    data = surrogate.prepare_features_and_targets(dataset, bert_outputs)
    
    print(f"Feature matrix shape: {data['X_train'].shape}")
    print(f"Target probabilities shape: {data['y_train_probs'].shape}")
    print(f"Target classes shape: {data['y_train_classes'].shape}")
    
    # Train surrogate model
    surrogate.train(data)
    
    # Evaluate performance
    results = surrogate.evaluate(data)
    
    # Show feature importance/interpretation
    coef_df = surrogate.interpret_coefficients()
    
    # Save model
    surrogate.save(args.output_path)
    
    print(f"\nLinear surrogate model training complete!")
    print(f"Approach: {args.approach}")
    if args.use_gold_labels:
        print("Training target: gold labels")
    elif args.train_on_probs:
        print("Training target: BERT probabilities")
    else:
        print("Training target: BERT class labels")
    print(f"Model saved to: {args.output_path}")