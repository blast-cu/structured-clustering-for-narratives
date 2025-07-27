#!/usr/bin/env python3
"""
Script to find matching predictions between neural net and BERT models.
Extracts examples where both models agree on predictions and saves them for SHAP comparison.
"""

import argparse
import pickle
import json
import numpy as np
import torch
from pyhocon import ConfigFactory
from models.neural_net import NeuralNetTrainer
from models.bert import Trainer as BertTrainer


def load_neural_net_predictions(config):
    """Load neural net model and get predictions on dev + test sets."""
    print("Loading neural net model and getting predictions...")
    
    trainer = NeuralNetTrainer(config)
    train_df, dev_df, test_df, label_encoder, train_dataloader, dev_dataloader, test_dataloader = trainer.load_dataset()
    
    # Load or train the model (assuming it's already trained)
    # For now, we'll assume we need to get predictions by running inference
    trainer.model.eval()
    
    # Collect dev + test data in same order as neural_net.py analyze_feature_importance
    eval_cluster_feats = []
    eval_role_feats = []
    eval_stance_feats = []
    eval_labels = []
    eval_texts = []
    eval_indices = []  # Track original indices
    
    current_idx = 0
    
    # Process dev set first (same order as neural_net.py)
    for cluster_feats, role_feats, stance_feats, labels in dev_dataloader:
        batch_size = cluster_feats.shape[0]
        eval_cluster_feats.append(cluster_feats)
        eval_role_feats.append(role_feats)
        eval_stance_feats.append(stance_feats)
        eval_labels.extend(labels.numpy())
        
        # Get corresponding text from dev_df
        start_idx = current_idx
        end_idx = current_idx + batch_size
        eval_texts.extend(dev_df['text'].iloc[start_idx:end_idx].tolist())
        eval_indices.extend([('dev', i) for i in range(start_idx, end_idx)])
        current_idx += batch_size
    
    # Reset for test set
    current_idx = 0
    
    # Process test set second (same order as neural_net.py)
    for cluster_feats, role_feats, stance_feats, labels in test_dataloader:
        batch_size = cluster_feats.shape[0]
        eval_cluster_feats.append(cluster_feats)
        eval_role_feats.append(role_feats)
        eval_stance_feats.append(stance_feats)
        eval_labels.extend(labels.numpy())
        
        # Get corresponding text from test_df
        start_idx = current_idx
        end_idx = current_idx + batch_size
        eval_texts.extend(test_df['text'].iloc[start_idx:end_idx].tolist())
        eval_indices.extend([('test', i) for i in range(start_idx, end_idx)])
        current_idx += batch_size
    
    # Combine features for prediction
    sample_cluster = torch.cat(eval_cluster_feats, dim=0).to(trainer.device)
    sample_role = torch.cat(eval_role_feats, dim=0).to(trainer.device)
    sample_stance = torch.cat(eval_stance_feats, dim=0).to(trainer.device)
    eval_labels = np.array(eval_labels)
    
    # Get neural net predictions
    with torch.no_grad():
        outputs = trainer.model(sample_cluster, sample_role, sample_stance)
        nn_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    return {
        'predictions': nn_predictions,
        'labels': eval_labels,
        'texts': eval_texts,
        'indices': eval_indices,  # (split, index_within_split)
        'label_encoder': label_encoder,
        'dev_df': dev_df,
        'test_df': test_df
    }


def load_bert_predictions(config):
    """Load BERT predictions from saved file."""
    print("Loading BERT predictions...")
    
    bert_predictions_path = config["frame_prediction_data_path"] + "bert_predictions.pickle"
    
    try:
        with open(bert_predictions_path, 'rb') as f:
            bert_data = pickle.load(f)
    except FileNotFoundError:
        print(f"BERT predictions file not found: {bert_predictions_path}")
        print("You need to run bert.py first to generate predictions.")
        return None
    
    # Extract dev and test predictions
    predictions_by_split = {}
    for pred_data in bert_data['predictions']:
        split = pred_data['split']
        predictions = pred_data['predictions']
        predictions_by_split[split] = np.array(predictions)
    
    return predictions_by_split


def find_matching_predictions(nn_data, bert_predictions):
    """Find examples where both models agree on predictions."""
    print("Finding matching predictions between neural net and BERT...")
    
    if bert_predictions is None:
        return None
    
    nn_predictions = nn_data['predictions']
    nn_labels = nn_data['labels']
    nn_texts = nn_data['texts']
    nn_indices = nn_data['indices']
    label_encoder = nn_data['label_encoder']
    
    matching_examples = []
    
    for i, (nn_pred, true_label, text, (split, split_idx)) in enumerate(zip(
        nn_predictions, nn_labels, nn_texts, nn_indices
    )):
        # Get corresponding BERT prediction
        if split in bert_predictions and split_idx < len(bert_predictions[split]):
            bert_pred = bert_predictions[split][split_idx]
            
            # Check if predictions match
            if nn_pred == bert_pred:
                matching_examples.append({
                    'neural_net_index': i,  # Index for SHAP retrieval in neural_net
                    'bert_split': split,
                    'bert_index': split_idx,  # Index for SHAP retrieval in BERT
                    'text': text,
                    'true_label': int(true_label),
                    'true_label_name': label_encoder.classes_[true_label],
                    'predicted_label': int(nn_pred),
                    'predicted_label_name': label_encoder.classes_[nn_pred],
                    'prediction_correct': bool(nn_pred == true_label)
                })
    
    print(f"Found {len(matching_examples)} examples where both models agree")
    
    # Summary statistics
    total_examples = len(nn_predictions)
    correct_matches = sum(1 for ex in matching_examples if ex['prediction_correct'])
    incorrect_matches = len(matching_examples) - correct_matches
    
    print(f"Total examples: {total_examples}")
    print(f"Matching predictions (both models agree): {len(matching_examples)}")
    print(f"  - Correctly predicted: {correct_matches}")
    print(f"  - Incorrectly predicted: {incorrect_matches}")
    print(f"Agreement rate: {len(matching_examples)/total_examples*100:.2f}%")
    
    return matching_examples


def save_matching_examples(matching_examples, output_path):
    """Save matching examples to JSON file."""
    print(f"Saving {len(matching_examples)} matching examples to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(matching_examples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_path}")
    
    # Also save a summary file
    summary_path = output_path.replace('.json', '_summary.json')
    
    summary = {
        'total_matching_examples': len(matching_examples),
        'correct_predictions': sum(1 for ex in matching_examples if ex['prediction_correct']),
        'incorrect_predictions': sum(1 for ex in matching_examples if not ex['prediction_correct']),
        'label_distribution': {}
    }
    
    # Count predictions by label
    for ex in matching_examples:
        label_name = ex['predicted_label_name']
        if label_name not in summary['label_distribution']:
            summary['label_distribution'][label_name] = {'correct': 0, 'incorrect': 0}
        
        if ex['prediction_correct']:
            summary['label_distribution'][label_name]['correct'] += 1
        else:
            summary['label_distribution'][label_name]['incorrect'] += 1
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Find matching predictions between neural net and BERT models')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--output', default='matching_predictions.json', help='Output JSON file')
    
    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]
    
    # Load neural net predictions
    nn_data = load_neural_net_predictions(config)
    
    # Load BERT predictions
    bert_predictions = load_bert_predictions(config)
    
    if bert_predictions is None:
        print("Cannot proceed without BERT predictions. Please run bert.py first.")
        return
    
    # Find matching predictions
    matching_examples = find_matching_predictions(nn_data, bert_predictions)
    
    if matching_examples:
        # Save results
        save_matching_examples(matching_examples, args.output)
        
        print("\nUsage for SHAP comparison:")
        print("1. For neural_net.py SHAP values: use 'neural_net_index' to index into the SHAP results")
        print("2. For bert.py SHAP values: use 'bert_split' and 'bert_index' to locate the example")
        print("   - Make sure to compute SHAP values for the same dev/test splits")
    else:
        print("No matching examples found or error occurred.")


if __name__ == "__main__":
    main()