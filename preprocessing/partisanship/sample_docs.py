import argparse
import json
import random
from collections import defaultdict


def sample_documents(documents_dict, sample_size, label_key="label", label_values=("left", "right")):
    random.seed(42)

    # Group documents by their label
    label_groups = defaultdict(list)

    for doc_id, doc in documents_dict.items():
        # Skip documents that don't have the label field
        if label_key not in doc:
            continue

        # Get the label
        label = doc[label_key]

        # Skip documents with labels not in label_values
        if label not in label_values:
            continue

        label_groups[label].append(doc_id)

    # Calculate how many documents to sample from each label group
    target_per_label = sample_size // len(label_values)

    # Adjust if we can't get exactly sample_size due to integer division
    remainder = sample_size % len(label_values)

    # Sample documents from each label group
    sampled_documents = {}

    for i, label in enumerate(label_values):
        # Determine number to sample for this label
        num_to_sample = target_per_label + (1 if i < remainder else 0)

        available_docs = label_groups.get(label, [])

        # If we don't have enough documents with this label
        if len(available_docs) < num_to_sample:
            # Take all available documents with this label
            selected_doc_ids = available_docs
            print(
                f"Warning: Only {len(available_docs)} documents available with label '{label}', requested {num_to_sample}", flush=True)
        else:
            # Randomly sample wit4hout replacement
            selected_doc_ids = random.sample(available_docs, num_to_sample)

        # Add selected documents to our result
        for doc_id in selected_doc_ids:
            sampled_documents[doc_id] = documents_dict[doc_id]

    # Print statistics about the sampling
    print(f"Requested sample size: {sample_size}", flush=True)
    print(f"Actual sample size: {len(sampled_documents)}", flush=True)

    # Count labels in sample for validation
    sampled_label_counts = defaultdict(int)
    for doc_id, doc in sampled_documents.items():
        if label_key in doc:
            sampled_label_counts[doc[label_key]] += 1

    print("\nLabel distribution in sample:", flush=True)
    for label in label_values:
        print(f"{label}: {sampled_label_counts.get(label, 0)} documents", flush=True)

    return sampled_documents


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="processed corpus file")
    parser.add_argument("--sample_size", type=int, default=3000, help="number of documents to sample")
    parser.add_argument("--output_file", help="output file with sampled dataset")
    args = parser.parse_args()
    print(vars(args), flush=True)

    with open(args.corpus, 'r') as f:
        documents = json.load(f)

    sampled_docs = sample_documents(documents, sample_size=args.sample_size)

    with open(args.output_file, 'w') as f:
        json.dump(sampled_docs, f, indent=2)