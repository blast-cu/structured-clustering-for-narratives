import argparse
import json
import random
from collections import defaultdict


def sample_documents(documents_dict, sample_size=3000, ignore_frames=None):
    random.seed(42)

    # Group documents by their primary frame
    frame_groups = defaultdict(list)

    for doc_id, doc in documents_dict.items():
        # Skip documents that don't have a primary_frame field
        if 'primary_frame' not in doc:
            continue

        # Get the primary frame
        primary_frame = doc['primary_frame']

        # Skip Other primary frame
        if primary_frame == 'Other primary':
            continue

        frame_groups[primary_frame].append(doc_id)

    # Calculate how many documents to sample from each frame group
    unique_frames = list(frame_groups.keys())

    # Get counts for each frame
    frame_counts = {frame: len(docs) for frame, docs in frame_groups.items()}

    # Find the minimum number of documents in any frame group
    min_count = min(frame_counts.values())

    # Initial allocation: give each category at least the minimum count,
    # but cap at actual available count
    initial_allocation = {frame: min(min_count, frame_counts[frame]) for frame in unique_frames}

    # Calculate how many documents we've allocated so far
    allocated_so_far = sum(initial_allocation.values())

    # Calculate remaining items to distribute
    remaining_items = sample_size - allocated_so_far

    # If we need more documents than we can evenly distribute
    if remaining_items > 0:
        # Calculate proportions for remaining items based on counts above minimum
        total_above_min = sum(max(0, frame_counts[frame] - initial_allocation[frame])
                              for frame in unique_frames)

        # Distribute remaining items proportionally
        for frame in unique_frames:
            above_min = max(0, frame_counts[frame] - initial_allocation[frame])
            if total_above_min > 0:
                proportion = above_min / total_above_min
                additional_items = round(remaining_items * proportion)
                # Don't allocate more than available
                additional_items = min(additional_items, frame_counts[frame] - initial_allocation[frame])
                initial_allocation[frame] += additional_items

    # Step 3: Sample the calculated number of documents from each frame group
    sampled_documents = {}
    total_sampled = 0

    for frame, num_to_sample in initial_allocation.items():
        # Ensure we don't exceed the target sample size
        if total_sampled + num_to_sample > sample_size:
            num_to_sample = sample_size - total_sampled

        # If we have more documents than needed, randomly sample
        available_docs = frame_groups[frame]
        if len(available_docs) > num_to_sample:
            selected_doc_ids = random.sample(available_docs, num_to_sample)
        else:
            # If we need all documents in this frame, take them all
            selected_doc_ids = available_docs

        # Add selected documents to our result
        for doc_id in selected_doc_ids:
            sampled_documents[doc_id] = documents_dict[doc_id]

        total_sampled += len(selected_doc_ids)

        # If we've reached the target sample size, break
        if total_sampled >= sample_size:
            break

    # Print statistics about the sampling
    print(f"Requested sample size: {sample_size}", flush=True)
    print(f"Actual sample size: {len(sampled_documents)}", flush=True)

    # Count frames in sample for validation
    sampled_frame_counts = defaultdict(int)
    for doc_id, doc in sampled_documents.items():
        primary_frame = doc['primary_frame']
        if isinstance(primary_frame, (list, tuple)):
            primary_frame = primary_frame[0]
        sampled_frame_counts[primary_frame] += 1

    print("\nFrame distribution in sample:", flush=True)
    for frame in sorted(sampled_frame_counts.keys()):
        print(f"{frame}: {sampled_frame_counts[frame]} documents", flush=True)

    return sampled_documents


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