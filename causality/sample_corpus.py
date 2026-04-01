"""
Samples a subset of documents from a corpus for narrative analysis.

Sampling strategy
-----------------
The goal is to produce a sample that has both good longitudinal spread (coverage
across the full date range of the corpus) and high event density (docs with more
events carry more causal signal and are more useful for narrative analysis).

These two goals are in tension: naively selecting the highest-event docs would
skew toward recent years when posts were longer and more frequent; naively
sampling uniformly at random would include many low-event docs that contribute
little to the analysis.

The strategy resolves this as follows:

1. Filter: drop docs below a minimum event threshold (--min_events). These docs
   have too little causal content to be worth analyzing.

2. Bin: divide the full date range of the corpus into N equal-width temporal
   bins (--n_bins, default 24 i.e. half-year bins). Each eligible doc is
   assigned to a bin based on its datetime.

3. Proportional quota: each bin is allocated a quota proportional to the number
   of eligible docs it contains, scaled to sum to --n_docs. This means dense
   periods (where many docs were posted) get more slots, and sparse periods
   (e.g. early years) are not over-sampled relative to what is available.

4. Rank within bin: within each bin, docs are ranked by event count descending.
   The top-quota docs are selected. This ensures that within any given time
   period we pick the docs with the richest event structure.

5. Shortfall redistribution: if a bin has fewer eligible docs than its quota
   (can happen in sparse early periods), all docs in that bin are taken and the
   unused quota is redistributed proportionally across the remaining bins.
   Redistribution is applied iteratively until no bin is over-quota.

The result is a sample that spans the full timeline of the corpus while
preferring event-dense documents within each time period.
"""

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import timezone

from dateutil import parser as dateutil_parser
from pyhocon import ConfigFactory


def count_events(doc):
    return sum(len(s['events']) for s in doc['sentences'].values())


def assign_bins(docs, n_bins):
    """Assign each doc to a temporal bin index in [0, n_bins)."""
    dates = {doc_id: dateutil_parser.parse(doc['datetime']) for doc_id, doc in docs.items()}

    min_date = min(dates.values())
    max_date = max(dates.values())
    total_span = (max_date - min_date).total_seconds()

    bin_assignments = defaultdict(list)
    for doc_id, date in dates.items():
        elapsed = (date - min_date).total_seconds()
        # Clamp max_date doc to last bin rather than bin index n_bins
        bin_idx = min(int(elapsed / total_span * n_bins), n_bins - 1)
        bin_assignments[bin_idx].append(doc_id)

    return bin_assignments


def compute_quotas(bin_assignments, n_docs, n_bins):
    """Compute per-bin sampling quotas proportional to bin size, summing to n_docs."""
    total_eligible = sum(len(docs) for docs in bin_assignments.values())
    quotas = {}
    for bin_idx, docs in bin_assignments.items():
        quotas[bin_idx] = len(docs) / total_eligible * n_docs
    # Distribute integer quotas using largest remainder method
    floors = {b: int(q) for b, q in quotas.items()}
    remainders = sorted(quotas.keys(), key=lambda b: -(quotas[b] - floors[b]))
    shortfall = n_docs - sum(floors.values())
    for i in range(shortfall):
        floors[remainders[i]] += 1
    return floors


def sample_with_redistribution(bin_assignments, quotas, event_counts, rng):
    """
    Select top-quota docs per bin by event count.
    Redistribute unused quota from under-populated bins iteratively.
    """
    # Sort each bin by event count descending
    sorted_bins = {
        b: sorted(docs, key=lambda d: event_counts[d], reverse=True)
        for b, docs in bin_assignments.items()
    }

    selected = []
    remaining_quotas = dict(quotas)

    # Iteratively fill bins; redistribute shortfalls until stable
    while True:
        shortfall = 0
        new_quotas = {}

        for bin_idx, docs in sorted_bins.items():
            quota = remaining_quotas.get(bin_idx, 0)
            available = len(docs)
            take = min(quota, available)
            shortfall += quota - take
            new_quotas[bin_idx] = take

        if shortfall == 0:
            break

        # Apply current round's selections, remove saturated bins
        remaining_quotas = {}
        total_remaining = sum(
            len(docs) - new_quotas[b]
            for b, docs in sorted_bins.items()
            if len(docs) > new_quotas[b]
        )

        if total_remaining == 0:
            break

        for bin_idx, docs in sorted_bins.items():
            take = new_quotas[bin_idx]
            available = len(docs)
            if available > take:
                # Redistribute proportionally to remaining capacity
                extra = int((available - take) / total_remaining * shortfall)
                remaining_quotas[bin_idx] = take + extra
            else:
                remaining_quotas[bin_idx] = take

        # Fix rounding in redistribution
        rounding_gap = sum(new_quotas.values()) + shortfall - sum(remaining_quotas.values())
        if rounding_gap > 0:
            eligible = [b for b in remaining_quotas if len(sorted_bins[b]) > remaining_quotas[b]]
            for b in eligible[:rounding_gap]:
                remaining_quotas[b] += 1

    # Final selection
    for bin_idx, take in new_quotas.items():
        docs = sorted_bins[bin_idx]
        selected.extend(docs[:take])

    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample documents for narrative analysis')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--n_docs', type=int, default=1000, metavar='N',
                        help='Number of documents to sample')
    parser.add_argument('--min_events', type=int, default=3, metavar='MIN_EVENTS',
                        help='Minimum number of events for a document to be eligible')
    parser.add_argument('--n_bins', type=int, default=24, metavar='N_BINS',
                        help='Number of temporal bins (default: 24, i.e. half-year bins)')
    args = parser.parse_args()
    print(vars(args))

    config = ConfigFactory.parse_file('./config.conf')[args.c]
    rng = random.Random(config['seed'])

    with open(config['corpus_path'], 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    print(f"Total docs in corpus: {len(corpus)}")

    # Step 1: filter by minimum event count
    event_counts = {doc_id: count_events(doc) for doc_id, doc in corpus.items()}
    eligible = {doc_id: doc for doc_id, doc in corpus.items()
                if event_counts[doc_id] >= args.min_events}

    print(f"Eligible docs (>= {args.min_events} events): {len(eligible)}")

    if len(eligible) < args.n_docs:
        raise ValueError(
            f"Only {len(eligible)} eligible docs but requested {args.n_docs}. "
            f"Lower --min_events or --n_docs."
        )

    # Step 2: assign docs to temporal bins
    bin_assignments = assign_bins(eligible, args.n_bins)
    print(f"Non-empty bins: {len(bin_assignments)}/{args.n_bins}")

    # Step 3: compute proportional quotas
    quotas = compute_quotas(bin_assignments, args.n_docs, args.n_bins)

    # Step 4 & 5: select top-event docs per bin, redistribute shortfalls
    selected_ids = sample_with_redistribution(bin_assignments, quotas, event_counts, rng)

    print(f"Selected {len(selected_ids)} documents")

    # Print bin summary
    print("\nBin summary (bin_idx: quota -> selected, total_eligible):")
    bin_selected = defaultdict(list)
    selected_set = set(selected_ids)
    for bin_idx, docs in bin_assignments.items():
        bin_selected[bin_idx] = [d for d in docs if d in selected_set]
    for bin_idx in sorted(bin_assignments.keys()):
        print(f"  Bin {bin_idx:02d}: quota={quotas.get(bin_idx, 0):3d}, "
              f"selected={len(bin_selected[bin_idx]):3d}, "
              f"eligible={len(bin_assignments[bin_idx]):3d}")

    total_pairs = sum(event_counts[d] * (event_counts[d] - 1) // 2 for d in selected_ids)
    print(f"\nTotal event pairs (no window) in sample: {total_pairs:,}")

    sampled_corpus = {doc_id: corpus[doc_id] for doc_id in selected_ids}

    with open(config['sampled_corpus_path'], 'w', encoding='utf-8') as f:
        json.dump(sampled_corpus, f)

    print(f"Saved to {config['sampled_corpus_path']}")
