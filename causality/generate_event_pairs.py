import argparse
import json
import pickle
import re

from itertools import combinations
from pyhocon import ConfigFactory


def strip_verb(verb):
    """Strip the disambiguating suffix from a verb. e.g. 'diagnose_1' -> 'diagnose'"""
    return re.sub(r'_-?\d+$', '', verb)


def generate_event_pairs(data, sentence_window=2):
    """Generate event pairs within the sentence window for each document.

    Returns a dict keyed by document ID, each containing the document text
    and a list of event pairs.
    """
    dataset = {}

    for doc_id, doc in data.items():
        # Collect all events with their sentence IDs, sorted by sentence order
        events = []
        for sent_id, sentence in doc['sentences'].items():
            for event in sentence['events']:
                events.append({
                    'event_id': event[0],
                    'verb': strip_verb(event[1]),
                    'object': event[2],
                    'sentence_id': int(sent_id),
                    'sentence_text': sentence['text']
                })
        events.sort(key=lambda e: e['sentence_id'])

        # Generate pairs within the sentence window.
        # Since events are sorted by sentence_id, combinations() guarantees
        # event_1 comes before event_2 in the document.
        event_pairs = []
        for e1, e2 in combinations(events, 2):
            if abs(e1['sentence_id'] - e2['sentence_id']) <= sentence_window:
                event_pairs.append({
                    'event_1': {
                        'event_id': e1['event_id'],
                        'verb': e1['verb'],
                        'object': e1['object'],
                        'sentence_id': e1['sentence_id'],
                        'sentence_text': e1['sentence_text']
                    },
                    'event_2': {
                        'event_id': e2['event_id'],
                        'verb': e2['verb'],
                        'object': e2['object'],
                        'sentence_id': e2['sentence_id'],
                        'sentence_text': e2['sentence_text']
                    }
                })

        if event_pairs:
            dataset[doc_id] = {
                'corpus_id': doc['corpus_id'],
                'text': doc['text'],
                'event_pairs': event_pairs
            }

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate event pairs dataset for causality identification')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--sentence_window', type=int, default=2, metavar='WINDOW',
                        help='Max sentence distance between events in a pair')
    args = parser.parse_args()
    print(vars(args))
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    with open(config["sampled_corpus_path"], 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = generate_event_pairs(data, args.sentence_window)

    total_pairs = sum(len(doc['event_pairs']) for doc in dataset.values())
    print(f"Documents with event pairs: {len(dataset)}/{len(data)}")
    print(f"Total event pairs: {total_pairs}")
    print(f"Avg pairs per doc: {total_pairs / len(dataset):.1f}")

    with open(config["event_pairs_path"], 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved to {config['event_pairs_path']}")
