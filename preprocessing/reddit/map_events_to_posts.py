import argparse
import json
import pickle as pk
import re

from tqdm import tqdm

output_dict = {}

def get_line_from_corpus(file, line_number):
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i == line_number:
                return line

def generate_event_2_posts_map(reddit_corpus, processed_corpus, po_tuple_features, doc_2_sent):
    print('inside generate_event_2_generate_event_2_posts_map')

    with open(reddit_corpus, 'r') as f:
        reddit_corpus_dict = json.load(f)

    with open(po_tuple_features, 'rb') as svos_pkl_file:
        mydict = pk.load(svos_pkl_file)
        sent_id_2_vocab_freq = mydict['sent_id_2_vocab_freq']
        with open(doc_2_sent, 'rb') as json_file:
            doc_id_2_sent_ids = json.load(json_file)
            for doc_id in tqdm(doc_id_2_sent_ids):
                sent_ids = doc_id_2_sent_ids[doc_id]
                doc_dict = {'corpus_id': reddit_corpus_dict[doc_id]['id'],
                            'text': reddit_corpus_dict[doc_id]['text'],
                            'sentences': {},
                            'datetime': reddit_corpus_dict[doc_id]['datetime']}
                text = reddit_corpus_dict[doc_id]['text']
                event_idx = 0
                for sent_id in sent_ids:
                    sentence = get_line_from_corpus(processed_corpus, sent_id)
                    doc_dict['sentences'][str(sent_id)] = {}
                    doc_dict['sentences'][str(sent_id)]['text'] = sentence.strip()
                    doc_dict['sentences'][str(sent_id)]['events'] = []
                    match = re.search(re.escape(sentence.rstrip()), text)
                    if match is not None:
                        doc_dict['sentences'][str(sent_id)]['sentence_span'] = match.span()
                        if str(sent_id) in sent_id_2_vocab_freq:
                            doc_dict['sentences'][str(sent_id)]['events'].extend(
                                sent_id_2_vocab_freq[str(sent_id)])
                            doc_dict['sentences'][str(sent_id)]['events'] = list(
                                set(doc_dict['sentences'][str(sent_id)]['events']))
                    if len(doc_dict['sentences'][str(sent_id)]['events']) > 0:
                        for idx, event in enumerate(doc_dict['sentences'][str(sent_id)]['events']):
                            event_list = list(event)
                            event_list.insert(0, str(event_idx))
                            doc_dict['sentences'][str(sent_id)]['events'][idx] = tuple(event_list)
                            event_idx += 1
                doc_dict['text'] = text
                output_dict[doc_id] = doc_dict

def write_to_file(output_file):
    with open(output_file, 'w') as fp:
        json.dump(output_dict, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reddit_corpus", help="reddit corpus file")
    parser.add_argument("--processed_corpus", help="processed mfc corpus file")
    parser.add_argument("--po_tuple_features", help="po tuple features file")
    parser.add_argument("--doc_2_sent", help="doc id to sentence id map file")
    parser.add_argument("--output_file", help="output file with extracted events mapped to each article")
    args = parser.parse_args()
    print(vars(args))

    generate_event_2_posts_map(args.reddit_corpus, args.processed_corpus, args.po_tuple_features, args.doc_2_sent)
    write_to_file(args.output_file)