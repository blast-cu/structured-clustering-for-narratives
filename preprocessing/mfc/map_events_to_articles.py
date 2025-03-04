import argparse
import json
import pickle as pk
import re

from tqdm import tqdm

from utils.lemmatize import lemmatize_sentence

output_dict = {}


def get_text_from_corpus(file, article_id):
    with open(file, 'r') as f:
        corpus_dict = json.load(f)
        id = 'Immigration1.0-' + article_id.split('-')[1]
        return corpus_dict[id]['text']


def get_line_from_corpus(file, line_number):
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i == line_number:
                return line


def generate_event_2_mfc_map(mfc_corpus, processed_corpus, po_tuple_features, doc_2_sent):
    print('inside generate_event_2_mfc_map')

    with open(po_tuple_features, 'rb') as svos_pkl_file:
        mydict = pk.load(svos_pkl_file)
        sent_id_2_vocab_freq = mydict['sent_id_2_vocab_freq']
        with open(doc_2_sent, 'rb') as json_file:
            doc_id_2_sent_ids = json.load(json_file)
            for doc_id in tqdm(doc_id_2_sent_ids):
                sent_ids = doc_id_2_sent_ids[doc_id]
                doc_dict = {'corpus_id': get_line_from_corpus(processed_corpus, sent_ids[0]).strip(),
                            'text': '',
                            'sentences': {}}
                text = get_text_from_corpus(mfc_corpus, doc_dict['corpus_id'])
                event_idx = 0
                for sent_id in sent_ids[1:]:
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


def add_mfc_frame(mfc_corpus, mfc_codes):
    print('inside add_mfc_frame')

    with open(mfc_corpus, 'r') as f:
        corpus_dict = json.load(f)
    with open(mfc_codes, 'r') as f:
        mfc_frames = json.load(f)
    for doc_id in tqdm(output_dict):
        id = 'Immigration1.0-' + output_dict[doc_id]['corpus_id'].split('-')[1]
        article = corpus_dict[id]
        output_dict[doc_id]['primary_frame'] = mfc_frames[str(article['primary_frame'])]
        for sentence_id in output_dict[doc_id]['sentences']:
            sentence_span = output_dict[doc_id]['sentences'][sentence_id]['sentence_span']
            output_dict[doc_id]['sentences'][sentence_id]['mfc_frame'] = {}
            annotator_idx = 0
            for annotator in article['annotations']['framing']:
                output_dict[doc_id]['sentences'][sentence_id]['mfc_frame'][annotator_idx] = []
                for annotation in article['annotations']['framing'][annotator]:
                    if annotation['start'] > sentence_span[1] or annotation['end'] < sentence_span[0]:
                        continue
                    else:
                        output_dict[doc_id]['sentences'][sentence_id]['mfc_frame'][annotator_idx].append(
                            mfc_frames[str(annotation['code'])])
                annotator_idx += 1


def add_mfc_frame_phrase(mfc_corpus, mfc_codes):
    print('inside add_mfc_frame_phrase')

    with open(mfc_corpus, 'r') as f:
        corpus_dict = json.load(f)
    with open(mfc_codes, 'r') as f:
        mfc_frames = json.load(f)
    for doc_id in tqdm(output_dict):
        id = 'Immigration1.0-' + output_dict[doc_id]['corpus_id'].split('-')[1]
        article = corpus_dict[id]
        output_dict[doc_id]['primary_frame'] = mfc_frames[str(article['primary_frame'])]
        text = output_dict[doc_id]['text']
        for sentence_id in output_dict[doc_id]['sentences']:
            sentence_span = output_dict[doc_id]['sentences'][sentence_id]['sentence_span']
            sentence_text = output_dict[doc_id]['sentences'][sentence_id]['text']
            if sentence_text == 'PRIMARY':
                continue
            events = output_dict[doc_id]['sentences'][sentence_id]['events']
            if len(events) == 0:
                continue
            output_dict[doc_id]['sentences'][sentence_id]['phrases_frame'] = []
            phrase_annotations = article['annotations']['framing']
            for annotator in phrase_annotations:
                annotations = phrase_annotations[annotator]
                for annotation in annotations:
                    if annotation['start'] > sentence_span[1] or annotation['end'] < sentence_span[0]:
                        continue
                    else:
                        phrase = text[annotation['start']:annotation['end']]
                        phrase_lemmatized = lemmatize_sentence(phrase)
                        for event in events:
                            if event[0][:-2] in phrase_lemmatized and event[1] in phrase_lemmatized:
                                event_str = event[0] + ' ' + event[1]
                                frame = mfc_frames[str(annotation['code'])]
                                tuple_entry = (event_str, frame)
                                output_dict[doc_id]['sentences'][sentence_id]['phrases_frame'].append(tuple_entry)


def write_to_file(output_file):
    with open(output_file, 'w') as fp:
        json.dump(output_dict, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mfc_corpus", help="mfc corpus file")
    parser.add_argument("--processed_corpus", help="processed mfc corpus file")
    parser.add_argument("--mfc_codes", help="mfc codes file")
    parser.add_argument("--po_tuple_features", help="po tuple features file")
    parser.add_argument("--doc_2_sent", help="doc id to sentence id map file")
    parser.add_argument("--output_file", help="output file with extracted events mapped to each article")
    args = parser.parse_args()
    print(vars(args))

    generate_event_2_mfc_map(args.mfc_corpus, args.processed_corpus, args.po_tuple_features, args.doc_2_sent)
    add_mfc_frame(args.mfc_corpus, args.mfc_codes)
    add_mfc_frame_phrase(args.mfc_corpus, args.mfc_codes)
    write_to_file(args.output_file)
