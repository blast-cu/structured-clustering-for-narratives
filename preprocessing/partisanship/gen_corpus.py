import argparse
import json
import re
from collections import defaultdict

import nltk
import spacy
from nltk import sent_tokenize
from tqdm import tqdm


def main(input_file, save_path):
    nlp = spacy.load("en_core_web_lg")
    output_data = []

    f = open(save_path + "corpus.txt", 'w+')
    doc_id_2_sent_id = defaultdict(list)

    # read input file
    with open(input_file, 'r') as file:
        input_data = json.load(file)

    sent_id = 0
    for key, values in tqdm(input_data.items()):
        if values['label'] is None:
            continue
        # Preprocess the text by splitting it into lines and removing empty lines
        # lines = [line.strip() for line in values['text'].split('\n') if line.strip()]
        lines = split_article_into_sentences(values['text'])
        pass
        # Process each line using spaCy
        for line in lines:
            doc = nlp(line)
            for idx, sent in enumerate(doc.sents):
                doc_id_2_sent_id[key].append(sent_id)
                sent_id += 1
                f.write(str(sent) + "\n")

    with open(save_path + "doc_id_2_sent_ids_corpus_labeled.json", "w+") as outfile:
        json.dump(doc_id_2_sent_id, outfile)


def split_article_into_sentences(article_text):
    # Download the punkt tokenizer model if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Create a custom tokenizer and train it with common abbreviations
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Add common abbreviations to the tokenizer's abbreviation set
    common_abbreviations = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Inc.', 'Ltd.',
                            'Co.', 'Corp.', 'U.S.', 'U.K.', 'E.U.', 'e.g.', 'i.e.',
                            'etc.', 'vs.', 'Jan.', 'Feb.', 'Mar.', 'Apr.', 'Aug.',
                            'Sept.', 'Oct.', 'Nov.', 'Dec.', 'a.m.', 'p.m.', 'St.',
                            'Ave.', 'Blvd.', 'Rd.', 'Gov.', 'Sen.', 'Rev.', 'Ph.D.',
                            'M.D.', 'B.A.', 'M.A.', 'M.B.A.', 'Jr.', 'Sr.', 'No.']

    for abbr in common_abbreviations:
        tokenizer._params.abbrev_types.add(abbr.lower())

    # Pre-process the text to handle some common issues in news articles
    # Remove extra whitespace
    article_text = re.sub(r'\s+', ' ', article_text).strip()

    # Handle common quote patterns that might confuse the tokenizer
    article_text = re.sub(r'([""])(.+?)([""])', r'"\2"', article_text)

    # Replace ellipses with a placeholder to prevent incorrect sentence splitting
    article_text = re.sub(r'\.{3,}', ' ELLIPSIS_PLACEHOLDER ', article_text)

    # Replace periods after numbers with a placeholder
    article_text = re.sub(r'(\d+)\.(?!\d)', r'\1 NUM_PERIOD_PLACEHOLDER', article_text)

    # Use the custom trained tokenizer
    sentences = tokenizer.tokenize(article_text)

    # Post-processing to clean up sentences and restore placeholders
    cleaned_sentences = []
    for sentence in sentences:
        # Remove leading/trailing whitespace
        sentence = sentence.strip()

        # Restore ellipses
        sentence = sentence.replace('ELLIPSIS_PLACEHOLDER', '...')

        # Restore periods after numbers
        sentence = re.sub(r'(\d+) NUM_PERIOD_PLACEHOLDER', r'\1.', sentence)

        # Skip empty sentences
        if sentence:
            cleaned_sentences.append(sentence)

    return cleaned_sentences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", help="input corpus file")
    parser.add_argument("--save_path", help="output corpus with one sentence per line")
    args = parser.parse_args()
    print(vars(args))
    main(args.input_file, args.save_path)