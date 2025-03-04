import argparse
import json
from collections import defaultdict
import spacy
from tqdm import tqdm


def main(input_file, save_path):
    nlp = spacy.load("en_core_web_lg")

    f = open(save_path + "corpus.txt", 'w+')
    doc_id_2_sent_id = defaultdict(list)

    # read input file
    with open(input_file, 'r') as file:
        input_data = json.load(file)

    sent_id = 0
    for doc_id, (key, values) in tqdm(enumerate(input_data.items())):
        if values['primary_frame'] is None:
            continue
        # Preprocess the text by splitting it into lines and removing empty lines
        lines = [line.strip() for line in values['text'].split('\n') if line.strip()]
        # Process each line using spaCy
        for line in lines:
            doc = nlp(line)
            for idx, sent in enumerate(doc.sents):
                doc_id_2_sent_id[doc_id].append(sent_id)
                sent_id += 1
                f.write(str(sent) + "\n")

    with open(save_path + "doc_id_2_sent_ids_immigrants_labeled.json", "w+") as outfile:
        json.dump(doc_id_2_sent_id, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", help="input corpus file")
    parser.add_argument("--save_path", help="output corpus with one sentence per line")
    args = parser.parse_args()
    print(vars(args))
    main(args.input_file, args.save_path)