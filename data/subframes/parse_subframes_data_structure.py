import argparse
import pickle
import json

from tqdm import tqdm

def main(input_file, save_path):
    with open(input_file, "rb+") as f:
        data = pickle.load(f)

    corpus = {}

    headlines = data[2]
    article_text = data[3]
    labels = data[4]

    for k,v in tqdm(headlines.items()):
        corpus[k] = {
            "headline": v,
            "text": article_text[k],
            "label": labels[k]
        }

    with open(save_path + "corpus_labeled.json", "w+") as f:
        json.dump(corpus, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", help="input corpus file")
    parser.add_argument("--save_path", help="output corpus with one sentence per line")
    args = parser.parse_args()
    print(vars(args))
    main(args.input_file, args.save_path)