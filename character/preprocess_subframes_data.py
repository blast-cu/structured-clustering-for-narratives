import json
import pickle
import os
import argparse
from dotenv import load_dotenv


# load the environment variables.
load_dotenv()
DATA_PATH = os.environ.get("DATA_DIR")
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

def main(args):

    in_filepath = os.path.join(DATA_PATH, "raw_data", args.in_file)
    with open(in_filepath, "rb") as in_file:
        [article2URL, article2dop, article2headline, article2text, \
         art2label, art2segment_ids, seg_id2text] = pickle.load(in_file)
    
    data = {}
    for art_id, headline in article2headline.items():
        text = headline + "\n " + article2text[art_id]
        art_id = art_id.replace("article#", "")

        data[art_id] = {
            "text": text,
        }

    out_file = args.out_file
    with open(os.path.join(DATA_PATH, out_file), "w") as f:
        json.dump(data, f, indent=4)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_file", type=str,
        help="The name of the picle file to preprocess.",
    )
    parser.add_argument(
        "--out_file", type=str,
        help="The name of the output file.",
    )
    args = parser.parse_args()
    main(args)