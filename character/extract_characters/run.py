import json
import yaml
from tqdm import tqdm
import numpy as np
import os
import time
import argparse
import logging
from dotenv import load_dotenv

from character.extract_characters.utils import \
    CharacterAnnotate



# set up logging.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# load the environment variables.
load_dotenv()
DATA_PATH = os.environ.get("DATA_DIR")

"""
This script is used to generate the initial exploration of the data with an LLM of the user's choice.
Tested with:

"""


def main(args):

    # ensure the results directory exists.
    os.makedirs(os.path.join(DATA_PATH, "results"), exist_ok=True)

    # # check for existing results.
    # if os.path.exists(os.path.join(DATA_PATH, "results", args.out_filename)):
    #     existing_data = json.load(
    #         open(os.path.join(DATA_PATH, "results", args.out_filename))
    #     )
    #     old_out_list = existing_data["data"] if "data" in existing_data else []  # TODO: implement this.

    # load prompt data.
    prompt_path = os.path.join("character", "extract_characters", "prompts", args.prompt_file)
    prompt_data = json.load(open(prompt_path))

    # load the data from json.
    if not args.dataset.endswith(".json"):
        args.dataset = f"{args.dataset}.json"
    dataset = json.load(open(os.path.join(DATA_PATH, args.dataset)))

    annotator = CharacterAnnotate(
        args,
        prompt_data,
        dataset,
        DATA_PATH
    )
    annotator.process_articles(args.num_workers, args.save_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the initial exploration of the data with an LLM of the user's choice."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Name of config file in prompt/configs directory"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        type=str,
        help="The host for the Ollama API."
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="The port for the Ollama API."
    )
    parser.add_argument(
        "--out_filename",
        default="initial_exploration.json",
        type=str,
        help="The name of the output file to save the results in DATA_PATH/results."
    )
    parser.add_argument(
        "--model",
        default='llama3:70b-instruct-q4_0',
        type=str,
        help="HF model id of the LLM to use for the initial exploration.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32768,
        help="The maximum prompt length for the model.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The seed for the random number generator."
    )
    parser.add_argument(
        "--prompt_file",
        default="immigration_default.json",
        type=str,
        help="The name of the json file in prompt/prompts containing the prompt template.",
    )
    parser.add_argument(
        "--dataset",
        default="immigration_processed_corpus.json",
        type=str,
        help="Name of dataset to analyze in the 'data/' directory.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="The temperature for sampling."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="The top_p for sampling."
    )

    args = parser.parse_args()

    # load the config file containing defaults.
    if args.config is not None:
        config_file = os.path.join("reader", "configs", args.config)
        config = yaml.safe_load(open(config_file)) if args.config is not None else {}
        parser.set_defaults(**config)
    main(args)
