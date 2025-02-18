import json
import os
import argparse
from dotenv import load_dotenv

from character.extract_characters.utils import \
    CharacterAnnotate

# load the environment variables.
load_dotenv()
DATA_PATH = os.environ.get("DATA_DIR")

"""
This script is used to generate the initial exploration of the data with an LLM of the user's choice.
Tested with:

"""


def main(args):

    if args.out_filename is None:
        data_file = args.dataset.split(".")[0]
        args.out_filename = f"{data_file}_processed.json"

    # ensure the results directory exists.
    os.makedirs(os.path.join(DATA_PATH, "results"), exist_ok=True)

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
    annotator.process_articles(args.workers, args.save_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the initial exploration of the data with an LLM of the user's choice."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of workers to use for parallel processing."
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="The interval at which to save the results."
    )
    parser.add_argument(
        '--host',
        metavar='HOST',
        help="The host for the Ollama API."
    )
    parser.add_argument(
        '--port',
        default=9999, metavar='PORT',
        help="The port for the Ollama API."
    )
    parser.add_argument(
        "--out_filename",
        type=str,
        help="The name of the output file to save the results in DATA_PATH/results."
    )
    parser.add_argument(
        "--model",
        default='llama3:70b-instruct-q4_0',
        type=str,
        help="ollama model id of the LLM to use for the initial exploration.",
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
    main(args)
