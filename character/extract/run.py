import json
import yaml
import os
import argparse
from dotenv import load_dotenv

from character.extract.utils import \
    Annotate


# load the environment variables.
load_dotenv()
DATA_PATH = os.environ.get("DATA_DIR")
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

"""
This script is used to generate annotations by an LLM 
for a given dataset using the Annotate class.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate a dataset using an LLM."
    )
    parser.add_argument(
        "--config", type=str, default="default.yaml",
        help="Name of config file in configs directory"
    )
    parser.add_argument(
        "--prompt_file", type=str,
        help="The name of the json file in prompts/ to be used.",
    )
    parser.add_argument(
        "--dataset", type=str,
        help="Name of dataset to analyze in the 'DATA_DIR/' directory.",
    )
    parser.add_argument(
        "--out_filename",
        type=str,
        help="Name of the output file to save to in DATA_DIR/results."
    )

    # LLM arguments for ollama.
    parser.add_argument(
        "--model", type=str,
        help="ollama model id of the LLM to use for the initial exploration.",
    )
    parser.add_argument(
        "--seed", type=int, help="The seed for the random number generator."
    )
    parser.add_argument(
        "--temperature", type=float, help="The temperature for sampling."
    )
    parser.add_argument(
        '--host', metavar='HOST', required=True,
        help="The host for the Ollama API."
    )
    parser.add_argument(
        '--port', metavar='PORT', required=True,
        help="The port for the Ollama API."
    )

    # Arguments for parallel processing and saving.
    parser.add_argument(
        "--workers", type=int,
        help="Number of workers to use for parallel processing."
    )
    parser.add_argument(
        "--save_interval", type=int,
        help="The interval at which to save the results."
    )
    parser.add_argument(
        "--max_retries", type=int, default=5,
        help="Max # of tries for LLM to gen correctly formatted response.",
    )

    args = parser.parse_args()
    if args.config is not None:  # load the config file containing defaults.
        config_file = os.path.join(SCRIPT_PATH, "configs", args.config)
        config = yaml.safe_load(open(config_file)) if args.config is not None else {}
        parser.set_defaults(**config)
        args = parser.parse_args()
    
    print(args)

    annotator = Annotate(
        args,
        SCRIPT_PATH,
        DATA_PATH
    )
    annotator.process_docs()
