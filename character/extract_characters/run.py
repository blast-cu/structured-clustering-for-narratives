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
    make_demo, format_article

from utils.ollama_client import Ollama

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


def save_results(args: dict, out_list: list):
    """
    Save the results to a json file.

    args:
        args (dict): the arguments passed to the script.
        out_list (list): the list of dictionaries containing the output.
    """
    # add config information and save results as json.
    final_output = {}
    final_output["config"] = vars(args)
    final_output["data"] = out_list
    final_output["time_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(DATA_PATH, "results", args.out_filename), "w") as f:
        json.dump(final_output, f, indent=4)


def main(args):

    # ensure the results directory exists.
    os.makedirs(os.path.join(DATA_PATH, "results"), exist_ok=True)

    # check for existing results.
    if os.path.exists(os.path.join(DATA_PATH, "results", args.out_filename)):
        existing_data = json.load(
            open(os.path.join(DATA_PATH, "results", args.out_filename))
        )
        old_out_list = existing_data["data"] if "data" in existing_data else []  # TODO: implement this.

    # load the model or setup the api.
    llm = Ollama(
        args.host, args.port,
        args.model, seed=args.seed,
        temperature=args.temperature
    )  # takes a while to load the model.

    # generate prompts.
    np.random.seed(args.seed)

    # load prompt data.
    prompt_path = os.path.join("character", "extract_characters", "prompts", args.prompt_file)
    # character/extract_characters/prompts/immigration_default.json
    prompt_data = json.load(open(prompt_path))

    # load the data from json.
    if not args.dataset.endswith(".json"):
        args.dataset = f"{args.dataset}.json"
    dataset = json.load(open(os.path.join(DATA_PATH, args.dataset)))

    logger.info("Formatting raw article data...")
    out_list = []
    for article_id, article_data in tqdm(dataset.items()):
        entry = format_article(article_id, article_data)
        out_list.append(entry)

    head_user_prompt = []
    for demo_item in prompt_data["demos"]:  # 4 shot demo.
        # each demo has user and assistant content. 
        user_turn = {}
        user_turn["role"] = "user"
        user_turn["content"] = prompt_data["question"] + "\n" + demo_item["article_text"]
        
        assissant_turn = {}
        assissant_turn["role"] = "assistant"
        assissant_turn["content"] = str(demo_item["answer"])

        head_user_prompt.append(user_turn)
        head_user_prompt.append(assissant_turn)


    logger.info("Generating outputs...")
    for idx, item in enumerate(tqdm(out_list)):
        messages = []
        messages.append({"role": "system", "content": prompt_data["system_prompt"]})
        messages.extend(head_user_prompt)
        user_prompt = prompt_data["question"] + '\n' + item["article_text"]
        messages.append({"role": "user", "content": user_prompt})
        options: ollama.Options = {
            "seed": args.seed,
            "temperature": args.temperature
        }

        response = llm.client.chat(
            llm.model,
            messages=messages,
            options=options
        )['message']['content']
        print(response)
        out_list[idx]["output"] = response
        exit()

        if idx % 100 == 0:
            save_results(args, out_list)

    logger.info(
        f"#Cases when prompts exceed max length: {llm.prompt_exceed_max_length}"
    )
    logger.info(f"#Cases when max new tokens < 50: {llm.fewer_than_50}")

    # save the results.
    save_results(args, out_list)


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
