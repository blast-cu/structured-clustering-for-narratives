from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch
import logging
import time

"""
Utility functions for the initial exploration of the data with an LLM.
Includes functions for loading models, generating text, and formatting data.
"""


def format_article(article_id: str, article_data: dict) -> dict:
    """
    format the conversation to include roles and newlines.

    args:
        row (dict): the row from the dataset.
    returns:
        dict: the formatted conversation with id, language and conversation 
                string with roles and newlines.
    """
    entry = {}
    entry["id"] = article_id

    article_sents = article_data["sentences"]
    article_sents_text = [s["text"] for s in article_sents.values()]
    article_text = " ".join(article_sents_text).strip()
    entry["article_text"] = article_text + "\n"
    return entry


def make_demo(item: dict, prompt: dict, demo: bool) -> str:
    """
    for demo prompt -
    - {INST}: the instruction.
    - {D}: the documents.
    - {Q}: the question.
    - {A}: the answers.
    ndoc: number of documents to put in context.
    use_shorter: None, "summary", or "extraction".
    """

    # replace placeholders in the prompt with actual values.
    prompt_str = (
        prompt["demo_prompt"]
        .replace("{INST}", prompt["instruction"])
        .replace("{Q}", prompt["question"])
    )
    prompt_str = prompt_str.replace("{T}", item["article_text"])

    if demo:  # if it's a demo example (for n-shot prompting).
        # join answers if they are in a list, otherwise use the answer directly.
        answer = (
            "\n" + "\n".join(item["answer"])
            if isinstance(item["answer"], list)
            else item["answer"]
        )
        # append the answer to the prompt.
        prompt_str = prompt_str.replace("{A}", "").rstrip() + answer
    else:
        # remove any space or newline characters if not a demo.
        prompt_str = prompt_str.replace("{A}", "").rstrip()

    return prompt_str


def get_max_memory() -> dict:
    """
    get the maximum memory available for loading models for the current gpus.

    returns:
        dict: a dictionary where the keys are gpu indices and the values are
                the maximum memory available for each gpu.
    """
    # get the free memory in gb.
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)

    # reserve 6gb for system processes and other overhead.
    max_memory = f"{free_in_GB - 6}GB"

    # get the number of gpus available.
    n_gpus = torch.cuda.device_count()

    # create a dictionary with the maximum memory for each gpu.
    max_memory = {i: max_memory for i in range(n_gpus)}

    return max_memory