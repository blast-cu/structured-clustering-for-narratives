from tqdm import tqdm
import time
import os
import json

import concurrent
import logging

from utils.ollama_client import Ollama
import ollama

from pydantic import BaseModel


class CharacterAnnotate:
    """
    Class to annotate the characters in the articles using the Ollama API.
    """

    class Characters(BaseModel):
        """
        Class to define desired llm output.
        """
        characters: list[str]

    def __init__(self, args, prompt_data, articles, data_path, logger=None):

        # set up logging.
        if logger is None:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
        self.logger = logger

        self.config = args

        # initialize the ollama client.
        self.ollama_client = Ollama(
            args.host,
            args.port,
            args.model,
            seed=args.seed,
            temperature=args.temperature
        )

        self.ollama_options: ollama.Options = {
            "seed": self.config.seed,
            "temperature": self.config.temperature
        }

        self.prompt_data = prompt_data
        self.articles = articles
        self.set_head_msgs()
        self.data_path = data_path

        logger.info("Initialized Ollama client.")

    def format_article(self, article_data: dict) -> dict:
        """
        Format text by joining the sentences in the article.

        args:
            row (dict): the row from the dataset.
        returns:
            entry (dict): the formatted conversation.
        """
        entry = {}

        article_sents = article_data["sentences"]
        article_sents_text = [s["text"] for s in article_sents.values()]
        article_text = " ".join(article_sents_text).strip()
        entry["article_text"] = article_text + "\n"
        return entry

    def set_head_msgs(self):
        """
        Set the head messages for every llm call.
        """
        # start with the system prompt.
        self.head_msgs = []
        self.head_msgs.append({"role": "system", "content": self.prompt_data["system_prompt"]})

        shots = []
        if len(self.prompt_data["demos"]) > 0:
            head_user_prompt = ""
            for demo_item in self.prompt_data["demos"]:
                # strings instead of list of turns.
                user_turn = "user: " + self.prompt_data["question"] + "\n" + demo_item["article_text"]
                assissant_turn = "assistant: " + str(demo_item["answer"])
                shots.append(user_turn + "\n" + assissant_turn)

            head_user_prompt = "\n".join(shots)   
            self.head_msgs.append({"role": "user", "content": head_user_prompt})
        else:
            self.head_msgs.append({"role": "user", "content": ""})

    def save_results(self, output: dict):
        """
        Save the results to a json file.

        args:
            output (dict): the output of the model.
        """
        # add config information and save results as json.
        final_output = {}
        final_output["config"] = vars(self.config)
        final_output["time_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        final_output["prompt_data"] = self.prompt_data
        final_output["data"] = output

        out_path = os.path.join(self.data_path, "results", self.config.out_filename)
        with open(out_path, "w") as f:
            json.dump(final_output, f, indent=4)
        # exit()  # temp exit for testing.

    def annotate(self, messages):
        """
        Annotate the characters in the article with llm, 
        check for invalid responses and retry if necessary.
        """

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.ollama_client.client.chat(
                    self.config.model,
                    messages=messages,
                    options=self.ollama_options,
                    format=self.Characters.model_json_schema()
                )

                try:  # https://github.com/ollama/ollama/releases/tag/v0.5.0
                    response = self.Characters.model_validate_json(
                        response.message.content
                    )
                    return response

                except Exception as e:
                    self.logger.exception("Exception: " + str(e))
                    self.logger.exception("Invalid response. Please try again.")
                    retry_count += 1
            except Exception as e:
                self.logger.exception("Exception: " + str(e))
                self.logger.exception("Ollama Error. Please try again.")
                retry_count += 1
        return None

    def process_articles(self, num_workers, save_interval):
        """
        Parallel processing of the articles in the dataset.
        """

        data = {}
        self.logger.info("Formatting articles.")
        for article_id, article_data in tqdm(self.articles.items()):
            entry = self.format_article(article_data)
            data[article_id] = entry

        annotated_docs = {}
        total_docs = len(data)

        # Use a ThreadPoolExecutor for parallel processing
        self.logger.info(f"Processing documents with {num_workers} workers.")  
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process_article, article_id, article): article_id for article_id, article in data.items()}

            # Initialize tqdm progress bar to track document processing
            with tqdm(total=total_docs) as pbar:
                processed_count = 0
                # Process documents and save at regular intervals
                for future in concurrent.futures.as_completed(futures):
                    doc_idx = futures[future]
                    try:
                        doc = future.result()
                        annotated_docs[doc_idx] = doc
                        processed_count += 1

                        # Update the progress bar
                        pbar.update(1)

                        # Save annotated_docs at regular intervals
                        if processed_count % save_interval == 0:
                            self.save_results(annotated_docs)
                            self.logger.info(f"Progress saved after processing {processed_count} documents.")

                    except Exception as e:
                        self.logger.exception(f"Error processing document {doc_idx}: {e}")
        self.save_results(annotated_docs)

    def process_article(self, article_idx, article):
        # process single article
        messages = self.head_msgs
        user_prompt = '\n' + self.prompt_data["question"] + '\n' + article["article_text"] + '\n'
        messages[-1]['content'] += user_prompt

        annotation = self.annotate(messages)
        if annotation is not None:
            annotation = annotation.characters

        article["characters"] = annotation

        return article
