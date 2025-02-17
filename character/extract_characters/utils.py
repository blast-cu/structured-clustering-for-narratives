from tqdm import tqdm
import time
import os
import json

import concurrent
import logging

from utils.ollama_client import Ollama
import ollama

"""
Utility functions for the initial exploration of the data with an LLM.
Includes functions for loading models, generating text, and formatting data.
"""


class CharacterAnnotate:

    def __init__(self, args, prompt_data, articles, data_path):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.config = args
        self.model = 'llama3.3'
        # print(f"Config: {vars(self.config)}")
        self.ollama_client = Ollama(
            self.config.host,
            self.config.port,
            # self.config.model,
            # 'llama3:70b-instruct-q4_0',
            self.model,
            seed=self.config.seed,
            temperature=self.config.temperature
        )
        self.prompt_data = prompt_data
        self.articles = articles
        self.get_head_msgs()
        self.data_path = data_path
        print("Initialized Ollama client.")

    def format_article(self, article_data: dict) -> dict:
        """
        format the conversation to include roles and newlines.

        args:
            row (dict): the row from the dataset.
        returns:
            dict: the formatted conversation with id, language and conversation 
                    string with roles and newlines.
        """
        entry = {}

        article_sents = article_data["sentences"]
        article_sents_text = [s["text"] for s in article_sents.values()]
        article_text = " ".join(article_sents_text).strip()
        entry["article_text"] = article_text + "\n"
        return entry

    def get_head_msgs(self):

        # start with the system prompt.
        self.head_msgs = []
        self.head_msgs.append({"role": "system", "content": self.prompt_data["system_prompt"]})

        # implement n shot prompting. 
        if len(self.prompt_data["demos"]) > 0:
            head_user_prompt = []
            for demo_item in self.prompt_data["demos"]:  # 4 shot demo.
                # each demo has user and assistant content. 
                user_turn = {}
                user_turn["role"] = "user"
                user_turn["content"] = self.prompt_data["question"] + "\n" + demo_item["article_text"]

                assissant_turn = {}
                assissant_turn["role"] = "assistant"
                assissant_turn["content"] = str(demo_item["answer"])

                head_user_prompt.append(user_turn)
                head_user_prompt.append(assissant_turn)
            self.head_msgs.extend(head_user_prompt)

    def save_results(self, out_list: list):
        """
        Save the results to a json file.

        args:
            args (dict): the arguments passed to the script.
            out_list (list): the list of dictionaries containing the output.
        """
        # add config information and save results as json.
        final_output = {}
        final_output["config"] = vars(self.config)
        final_output["data"] = out_list
        final_output["time_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")

        out_path = os.path.join(self.data_path, "results", self.config.out_filename)
        with open(out_path, "w") as f:
            json.dump(final_output, f, indent=4)
        exit()  # temp exit for testing.

    def annotate(self, messages):

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                options: ollama.Options = {
                    "seed": self.config.seed,
                    "temperature": self.config.temperature
                }

                response = self.ollama_client.client.chat(
                    self.model,
                    messages=messages,
                    options=options
                )['message']['content']
                self.logger.info(f"Response: {response}")

                try:
                    response = json.loads(response)
                    if 'character' in response:
                        if type(response['character']) is list:
                            return response
                        else:
                            self.logger.exception("Invalid response. Please try again.")
                            raise Exception("Invalid response.")
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

        data = {}
        for article_id, article_data in tqdm(self.articles.items()):
            entry = self.format_article(article_data)
            data[article_id] = entry

        annotated_docs = {}
        total_docs = len(data)

        # Use a ThreadPoolExecutor for parallel processing
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
        # process article
        messages = self.head_msgs
        user_prompt = self.prompt_data["question"] + '\n' + article["article_text"]
        messages.append({"role": "user", "content": user_prompt})

        annotation = self.annotate(messages)
        article["characters"] = annotation["character"]
        return article