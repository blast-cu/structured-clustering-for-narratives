from tqdm import tqdm
import time
import os
import json

import concurrent
import logging

import ollama
from pydantic import BaseModel

"""
Utility functions for the initial exploration of the data with an LLM.
Includes functions for loading models, generating text, and formatting data.
"""


class OllamaClient:
    """
    A class to handle the Ollama client.
    """
    class Answer(BaseModel):
        characters: list[str]

    def __init__(self, host, port, model, seed, temperature, logger):
        self.logger = logger
        server_host = f"{host}:{port}"
        self.client = ollama.Client(server_host)

        self.model = model
        self.seed = seed
        self.temperature = temperature

        ollama.pull(model)

        self.options: ollama.Options = {
            "seed": seed,
            "temperature": temperature
        }

    def chat(self, messages):
        """
        Chat with the LLM and check the response.
        """
        self.logger.info(messages)
        response = self.client.chat(
            self.model,
            messages=messages,
            options=self.options,
            format=self.Answer.model_json_schema()
        )
        self.logger.info(response)  # debugging
        self.logger.info()

        try:  # https://github.com/ollama/ollama/releases/tag/v0.5.0
            response = self.Answer.model_validate_json(
                response.message.content
            )
            return response

        except Exception as e:
            self.logger.exception("Exception: " + str(e))
            self.logger.exception("Invalid response. Please try again.")
            return None


class Annotate:
    """
    Class for annotating text with LLM output.
    """

    def __init__(self, args, script_path, data_path, logger=None):
        # set up logging.
        if logger is None:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
        self.logger = logger

        logger.info("Setting Annotator variables and initializing Ollama client.")
        self.config = args

        # initialize the ollama client.
        self.ollama_client = OllamaClient(
            args.host,
            args.port,
            args.model,
            args.seed,
            args.temperature,
            logger
        )

        # set the output filename if not provided.
        if args.out_filename is None:
            data_file = args.dataset.split(".")[0]
            self.config.out_filename = f"{data_file}_processed.json"

        # set directories for reading/writing data.
        self.data_path = data_path
        self.results_path = os.path.join(data_path, "results")
        os.makedirs(self.results_path, exist_ok=True)

        # load the data.
        self.prompt_data, self.docs = \
            self.load_data(script_path, data_path)

        # check for existing results.
        self.already_processed, self.docs = \
            self.handle_processed()

    def load_data(self, script_path, data_path):

        # load prompt data.
        prompt_path = os.path.join(
            script_path, "prompts", self.config.prompt_file
        )
        try:
            prompt_data = json.load(open(prompt_path))
        except FileNotFoundError:
            self.logger.error(f"Prompt file {prompt_path} not found.")

        # load the data from json.
        dataset = json.load(open(os.path.join(data_path, self.config.dataset)))

        return prompt_data, dataset

    def handle_processed(self):
        """
        Load docs that have already been processed and 
        remove them from the list of docs to process.
        """
        # check for existing results.
        out_file_path = os.path.join(
            self.results_path, self.config.out_filename
        )
        already_processed = {}
        to_process = self.docs
        if os.path.exists(out_file_path):
            existing_data = json.load(open(out_file_path))
            already_processed = existing_data["data"] if "data" in existing_data else {}
            self.logger.info(f"Found {len(already_processed)} existing results.")

            # remove already processed docs from the list.
            to_process = {doc_id: doc for doc_id, doc in self.docs.items()
                          if doc_id not in already_processed.keys()}

        return already_processed, to_process

    def format_doc(self, doc_data: dict) -> dict:
        """
        Format the doc data into an entry for processing.
        returns
            - dict: {"text": str}
        """
        entry = {}
        article_sents = doc_data["sentences"]
        article_sents_text = [s["text"] for s in article_sents.values()]
        article_text = " ".join(article_sents_text).strip()
        entry["text"] = article_text + "\n"

        return entry

    def set_head_msgs(self):
        """
        Set the head messages for the conversations.
        """
        self.head_msgs = []
        # start with the system prompt.
        self.head_msgs.append({"role": "system", "content": self.prompt_data["system_prompt"]})

        # implement n shot prompting.
        shots = []
        if "demos" in self.prompt_data.keys() and len(self.prompt_data["demos"]) > 0:
            head_user_prompt = ""
            for demo_item in self.prompt_data["demos"]:
                user_turn = "user: " + self.prompt_data["question"] + "\n" + demo_item["text"]
                assissant_turn = "assistant: " + str(demo_item["answer"])
                shots.append(user_turn + "\n" + assissant_turn)

            head_user_prompt = "\n\n".join(shots)
            self.head_msgs.append({"role": "user", "content": head_user_prompt})
        else:
            self.head_msgs.append({"role": "user", "content": ""})

    def save_results(self, out_list: dict):
        """
        Save the results and config details to a json file.
        """
        final_output = {}
        final_output["config"] = vars(self.config)
        final_output["time_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        final_output["prompt_data"] = self.prompt_data
        final_output["data"] = out_list

        out_path = os.path.join(
            self.data_path, "results", self.config.out_filename
        )
        with open(out_path, "w") as f:
            json.dump(final_output, f, indent=4)

    def annotate(self, messages):
        """
        Annotate the messages with the LLM.
        """
        max_retries = self.config.max_retries
        retry_count = 0
        while retry_count < max_retries:
            try:
                annotation = self.ollama_client.chat(
                    messages
                )
                if annotation is not None:
                    return annotation
                else:  # retry if the response did not match the schema.
                    retry_count += 1

            except Exception as e:
                self.logger.exception("Exception: " + str(e))
                self.logger.exception("Ollama Error. Please try again.")
                retry_count += 1
        return None

    def process_docs(self):
        """
        Process the docs in parallel.
        """

        num_workers = self.config.workers
        save_interval = self.config.save_interval

        self.logger.info("Formatting docs")

        data = {}
        for doc_id, doc_data in tqdm(self.docs.items()):
            entry = self.format_doc(doc_data)
            data[doc_id] = entry

        # set the head messages for the conversations.
        self.set_head_msgs()

        annotated_docs = self.already_processed
        total_docs = len(data)

    #    # Process documents
    #     with tqdm(total=total_docs) as pbar:
    #         for doc_id, doc in data.items():
    #             processed_count = 0
    #             # Process docs and save at regular intervals
    #             try:
    
    #                 doc = self.process_doc(doc)
    #                 annotated_docs[doc_id] = doc
    #                 processed_count += 1

    #                 # Update the progress bar
    #                 pbar.update(1)

    #                 # Save annotated_docs at regular intervals
    #                 if processed_count % save_interval == 0:
    #                     self.save_results(annotated_docs)
    #                     self.logger.info(f"Progress saved after processing {processed_count} docs.")

    #             except Exception as e:
    #                 self.logger.exception(f"Error processing doc {doc_id}: {e}")

        # Use a ThreadPoolExecutor for parallel processing
        self.logger.info(f"Processing docs with {num_workers} workers.")  
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process_doc, doc): doc_id for doc_id, doc in data.items()}

            # Initialize tqdm progress bar to track doc processing
            with tqdm(total=total_docs) as pbar:
                processed_count = 0
                # Process docs and save at regular intervals
                for future in concurrent.futures.as_completed(futures):
                    doc_idx = futures[future]
                    try:
                        # Get the results of process_doc() for each doc
                        doc = future.result()
                        annotated_docs[doc_idx] = doc
                        processed_count += 1

                        # Update the progress bar
                        pbar.update(1)

                        # Save annotated_docs at regular intervals
                        if processed_count % save_interval == 0:
                            self.save_results(annotated_docs)
                            self.logger.info(f"Progress saved after processing {processed_count} docs.")

                    except Exception as e:
                        self.logger.exception(f"Error processing doc {doc_idx}: {e}")

        # Save the final results.
        self.save_results(annotated_docs)

    def process_doc(self, doc):
        """
        Process the doc with the LLM.
        """
        # Add the user prompt to the messages.
        messages = self.head_msgs
        user_prompt = self.prompt_data["question"] + '\n' + doc["text"]
        messages[-1]['content'] += user_prompt

        annotation = self.annotate(messages)
        if annotation is not None:  # make the output serializable.
            annotation = annotation.model_dump()

        doc["annotation"] = annotation
        return doc
