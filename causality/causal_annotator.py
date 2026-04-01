import argparse
import os
import pickle
import re
import shutil
import concurrent.futures
import threading

from pyhocon import ConfigFactory
from tqdm import tqdm

from schemas import CausalPrediction
from utils.ollama_client import Ollama


class CausalAnnotator:
    def __init__(self, host, port, config, domain):
        self.config = config
        self.domain = domain

        self.save_lock = threading.Lock()

        self.reasoning_model = Ollama(host,
                                      port,
                                      config['reasoning_model'],
                                      config['seed'],
                                      config['temperature'])

        self.output_model = Ollama(host,
                                   port,
                                   config['output_model'],
                                   config['seed'],
                                   config['temperature'])

        self.num_ctx = config['num_ctx']

        with open("./causality/prompts/system_prompt.md", 'r', encoding='utf-8') as file:
            self.reasoning_system_prompt = file.read()
        with open("./causality/prompts/structured_output.md", 'r', encoding='utf-8') as file:
            self.structured_output_system_prompt = file.read()

    def process_documents(self, num_workers, save_interval, sequential=False):
        with open(self.config["event_pairs_path"], 'rb') as f:
            dataset = pickle.load(f)

        processed_docs = self.load_existing_progress()
        if len(processed_docs) > 0:
            annotated_docs = processed_docs
        else:
            annotated_docs = {}

        total_docs = len(dataset)
        existing_docs = len(annotated_docs)
        docs_to_process = total_docs - existing_docs

        total_pairs = sum(len(doc['event_pairs']) for doc in dataset.values())
        existing_pairs = sum(len(doc['event_pairs']) for doc in annotated_docs.values())

        print(f"Total documents: {total_docs}", flush=True)
        print(f"Total event pairs: {total_pairs}", flush=True)
        print(f"Existing processed documents: {existing_docs}", flush=True)
        print(f"Existing processed pairs: {existing_pairs}", flush=True)
        print(f"Documents to process: {docs_to_process}", flush=True)

        if sequential:
            with tqdm(total=docs_to_process, desc="Processing documents") as pbar:
                processed_count = 0
                for doc_id, doc in dataset.items():
                    if doc_id in annotated_docs:
                        continue
                    try:
                        processed_doc = self.process_document(doc)
                        annotated_docs[doc_id] = processed_doc
                        processed_count += 1
                        pbar.update(1)

                        if processed_count % save_interval == 0:
                            self.save_progress(annotated_docs)
                            print(f"Progress saved after processing {processed_count} documents.", flush=True)

                    except Exception as e:
                        print(f"Error processing document {doc_id}: {e}", flush=True)
                        pbar.update(1)
        else:
            unprocessed_items = {doc_id: doc for doc_id, doc in dataset.items()
                                 if doc_id not in annotated_docs}

            if not unprocessed_items:
                print("All documents already processed!", flush=True)
                return

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self.process_document, doc): doc_id
                           for doc_id, doc in unprocessed_items.items()}

                with tqdm(total=len(unprocessed_items), desc="Processing documents") as pbar:
                    processed_count = 0

                    for future in concurrent.futures.as_completed(futures):
                        doc_id = futures[future]
                        try:
                            doc = future.result()
                            annotated_docs[doc_id] = doc
                            processed_count += 1
                            pbar.update(1)

                            if processed_count % save_interval == 0:
                                self.save_progress(annotated_docs)
                                print(f"Progress saved after processing {processed_count} documents.", flush=True)

                        except Exception as e:
                            print(f"Error processing document {doc_id}: {e}", flush=True)
                            pbar.update(1)

        self.save_progress(annotated_docs)
        print(f"Processing complete. Final save completed.", flush=True)

    def process_document(self, doc):
        article = doc['text']

        for pair in doc['event_pairs']:
            e1 = pair['event_1']
            e2 = pair['event_2']

            prediction = self.annotate(
                event_1=(e1['verb'], e1['object']),
                sentence_1=e1['sentence_text'],
                event_2=(e2['verb'], e2['object']),
                sentence_2=e2['sentence_text'],
                article=article
            )
            pair['prediction'] = prediction

        return doc

    def annotate(self, event_1, sentence_1, event_2, sentence_2, article):
        reasoning_user_prompt = (
            f"DOMAIN: \"{self.domain}\"\n"
            f"EVENT_1: ({event_1[0]}, {event_1[1]})\n"
            f"SENTENCE_1: \"{sentence_1}\"\n"
            f"EVENT_2: ({event_2[0]}, {event_2[1]})\n"
            f"SENTENCE_2: \"{sentence_2}\"\n"
            f"ARTICLE: \"{article}\""
        )

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                reasoning_model_response = self.reasoning_model.chat(
                    self.reasoning_system_prompt,
                    reasoning_user_prompt,
                    think=True,
                    num_ctx=self.num_ctx)

                structured_response = self.output_model.chat(
                    self.structured_output_system_prompt,
                    reasoning_model_response,
                    think=False,
                    repeat_penalty=True,
                    format=CausalPrediction.model_json_schema(),
                    num_ctx=self.num_ctx)

                try:
                    response = CausalPrediction.model_validate_json(structured_response)
                    return response.model_dump()
                except Exception as e:
                    print(f"Exception: {e}", flush=True)
                    print("Invalid response. Retrying.", flush=True)
                    retry_count += 1
            except Exception as e:
                print(f"Exception: {e}", flush=True)
                print("Ollama Error. Retrying.", flush=True)
                retry_count += 1
        return None

    def save_progress(self, annotated_docs):
        """Thread-safe atomic save using temporary file and rename"""
        with self.save_lock:
            final_path = self.config["causal_annotations_path"]
            temp_path = final_path + ".tmp"

            try:
                with open(temp_path, 'wb') as f:
                    pickle.dump(annotated_docs, f)
                    f.flush()
                    os.fsync(f.fileno())

                shutil.move(temp_path, final_path)

            except Exception as e:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                print(f"Error during save: {e}", flush=True)
                raise

    def load_existing_progress(self):
        """Load existing processed documents if save file exists, creating a backup first"""
        save_path = self.config["causal_annotations_path"]

        if os.path.exists(save_path):
            print(f"Found existing save file: {save_path}")
            try:
                existing_data = self.create_backup_and_load(save_path)
                print(f"Loaded {len(existing_data)} existing documents")
                return existing_data
            except Exception as e:
                print(f"Error loading existing save file: {e}")
                print("Starting fresh...")
                return {}
        else:
            print("No existing save file found. Starting fresh...")
            return {}

    @staticmethod
    def get_next_backup_number(base_path):
        """Find the next highest backup number for the given file path"""
        directory = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        extension = os.path.splitext(base_path)[1]

        backup_numbers = []

        if os.path.exists(directory):
            for filename in os.listdir(directory):
                backup_pattern = f"{base_name}_backup_(\\d+){re.escape(extension)}"
                match = re.match(backup_pattern, filename)
                if match:
                    backup_numbers.append(int(match.group(1)))

        if backup_numbers:
            return max(backup_numbers) + 1
        else:
            return 1

    @staticmethod
    def create_backup_and_load(save_path):
        """Create a backup of existing save file and load the data from the original file"""
        backup_number = CausalAnnotator.get_next_backup_number(save_path)

        directory = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        extension = os.path.splitext(save_path)[1]

        backup_name = f"{base_name}_backup_{backup_number}{extension}"
        backup_path = os.path.join(directory, backup_name)

        shutil.copy2(save_path, backup_path)
        print(f"Created backup: {backup_path}")

        try:
            with open(save_path, 'rb') as f:
                existing_data = pickle.load(f)
            print(f"Successfully loaded data from original file: {save_path}")
            return existing_data
        except EOFError as e:
            print(f"EOFError - pickle file appears truncated: {e}")
            print("Starting with empty data")
            return {}
        except Exception as e:
            print(f"Unexpected error loading pickle file: {e}")
            print("Starting with empty data")
            return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Silver Label Generator for Event Causality')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--host', metavar='HOST')
    parser.add_argument('--port', default=9999, metavar='PORT')
    parser.add_argument('--workers', type=int, default=4, metavar='WORKERS', help='Number of worker threads')
    parser.add_argument('--domain', metavar='DOMAIN')
    parser.add_argument('--save_interval', type=int, default=10, metavar='SAVE_INTERVAL',
                        help='Number of documents to process before saving progress')
    parser.add_argument('--sequential', action='store_true', help='Run in sequential mode instead of parallel')
    args = parser.parse_args()
    print(vars(args))
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    generator = CausalAnnotator(args.host, args.port, config, args.domain)
    generator.process_documents(args.workers, args.save_interval, args.sequential)
