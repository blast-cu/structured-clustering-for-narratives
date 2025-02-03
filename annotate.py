import argparse
import json
import pickle
import concurrent.futures
from ollama_client import Ollama
from pyhocon import ConfigFactory
from tqdm import tqdm


class Annotator:
    def __init__(self, host, port, config):
        self.ollama_client = Ollama(host,
                                    port,
                                    'llama3.3',
                                    seed=42,
                                    temperature=0.1)
        self.config = config

    def process_documents(self, num_workers, save_interval):
        with open(self.config["event_chains_path"], 'rb') as f:
            data = pickle.load(f)

        annotated_docs = {}
        total_docs = len(data)

        # Use a ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process_document, doc_idx, doc): doc_idx for doc_idx, doc in data.items()}

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
                            self.save_progress(annotated_docs)
                            print(f"Progress saved after processing {processed_count} documents.", flush=True)

                    except Exception as e:
                        print(f"Error processing document {doc_idx}: {e}", flush=True)

        # Final save at the end, in case it wasn't saved during the last interval
        self.save_progress(annotated_docs)

    def process_document(self, doc_idx, doc):
        # Process each document's event chains
        for event_chain_idx, event_chain in doc['event_chains'].items():
            annotation = self.annotate(event_chain['chain_text'])
            doc['event_chains'][event_chain_idx]['annotation'] = annotation
        return doc

    def save_progress(self, annotated_docs):
        with open(self.config["annotated_event_chains_path"], 'wb') as f:
            pickle.dump(annotated_docs, f)

    def annotate(self, event_chain):
        system_prompt = """You are an annotator. Using character theory and affect control theory, you will perform the following annotation tasks on the following text. 1. Is there an immigrant character (or similar) role in the text? Or does the text talk about immigrants? If yes, are they framed as a hero, a victim, or a threat? If no, your answer should be neutral. 2. Is the stance of the text pro or anti immigration? If neither, your answer should be neutral. Your answer should be in JSON using the following format: {"role": "hero/victim/threat/neutral", "stance": "pro/anti/neutral", "reason": "your reasoning for your answers"}. Do not generate anything else."""

        user_prompt = f"Text: \"{event_chain}\""
        role_response = ["hero", "victim", "threat", "neutral"]
        stance_response = ["pro", "anti", "neutral"]

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.ollama_client.chat(system_prompt, user_prompt)
                try:
                    response = json.loads(response)
                    if 'role' in response and 'stance' in response:
                        if response['role'] in role_response and response['stance'] in stance_response:
                            return response
                        else:
                            print("Invalid response. Please try again.", flush=True)
                            raise Exception("Invalid response.")
                except Exception as e:
                    print("Exception: " + str(e), flush=True)
                    print("Invalid response. Please try again.", flush=True)
                    retry_count += 1
            except Exception as e:
                print("Exception: " + str(e), flush=True)
                print("Ollama Error. Please try again.", flush=True)
                retry_count += 1
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotator')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--host', metavar='HOST')
    parser.add_argument('--port', default=9999, metavar='PORT')
    parser.add_argument('--workers', type=int, default=4, metavar='WORKERS', help='Number of worker threads')
    parser.add_argument('--save_interval', type=int, default=10, metavar='SAVE_INTERVAL',
                        help='Number of documents to process before saving progress')
    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    annotator = Annotator(args.host, args.port, config)
    annotator.process_documents(args.workers, args.save_interval)
