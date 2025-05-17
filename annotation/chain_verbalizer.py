import argparse
import json
import pickle
import re
import concurrent.futures

from pyhocon import ConfigFactory
from tqdm import tqdm

from schemas import EventChainSentence
from utils.ollama_client import Ollama


class ChainVerbalizer:
    def __init__(self, host, port, config, domain, model='llama3.3'):
        self.config = config

        self.reasoning_model = Ollama(host,
                                      port,
                                      model,
                                      seed=42,
                                      temperature=0.1)

        self.output_model = Ollama(host,
                                   port,
                                   'gemma3:4b',
                                   seed=42,
                                   temperature=0.1)

        with open("./annotation/prompts/chain_verbalization/system_prompt.md", 'r', encoding='utf-8') as file:
            self.reasoning_system_prompt = file.read()
        with open("./annotation/prompts/chain_verbalization/structured_output.md", 'r', encoding='utf-8') as file:
            self.structured_output_system_prompt = file.read()

        self.domain = domain
        self.immigration_char_group = '{Immigrants, Refugees, Asylum Seekers, Workers, Politicians, Law Enforcement, ' \
                                      'Judiciary, Government, Immigration Advocates}'
        self.guncontrol_char_group = '{Politicans, Gun Control Advocates, Gun Right Advocates, Law Enforcement, Judiciary, Government, Gun Crime Victims}'



    def process_documents(self, num_workers, save_interval, sequential=False):
        with open(self.config["annotated_event_chains_path"], 'rb') as f:
            data = pickle.load(f)

        annotated_docs = {}
        total_docs = len(data)

        if sequential:
            # Sequential processing
            with tqdm(total=total_docs) as pbar:
                processed_count = 0
                for doc_idx, doc in data.items():
                    try:
                        processed_doc = self.process_document(doc)
                        annotated_docs[doc_idx] = processed_doc
                        processed_count += 1
                        
                        # Update the progress bar
                        pbar.update(1)
                        
                        # # Save annotated_docs at regular intervals
                        if processed_count % save_interval == 0:
                            self.save_progress(annotated_docs)
                            print(f"Progress saved after processing {processed_count} documents.", flush=True)
                            
                    except Exception as e:
                        print(f"Error processing document {doc_idx}: {e}", flush=True)
        else:
            # Parallel processing using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self.process_document, doc): doc_idx for doc_idx, doc in data.items()}

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

    def process_document(self, doc):
        # Process each document's event chains
        for event_chain_idx, event_chain in doc['event_chains'].items():
            if 'CAUSAL' in event_chain['event_chain']:
                chain = event_chain['event_chain']
                article = doc['text']
                if self.domain == 'Gun Control':
                    char_groups = self.guncontrol_char_group
                elif self.domain == 'Immigration':
                    char_groups = self.immigration_char_group
                annotation = self.annotate(self.domain, chain, char_groups, article)
            # doc['event_chains'][event_chain_idx]['annotation'] = annotation
        return doc

    def save_progress(self, annotated_docs):
        with open(self.config["annotated_event_chains_path"], 'wb') as f:
            pickle.dump(annotated_docs, f)

    def annotate(self, domain, event_chain, char_groups, article):
        reasoning_user_prompt = f"DOMAIN: {domain} EVENT CHAIN: {event_chain} CHARACTER GROUPS: {char_groups} ARTICLE: {article}"

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                reasoning_model_response = self.reasoning_model.chat(self.reasoning_system_prompt,
                                                                     reasoning_user_prompt)

                _, json_content = self.extract_thinking_response(reasoning_model_response)

                structured_response = self.output_model.chat(self.structured_output_system_prompt,
                                                             json_content,
                                                             format=EventChainSentence.model_json_schema())
                try:
                    response = EventChainSentence.model_validate_json(structured_response)
                    pass
                except Exception as e:
                    print("Exception: " + str(e), flush=True)
                    print("Invalid response. Please try again.", flush=True)
                    retry_count += 1
            except Exception as e:
                print("Exception: " + str(e), flush=True)
                print("Ollama Error. Please try again.", flush=True)
                retry_count += 1
        return None

    @staticmethod
    def extract_thinking_response(response):
        # Extract content within <think> tags
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, response, re.DOTALL)
        think_content = think_match.group(1) if think_match else None

        # Remove the entire <think> section from the text to prevent extracting JSON from within it
        if think_match:
            full_think_section = think_match.group(0)  # This includes the <think> tags
            response_without_think = response.replace(full_think_section, '')
        else:
            response_without_think = response

        # Extract JSON string from the remaining text
        # json_pattern = r'(\{.*?\}|\[.*?\])'
        # json_match = re.search(json_pattern, response_without_think, re.DOTALL)
        # json_content = json_match.group(0) if json_match else None
        #
        return think_content, response_without_think


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chain Verbalizer')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--host', metavar='HOST')
    parser.add_argument('--port', default=9999, metavar='PORT')
    parser.add_argument('--workers', type=int, default=4, metavar='WORKERS', help='Number of worker threads')
    parser.add_argument('--save_interval', type=int, default=10, metavar='SAVE_INTERVAL',
                        help='Number of documents to process before saving progress')
    parser.add_argument('--sequential', action='store_true', help='Run in sequential mode instead of parallel')
    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    annotator = ChainVerbalizer(args.host, args.port, config, 
                                domain='Immigration',
                                model='deepseek-r1:32b-qwen-distill-q4_K_M')
    annotator.process_documents(args.workers, args.save_interval, args.sequential)