import argparse
import os
import gc
import pickle
import re
import shutil
import concurrent.futures

from pyhocon import ConfigFactory
from tqdm import tqdm

from schemas import EventChainSentence
from utils.ollama_client import Ollama


class ChainVerbalizer:
    def __init__(self, host, port, config, domain):
        self.config = config

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

        with open("./annotation/prompts/chain_verbalization/system_prompt.md", 'r', encoding='utf-8') as file:
            self.reasoning_system_prompt = file.read()
        with open("./annotation/prompts/chain_verbalization/structured_output.md", 'r', encoding='utf-8') as file:
            self.structured_output_system_prompt = file.read()

        self.domain = domain
        self.immigration_char_group = '{Immigrants, Refugees, Asylum Seekers, Workers, Politicians, Law Enforcement, ' \
                                      'Judiciary, Government, Immigration Advocates}'
        self.guncontrol_char_group = ('{Politicans, Gun Control Advocates, Gun Rights Advocates, Law Enforcement, '
                                      'Judiciary, Government, Gun Crime Victims}')



    def process_documents(self, num_workers, save_interval, sequential=False):
        with open(self.config["relations_path"], 'rb') as f:
            data = pickle.load(f)

        processed_docs = self.load_existing_progress()
        if len(processed_docs) > 0:
            annotated_docs = processed_docs
        else:
            annotated_docs = {}
        total_docs = len(data)

        print(f"Total documents to process: {total_docs - len(annotated_docs)}", flush=True)

        if sequential:
            # Sequential processing
            with tqdm(total=total_docs) as pbar:
                processed_count = 0
                for doc_idx, doc in data.items():
                    if doc_idx in annotated_docs:
                        # Skip already processed documents
                        processed_count += 1
                        pbar.update(1)
                        continue
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
                        if doc_idx in annotated_docs:
                            processed_count += 1
                            pbar.update(1)
                            continue
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
        article = doc['text']
        if self.domain == 'guncontrol':
            char_groups = self.guncontrol_char_group
        elif self.domain == 'immigration':
            char_groups = self.immigration_char_group

        # Build event map from the document
        event_map = {}
        for sentence in doc['sentences']:
            for event in doc['sentences'][sentence]['events']:
                event_map[int(event[0])] = (event[1].split("_")[0], event[2])

        # Initialize the event chains dictionary
        event_chains = {}
        event_chain_idx = 0
        if 'relations' in doc.keys():
            for key in doc['relations']:
                event_chain = {}
                if doc['relations'][key][1] == 'causal':
                    event_1 = event_map[key[0]]
                    relation = doc['relations'][key][1].upper()
                    event_2 = event_map[key[1]]
                    event_chain[
                        'event_chain'] = f"(({event_1[0]}, {event_1[1]}), {relation}, ({event_2[0]}, {event_2[1]}))"
                    event_chain['chain_text'] = self.annotate(self.domain, event_chain['event_chain'], char_groups, article)

                if event_chain:
                    event_chains[event_chain_idx] = event_chain
                    event_chain_idx += 1

        doc['event_chains'] = event_chains

        del article, event_map, event_chains
        gc.collect()

        return doc

    def annotate(self, domain, event_chain, char_groups, article):
        reasoning_user_prompt = f"DOMAIN: {domain} EVENT CHAIN: {event_chain} CHARACTER GROUPS: {char_groups} ARTICLE: {article}"

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                reasoning_model_response = self.reasoning_model.chat(self.reasoning_system_prompt,
                                                                     reasoning_user_prompt,
                                                                     num_ctx=self.num_ctx)

                _, json_content = self.extract_thinking_response(reasoning_model_response)

                structured_response = self.output_model.chat(self.structured_output_system_prompt,
                                                             json_content,
                                                             format=EventChainSentence.model_json_schema(),
                                                             num_ctx=self.num_ctx)
                try:
                    response = EventChainSentence.model_validate_json(structured_response)
                    response = response.model_dump()
                    return response['sentence']
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
        return think_content, response_without_think
    
    def save_progress(self, annotated_docs):
        with open(self.config["event_chains_path"], 'wb') as f:
            pickle.dump(annotated_docs, f)
    
    def load_existing_progress(self):
        """Load existing processed documents if save file exists, creating a backup first"""
        save_path = self.config["event_chains_path"]

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
        """Find the next available backup number for the given file path"""
        directory = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        extension = os.path.splitext(base_path)[1]
        
        backup_number = 1
        while True:
            backup_name = f"{base_name}_backup_{backup_number}{extension}"
            backup_path = os.path.join(directory, backup_name)
            if not os.path.exists(backup_path):
                return backup_number
            backup_number += 1

    @staticmethod
    def create_backup_and_load(self, save_path):
        """Create a backup of existing save file and load the data from the original file"""
        backup_number = get_next_backup_number(save_path)
        
        directory = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        extension = os.path.splitext(save_path)[1]
        
        backup_name = f"{base_name}_backup_{backup_number}{extension}"
        backup_path = os.path.join(directory, backup_name)
        
        # Create backup by copying the existing file
        shutil.copy2(save_path, backup_path)
        print(f"Created backup: {backup_path}")
        
        # Load data from the original file
        with open(save_path, 'rb') as f:
            existing_data = pickle.load(f)
        print(f"Successfully loaded data from original file: {save_path}")
        return existing_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chain Verbalizer')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--host', metavar='HOST')
    parser.add_argument('--port', default=9999, metavar='PORT')
    parser.add_argument('--workers', type=int, default=4, metavar='WORKERS', help='Number of worker threads')
    parser.add_argument('--domain', metavar='DOMAIN')
    parser.add_argument('--save_interval', type=int, default=5, metavar='SAVE_INTERVAL',
                        help='Number of documents to process before saving progress')
    parser.add_argument('--sequential', action='store_true', help='Run in sequential mode instead of parallel')
    args = parser.parse_args()
    print(vars(args))
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    annotator = ChainVerbalizer(args.host, args.port, config, args.domain)
    annotator.process_documents(args.workers, args.save_interval, args.sequential)