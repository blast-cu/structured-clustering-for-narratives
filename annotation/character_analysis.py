import argparse
import os
import pickle
import re
import shutil
import concurrent.futures

from pyhocon import ConfigFactory
from tqdm import tqdm

from schemas import ImmigrationEventChainAnnotation, GunControlEventChainAnnotation
from utils.ollama_client import Ollama


class CharacterAnalyzer:
    def __init__(self, host, port, config, domain, use_excerpt=False):
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

        if use_excerpt:
            with open("./annotation/prompts/character_analysis/excerpt_system_prompt.md", 'r', encoding='utf-8') as file:
                self.reasoning_system_prompt = file.read()
        else:
            with open("./annotation/prompts/character_analysis/system_prompt.md", 'r', encoding='utf-8') as file:
                self.reasoning_system_prompt = file.read()
        with open("./annotation/prompts/character_analysis/structured_output.md", 'r', encoding='utf-8') as file:
            self.structured_output_system_prompt = file.read()

        self.domain = domain
        self.immigration_char_group = '{Immigrants, Refugees, Asylum Seekers, Workers, Politicians, Law Enforcement, ' \
                                      'Judiciary, Government, Immigration Advocates, Other}'
        self.guncontrol_char_group = '{Politicans, Gun Control Advocates, Gun Right Advocates, Law Enforcement, Judiciary, Government, Gun Crime Victims, Other}'

        if self.domain == 'guncontrol':
            with open("./annotation/prompts/character_analysis/guncontrol_role_descriptions.md", 'r', encoding='utf-8') as file:
                self.role_descriptions = file.read()
        elif self.domain == "immigration":
            with open("./annotation/prompts/character_analysis/immigration_role_descriptions.md", 'r', encoding='utf-8') as file:
                self.role_descriptions = file.read()

    def process_documents(self, num_workers, save_interval, use_excerpt=False, sequential=False):
        with open(self.config["event_chains_path"], 'rb') as f:
            data = pickle.load(f)
        
        processed_docs = self.load_existing_progress()
        if len(processed_docs) > 0:
            annotated_docs = processed_docs
        else:
            annotated_docs = {}
        
        total_docs = len(data)
        existing_docs = len(annotated_docs)
        docs_to_process = total_docs - existing_docs

        print(f"Total documents: {total_docs}", flush=True)
        print(f"Existing processed documents: {existing_docs}", flush=True)
        print(f"Documents to process: {docs_to_process}", flush=True)

        if sequential:
            # Sequential processing - only process unprocessed documents
            with tqdm(total=docs_to_process, desc="Processing documents") as pbar:
                processed_count = 0
                for doc_idx, doc in data.items():
                    if doc_idx in annotated_docs:
                        # Skip already processed documents - don't update progress bar
                        continue
                    try:
                        processed_doc = self.process_document(doc, use_excerpt)
                        annotated_docs[doc_idx] = processed_doc
                        processed_count += 1
                        
                        # Update the progress bar
                        pbar.update(1)
                        
                        # Save annotated_docs at regular intervals
                        if processed_count % save_interval == 0:
                            self.save_progress(annotated_docs)
                            print(f"Progress saved after processing {processed_count} documents.", flush=True)
                            
                    except Exception as e:
                        print(f"Error processing document {doc_idx}: {e}", flush=True)
                        pbar.update(1)  # Still update progress bar even if there's an error
        else:
            # Parallel processing - only submit futures for unprocessed documents
            unprocessed_items = {doc_idx: doc for doc_idx, doc in data.items() 
                            if doc_idx not in annotated_docs}
            
            if not unprocessed_items:
                print("All documents already processed!", flush=True)
                return

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self.process_document, doc, use_excerpt): doc_idx for doc_idx, doc in unprocessed_items.items()}
                # Initialize tqdm progress bar to track document processing
                with tqdm(total=len(unprocessed_items), desc="Processing documents") as pbar:
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
                            pbar.update(1)  # Still update progress bar even if there's an error

        # Final save at the end, in case it wasn't saved during the last interval
        self.save_progress(annotated_docs)
        print(f"Processing complete. Final save completed.", flush=True)

    def process_document(self, doc, use_excerpt=False):
        if self.domain == 'guncontrol':
            char_groups = self.guncontrol_char_group
            domain = "Gun Control"
        elif self.domain == 'immigration':
            char_groups = self.immigration_char_group
            domain = "Immigration"

        if 'event_chains' in doc:
            for event_chain_idx, event_chain in doc['event_chains'].items():
                if use_excerpt and 'excerpt' in event_chain:
                    article = event_chain['excerpt']
                else:
                    article = doc['text']
                annotation = self.annotate(domain, event_chain['chain_text'], char_groups, article)
                doc['event_chains'][event_chain_idx]['annotation'] = annotation

        return doc

    def annotate(self, domain, event_chain, char_groups, article):
        reasoning_user_prompt = (f"DOMAIN: \"{domain}\" \n EVENT CHAIN: \"{event_chain}\" \n CHARACTER GROUPS:"
                       f"{char_groups} \n ROLE DESCRIPTIONS: {self.role_descriptions} \n ARTICLE: \"{article}\" \n ")

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                reasoning_model_response = self.reasoning_model.chat(self.reasoning_system_prompt,
                                                                     reasoning_user_prompt,
                                                                     think=True,
                                                                     num_ctx=self.num_ctx)
                if domain == 'Gun Control':
                    structured_response = self.output_model.chat(self.structured_output_system_prompt,
                                                                 reasoning_model_response,
                                                                 think=False,
                                                                 repeat_penalty=True,
                                                                 format=GunControlEventChainAnnotation.model_json_schema(),
                                                                 num_ctx=self.num_ctx)
                elif domain == 'Immigration':
                    structured_response = self.output_model.chat(self.structured_output_system_prompt,
                                                                 reasoning_model_response,
                                                                 think=False,
                                                                 repeat_penalty=True,
                                                                 format=ImmigrationEventChainAnnotation.model_json_schema(),
                                                                 num_ctx=self.num_ctx)
                try:
                    response = None
                    if domain == 'Gun Control':
                        response = GunControlEventChainAnnotation.model_validate_json(structured_response)
                    elif domain == 'Immigration':
                        response = ImmigrationEventChainAnnotation.model_validate_json(structured_response)
                    response = response.model_dump()
                    return response
                except Exception as e:
                    print("Exception: " + str(e), flush=True)
                    print("Invalid response. Please try again.", flush=True)
                    retry_count += 1
            except Exception as e:
                print("Exception: " + str(e), flush=True)
                print("Ollama Error. Please try again.", flush=True)
                retry_count += 1
        return None

    def save_progress(self, annotated_docs):
        with open(self.config["char_event_chains_path"], 'wb') as f:
            pickle.dump(annotated_docs, f)
    
    def load_existing_progress(self):
        """Load existing processed documents if save file exists, creating a backup first"""
        save_path = self.config["char_event_chains_path"]

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
        
        # Find all existing backup files and extract their numbers
        backup_numbers = []
        
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                # Check if the file matches the backup pattern
                backup_pattern = f"{base_name}_backup_(\d+){re.escape(extension)}"
                match = re.match(backup_pattern, filename)
                if match:
                    backup_numbers.append(int(match.group(1)))
        
        # Return the next highest number
        if backup_numbers:
            return max(backup_numbers) + 1
        else:
            return 1

    @staticmethod
    def create_backup_and_load(save_path):
        """Create a backup of existing save file and load the data from the original file"""
        backup_number = CharacterAnalyzer.get_next_backup_number(save_path)
        
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
    parser = argparse.ArgumentParser(description='CharacterAnalyzer')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--host', metavar='HOST')
    parser.add_argument('--port', default=9999, metavar='PORT')
    parser.add_argument('--workers', type=int, default=4, metavar='WORKERS', help='Number of worker threads')
    parser.add_argument('--domain', metavar='DOMAIN')
    parser.add_argument('--use_excerpt', action='store_true')
    parser.add_argument('--save_interval', type=int, default=10, metavar='SAVE_INTERVAL',
                        help='Number of documents to process before saving progress')
    parser.add_argument('--sequential', action='store_true', help='Run in sequential mode instead of parallel')
    args = parser.parse_args()
    print(vars(args))
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    annotator = CharacterAnalyzer(args.host, args.port, config, args.domain, args.use_excerpt)
    annotator.process_documents(args.workers, args.save_interval, args.use_excerpt, args.sequential)
