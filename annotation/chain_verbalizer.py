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
    def __init__(self, host, port, config, domain, excerpt=0):
        self.config = config

        print("Reasoning model: " + self.config['reasoning_model'], flush=True)
        self.reasoning_model = Ollama(host,
                                        port,
                                        self.config['reasoning_model'],
                                        self.config['seed'],
                                        self.config['temperature'])

        print("Output model: " + self.config['output_model'], flush=True)
        self.output_model = Ollama(host,
                                    port,
                                    self.config['output_model'],
                                    self.config['seed'],
                                    self.config['temperature'])
        
        self.num_ctx = self.config['num_ctx']
        self.think = self.config['think']

        if excerpt > 0:
            with (open("./annotation/prompts/chain_verbalization/excerpt_system_prompt.md", 'r', encoding='utf-8') as
                  file):
                self.reasoning_system_prompt = file.read()
        else:
            with open("./annotation/prompts/chain_verbalization/system_prompt.md", 'r', encoding='utf-8') as file:
                self.reasoning_system_prompt = file.read()
        with open("./annotation/prompts/chain_verbalization/structured_output.md", 'r', encoding='utf-8') as file:
            self.structured_output_system_prompt = file.read()

        self.domain = domain
        self.immigration_char_group = '{Immigrants, Refugees, Asylum Seekers, Workers, Politicians, Law Enforcement, ' \
                                      'Judiciary, Government, Immigration Advocates}'
        self.guncontrol_char_group = ('{Politicans, Gun Control Advocates, Gun Rights Advocates, Law Enforcement, '
                                      'Judiciary, Government, Gun Crime Victims}')



    def process_documents(self, num_workers, save_interval, source, excerpt=0, sequential=False):
        with open(self.config["relations_path"], 'rb') as f:
            data = pickle.load(f)

        if excerpt > 0:
            data = self.generate_event_2_sentence_map(data)

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
                        processed_doc = self.process_document(doc, source, excerpt)
                        annotated_docs[doc_idx] = processed_doc
                        processed_count += 1
                        
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
                futures = {executor.submit(self.process_document, doc, source, excerpt): doc_idx for doc_idx, doc in unprocessed_items.items()}
                # Initialize tqdm progress bar to track document processing
                with tqdm(total=len(unprocessed_items)) as pbar:
                    processed_count = 0
                    
                    # Process documents and save at regular intervals
                    for future in concurrent.futures.as_completed(futures):
                        doc_idx = futures[future]
                        try:
                            doc = future.result()
                            annotated_docs[doc_idx] = doc
                            processed_count += 1

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

    def process_document(self, doc, source, excerpt=0):
        article = ''
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
            doc_relations = dict(sorted(doc['relations'].items(), key=lambda x: x[0][0]))
            for key in doc_relations:
                event_chain = {}
                if doc_relations[key][1] == 'causal':
                    event_1 = event_map[key[0]]
                    relation = doc_relations[key][1].upper()
                    event_2 = event_map[key[1]]
                    event_chain[
                        'event_chain'] = f"(({event_1[0]}, {event_1[1]}), {relation}, ({event_2[0]}, {event_2[1]}))"

                    if source == 'partisanship' and excerpt > 0:
                        excerpt_lb, excerpt_ub = self.get_excerpt(doc, excerpt, key)

                        for idx in range(excerpt_lb, excerpt_ub + 1):
                            article += doc['sentences'][str(idx)]['text'] + ' '
                            event_chain['excerpt'] = article.strip()
                    else:
                        article = doc['text']
                        event_chain['excerpt'] = None

                    event_chain['chain_text'] = self.annotate(self.domain,
                                                              event_chain['event_chain'],
                                                              char_groups,
                                                              article.strip(),
                                                              excerpt)

                if event_chain:
                    event_chains[event_chain_idx] = event_chain
                    event_chain_idx += 1

                if source == 'partisanship' and len(event_chains) >= self.config['event_chain_num_threshold']:
                    print(f"Reached event chain threshold of {self.config['event_chain_num_threshold']}. Stopping further processing.", flush=True)
                    break

        doc['event_chains'] = event_chains

        try:
            del article, event_map, event_chains
            gc.collect()
        except Exception as e:
            print(f"Error during garbage collection: {e}", flush=True)

        return doc

    def annotate(self, domain, event_chain, char_groups, article, excerpt=0):
        if excerpt > 0:
            reasoning_user_prompt = f"DOMAIN: {domain} EVENT CHAIN: {event_chain} CHARACTER GROUPS: {char_groups} " \
                                    f"ARTICLE EXCERPT: {article}"
        else:
            reasoning_user_prompt = f"DOMAIN: {domain} EVENT CHAIN: {event_chain} CHARACTER GROUPS: {char_groups} ARTICLE: {article}"

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                reasoning_model_response = self.reasoning_model.chat(self.reasoning_system_prompt,
                                                                     reasoning_user_prompt,
                                                                     think=self.think,
                                                                     num_ctx=self.num_ctx)

                _, json_content = self.extract_thinking_response(reasoning_model_response)

                structured_response = self.output_model.chat(self.structured_output_system_prompt,
                                                             json_content,
                                                             think=False,
                                                             repeat_penalty=True,
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

    @staticmethod
    def generate_event_2_sentence_map(data):
        """Generate a mapping of event IDs to sentences for each document"""
        for doc_idx, doc in data.items():
            event_2_sentence_map = {}
            for sentence_idx, sentence in doc['sentences'].items():
                for event in sentence['events']:
                    event_id = int(event[0])
                    event_2_sentence_map[event_id] = sentence_idx
            doc['event_2_sentence_map'] = event_2_sentence_map
        return data

    @staticmethod
    def get_excerpt(doc, excerpt, key):
        sent_ids = list(doc['sentences'].keys())
        sent_lb = int(sent_ids[0])
        sent_ub = int(sent_ids[-1])

        event_1_sent_id = int(doc['event_2_sentence_map'][key[0]])
        event_2_sent_id = int(doc['event_2_sentence_map'][key[1]])
        
        # Ensure event_1 comes before event_2 (swap if necessary)
        if event_1_sent_id > event_2_sent_id:
            event_1_sent_id, event_2_sent_id = event_2_sent_id, event_1_sent_id

        # Calculate target excerpt parameters
        target_before = excerpt  # target sentences before event_1
        target_after = excerpt   # target sentences after event_2
        
        # Calculate the span between events (inclusive)
        event_span = event_2_sent_id - event_1_sent_id + 1
        
        # Calculate available sentences on each side
        available_before = event_1_sent_id - sent_lb
        available_after = sent_ub - event_2_sent_id
        
        # Determine how many sentences we can actually get on each side
        actual_before = min(target_before, available_before)
        actual_after = min(target_after, available_after)
        
        # Calculate shortfalls
        shortfall_before = target_before - actual_before
        shortfall_after = target_after - actual_after
        
        # Compensate for shortfall on one side by adding to the other side
        if shortfall_before > 0:
            # We couldn't get enough sentences before event_1, try to get more after event_2
            additional_after = min(shortfall_before, available_after - actual_after)
            actual_after += additional_after
        
        if shortfall_after > 0:
            # We couldn't get enough sentences after event_2, try to get more before event_1
            additional_before = min(shortfall_after, available_before - actual_before)
            actual_before += additional_before
        
        # Set final excerpt bounds
        excerpt_lb = event_1_sent_id - actual_before
        excerpt_ub = event_2_sent_id + actual_after

        actual_excerpt_length = (excerpt_ub - excerpt_lb + 1)
        target_excerpt_length = (2 * excerpt) + 3

        if actual_excerpt_length > target_excerpt_length:
            raise Exception(
                f"Excerpt length {actual_excerpt_length} exceeds target length {target_excerpt_length}. "
                f"Excerpt bounds: {excerpt_lb} to {excerpt_ub}, "
                f"Event span: {event_span}, "
                f"Available before: {available_before}, Available after: {available_after}, "
                f"Actual before: {actual_before}, Actual after: {actual_after}"
            )

        return excerpt_lb, excerpt_ub
    
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
        backup_number = ChainVerbalizer.get_next_backup_number(save_path)
        
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
    parser.add_argument('--source', metavar='SOURCE')
    parser.add_argument('--domain', metavar='DOMAIN')
    parser.add_argument('--excerpt', type=int, default=0, metavar='EXCERPT', help='Use excerpt instead of full article')
    parser.add_argument('--save_interval', type=int, default=5, metavar='SAVE_INTERVAL',
                        help='Number of documents to process before saving progress')
    parser.add_argument('--sequential', action='store_true', help='Run in sequential mode instead of parallel')
    args = parser.parse_args()
    print(vars(args))
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    annotator = ChainVerbalizer(args.host, args.port, config, args.domain, args.excerpt)
    annotator.process_documents(args.workers, args.save_interval, args.source, args.excerpt, args.sequential)