import argparse
import json
import pickle
import re
import concurrent.futures

from pyhocon import ConfigFactory
from tqdm import tqdm

from utils.ollama_client import Ollama


class ChainVerbalizer:
    def __init__(self, host, port, config, domain, model='llama3.3', thinking=False):
        self.ollama_client = Ollama(host,
                                    port,
                                    model,
                                    seed=42,
                                    temperature=0.1)
        self.thinking = thinking
        self.config = config

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
                if self.domain == 'guncontrol':
                    char_groups = self.guncontrol_char_group
                elif self.domain == 'immigration':
                    char_groups = self.immigration_char_group
                annotation = self.annotate(self.domain, chain, char_groups, article)
            # doc['event_chains'][event_chain_idx]['annotation'] = annotation
        return doc

    def save_progress(self, annotated_docs):
        with open(self.config["annotated_event_chains_path"], 'wb') as f:
            pickle.dump(annotated_docs, f)

    def annotate(self, domain, event_chain, char_groups, article):
        system_prompt = """I want you to generate plausible sentences that expand on a causal event chain from a news article from a specific domain. Events correspond to what we perceive around us and is denoted as a (VERB, OBJECT) pair. The object is the direct object of the verb in a linguistic sense. An example of an event is (arrest, people). The verb and object will correspond to a word in the article and may or may not be in their lemmatized form. An event chain comprises of two events connected by a causal relation. It'll be denoted as a tuple as follows: (EVENT_1, CAUSAL, EVENT_2). CAUSAL indicates that either EVENT_1 caused EVENT_2 or EVENT_2 caused EVENT_1. An example of an event chain is ((arrest, people), CAUSAL, (protest, legislation)). I want you to expand the event chain into a plausible sentence. Make sure to include all relevant characters and organizations in the sentence that you generate. You will be given a set of character groups to consider. For each event chain, you will receive: 1. DOMAIN: Immigration or Gun Control 2. EVENT CHAIN: (EVENT_1, CASUAL, EVENT_2) 3. CHARACTER GROUPS: The set of character groups 4. ARTICLE: Full article in which the events appear. Generate a very short sentence that expands the events in the event chain and the causal relationship between them in the context of the news article. Only keep details and characters relevant to the event chain. Your answer should be in JSON using the following format: {sentence: "a sentence that expands on the event chain"}. Do not generate anything else."""

        output_format = {"sentence": "A sentence that expands on the event chain."}
        user_prompt = f"DOMAIN: {domain} EVENT CHAIN: {event_chain} CHARACTER GROUPS: {char_groups} ARTICLE: {article} Your answer should be in JSON using the following format: {json.dumps(output_format)}. Do not generate anything else."

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.ollama_client.chat(system_prompt, user_prompt)
                try:
                    if self.thinking:
                        _ , json_content = self.extract_thinking_response(response)
                        response = json.loads(json_content)
                    else:
                        response = json.loads(response)
                    if 'sentence' in response:
                        return response['sentence']
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

    def extract_thinking_response(self, response):
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
        json_pattern = r'(\{.*?\}|\[.*?\])'
        json_match = re.search(json_pattern, response_without_think, re.DOTALL)
        json_content = json_match.group(0) if json_match else None
        
        return think_content, json_content


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotator')
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
                                domain='immigration',
                                model='deepseek-r1:32b-qwen-distill-q4_K_M',
                                thinking=True)
    annotator.process_documents(args.workers, args.save_interval, args.sequential)