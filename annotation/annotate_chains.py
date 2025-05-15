import argparse
import json
import pickle
import concurrent.futures

from pyhocon import ConfigFactory
from tqdm import tqdm

from schemas import EventChainAnnotation
from utils.ollama_client import Ollama


class Annotator:
    def __init__(self, host, port, config, domain, model='llama3.3', thinking=False):
        self.reasoning_model = Ollama(host,
                                      port,
                                      model,
                                      seed=42,
                                      temperature=0.1)
        self.output_model = Ollama(host,
                                      port,
                                      'gemma3:12b',
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
        # for event_chain_idx, event_chain in doc['event_chains'].items():
        #     annotation = self.annotate(doc['text'], event_chain['chain_text'])
        #     doc['event_chains'][event_chain_idx]['annotation'] = annotation
        # return doc

        text = """PRIMARY ' WHOLE NATION' WATCHING SESSION'S VOTE ON GUN BILL When the Florida Legislature convenes at 2 p.m. today in special session, it is expected to pass a bill to punish adults who leave their guns within reach of children. The bill is expected to become law, and if it does, Florida will be the first state to prosecute and penalize gun owners with a statute that goes beyond existing negligence laws. "The whole nation is watching Florida," said Dennis Smith, director of public education for the Center for Handgun Violence, a non-profit research group based in Washington. "It's pretty clear: The availability of guns, loaded guns, in Florida has plagued the state with accidental shootings." The recent attention given to the gun safety bill, which passed only the House in regular session, was triggered by a tragic rash of accidents involving children shooting children. Silvio Claud Pierre, 4, became the latest victim Saturday night when he died at Tampa General Hospital. He had been listed in critical condition all week after undergoing seven hours of surgery June 11, the day he shot himself at his family's home. Silvio found a .25-caliber semiautomatic handgun under a couch while his mother was in the shower. In the last three weeks, in five incidents, two other children have been accidentally killed and four have been seriously injured by other children who found and fired their parents' loaded guns."""

        event_chain = "Florida was plagued by accidental shootings involving children, prompting the state legislature to pass a bill holding gun owners accountable for leaving firearms accessible to minors. The tragic killing of pupils in Connecticut led concerned children to express their worries about gun violence, prompting President Obama to announce new proposals aimed at reducing such incidents."

        annotation = self.annotate(text, event_chain)

        return doc

    def save_progress(self, annotated_docs):
        with open(self.config["annotated_event_chains_path"], 'wb') as f:
            pickle.dump(annotated_docs, f)

    def annotate(self, article, event_chain):
        with open("./annotation/prompts/role_and_stance.md", 'r', encoding='utf-8') as file:
            reasoning_system_prompt = file.read()

        with open("./annotation/prompts/role_descriptions.md", 'r', encoding='utf-8') as file:
            role_descriptions = file.read()

        reasoning_user_prompt = (f"DOMAIN: \"{self.domain}\" \n EVENT CHAIN: \"{event_chain}\" \n CHARACTER GROUPS:"
                       f"{self.guncontrol_char_group} \n ROLE DESCRIPTIONS: {role_descriptions} \n ARTICLE: \"{article}\" \n ")

        with open("./annotation/prompts/structured_output.md", 'r', encoding='utf-8') as file:
            structured_output_system_prompt = file.read()

        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                reasoning_model_response = self.reasoning_model.chat(reasoning_system_prompt,
                                                                     reasoning_user_prompt)
                structured_response = self.output_model.chat(structured_output_system_prompt,
                                                             reasoning_model_response,
                                                             format=EventChainAnnotation.model_json_schema())
                try:
                    response = json.loads(structured_response)
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
    parser.add_argument('--sequential', action='store_true', help='Run in sequential mode instead of parallel')
    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    annotator = Annotator(args.host, args.port, config,
                          domain='Gun Control',
                          model='deepseek-r1:70b-llama-distill-q4_K_M',
                          thinking=True)
    annotator.process_documents(args.workers, args.save_interval, args.sequential)
