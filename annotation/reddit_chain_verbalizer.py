import argparse
import pickle

from pyhocon import ConfigFactory

from annotation.chain_verbalizer import ChainVerbalizer


class RedditChainVerbalizer(ChainVerbalizer):
    def __init__(self, host, port, config, domain):
        super().__init__(host, port, config, domain, excerpt=0)

        with open("./annotation/prompts/chain_verbalization/reddit_system_prompt.md", 'r', encoding='utf-8') as f:
            self.reasoning_system_prompt = f.read()

    def process_documents(self, num_workers, save_interval, sequential=False):
        with open(self.config["causal_annotations_path"], 'rb') as f:
            data = pickle.load(f)

        processed_docs = self.load_existing_progress()
        annotated_docs = processed_docs if processed_docs else {}

        total_docs = len(data)
        existing_docs = len(annotated_docs)
        docs_to_process = total_docs - existing_docs

        print(f"Total documents: {total_docs}", flush=True)
        print(f"Existing processed documents: {existing_docs}", flush=True)
        print(f"Documents to process: {docs_to_process}", flush=True)

        import concurrent.futures
        from tqdm import tqdm

        if sequential:
            from tqdm import tqdm
            with tqdm(total=docs_to_process, desc="Processing documents") as pbar:
                processed_count = 0
                for doc_idx, doc in data.items():
                    if doc_idx in annotated_docs:
                        continue
                    try:
                        processed_doc = self.process_document(doc)
                        annotated_docs[doc_idx] = processed_doc
                        processed_count += 1
                        pbar.update(1)
                        if processed_count % save_interval == 0:
                            self.save_progress(annotated_docs)
                            print(f"Progress saved after processing {processed_count} documents.", flush=True)
                    except Exception as e:
                        print(f"Error processing document {doc_idx}: {e}", flush=True)
                        pbar.update(1)
        else:
            unprocessed_items = {doc_idx: doc for doc_idx, doc in data.items()
                                  if doc_idx not in annotated_docs}

            if not unprocessed_items:
                print("All documents already processed!", flush=True)
                return

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self.process_document, doc): doc_idx
                           for doc_idx, doc in unprocessed_items.items()}
                with tqdm(total=len(unprocessed_items), desc="Processing documents") as pbar:
                    processed_count = 0
                    for future in concurrent.futures.as_completed(futures):
                        doc_idx = futures[future]
                        try:
                            doc = future.result()
                            annotated_docs[doc_idx] = doc
                            processed_count += 1
                            pbar.update(1)
                            if processed_count % save_interval == 0:
                                self.save_progress(annotated_docs)
                                print(f"Progress saved after processing {processed_count} documents.", flush=True)
                        except Exception as e:
                            print(f"Error processing document {doc_idx}: {e}", flush=True)
                            pbar.update(1)

        self.save_progress(annotated_docs)
        print("Processing complete. Final save completed.", flush=True)

    def process_document(self, doc, source=None, excerpt=0):
        import gc
        article = doc['text']
        event_chains = {}
        event_chain_idx = 0

        for pair in doc['event_pairs']:
            if pair.get('prediction', {}).get('relation') != 'causal':
                continue

            e1 = pair['event_1']
            e2 = pair['event_2']

            event_chain_str = f"(({e1['verb']}, {e1['object']}), CAUSAL, ({e2['verb']}, {e2['object']}))"

            chain_text = self.annotate(self.domain, event_chain_str,
                                       e1['sentence_text'], e2['sentence_text'],
                                       article)

            event_chains[event_chain_idx] = {
                'event_chain': event_chain_str,
                'chain_text': chain_text,
            }
            event_chain_idx += 1

        doc['event_chains'] = event_chains

        try:
            del article, event_chains
            gc.collect()
        except Exception as e:
            print(f"Error during garbage collection: {e}", flush=True)

        return doc

    def annotate(self, domain, event_chain, sentence_1, sentence_2, article):
        reasoning_user_prompt = (
            f"DOMAIN: {domain}\n"
            f"EVENT CHAIN: {event_chain}\n"
            f"SENTENCE_1: \"{sentence_1}\"\n"
            f"SENTENCE_2: \"{sentence_2}\"\n"
            f"ARTICLE: \"{article}\""
        )

        from schemas import EventChainSentence
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                reasoning_model_response = self.reasoning_model.chat(
                    self.reasoning_system_prompt,
                    reasoning_user_prompt,
                    think=self.think,
                    num_ctx=self.num_ctx)

                _, json_content = self.extract_thinking_response(reasoning_model_response)

                structured_response = self.output_model.chat(
                    self.structured_output_system_prompt,
                    json_content,
                    think=False,
                    repeat_penalty=True,
                    format=EventChainSentence.model_json_schema(),
                    num_ctx=self.num_ctx)

                try:
                    response = EventChainSentence.model_validate_json(structured_response)
                    return response.model_dump()['sentence']
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
        with open(self.config["causal_verbalizations_path"], 'wb') as f:
            pickle.dump(annotated_docs, f)

    def load_existing_progress(self):
        save_path = self.config["causal_verbalizations_path"]
        import os
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reddit Chain Verbalizer')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('--host', metavar='HOST', required=True)
    parser.add_argument('--port', default=9999, metavar='PORT')
    parser.add_argument('--workers', type=int, default=4, metavar='WORKERS')
    parser.add_argument('--domain', metavar='DOMAIN', required=True,
                        help='Domain context, e.g. "Parkinson\'s Disease" or "Long Covid"')
    parser.add_argument('--save_interval', type=int, default=5, metavar='SAVE_INTERVAL')
    parser.add_argument('--sequential', action='store_true')
    args = parser.parse_args()
    print(vars(args))
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    verbalizer = RedditChainVerbalizer(args.host, args.port, config, args.domain)
    verbalizer.process_documents(args.workers, args.save_interval, args.sequential)
