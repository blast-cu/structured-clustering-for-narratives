import argparse
import ollama


class Ollama:
    def __init__(self, host, port, model, seed=42, temperature=1.0):
        # server_host = f"{host}:{port}"
        server_host = "127.0.0.1:11434"
        print(f"Connecting to Ollama server at {server_host}")
        self.client = ollama.Client(server_host)

        self.model = model
        self.seed = seed
        self.temperature = temperature
        print(f"Ollama client initialized with model: {model}")

        ollama.pull("llama3.3")

    def chat(self, system_prompt="", user_prompt=""):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        options: ollama.Options = {
            "seed": self.seed,
            "temperature": self.temperature
        }

        response = self.client.chat(self.model,
                                    messages=messages,
                                    options=options)

        return response['message']['content']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ollama Client')
    parser.add_argument('--host', metavar='HOST')
    parser.add_argument('--port', default=9999, metavar='PORT')
    args = parser.parse_args()
    ollama = Ollama(args.host,
                    args.port,
                    'llama3:70b-instruct-q4_0',
                    seed=42,
                    temperature=0.1)

    system_prompt = """You are an annotator who is developing a dataset for an event type similarity task. An event is denoted by a (verb, object) tuple. You are given two events and the corresponding sentences in which they appear. Your task is to annotate whether the two events are of the same type. Your answer should be in JSON using the following format: {"answer": "yes/no", "reason": "your reasoning for the answer"}."""

    user_prompt = """First Event: "(throw, tantrum): \" When I threw temper tantrums /" Second Event: "(cast, shadow): The confrontations are casting a shadow over Mideast peace talks in Paris ." Are these two events of the same type?"""

    print(ollama.chat(system_prompt, user_prompt))

