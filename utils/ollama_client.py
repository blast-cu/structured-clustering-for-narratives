import ollama


class Ollama:
    def __init__(self, host, port, model, seed=42, temperature=1.0, top_p=None, top_k=None):
        server_host = f"{host}:{port}"
        self.client = ollama.Client(server_host)

        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        ollama.pull(self.model)

    def chat(self, system_prompt="", user_prompt="", think=False, repeat_penalty=False, format=None, num_ctx=2048):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if repeat_penalty:
            options: ollama.Options = {
                "seed": self.seed,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_ctx": num_ctx,
                "repeat_penalty": 1,
                "num_predict": 2048
            }
        else:
            options: ollama.Options = {
                "seed": self.seed,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_ctx": num_ctx,
                "num_predict": 2048
            }

        response = self.client.chat(self.model,
                                    messages=messages,
                                    format=format,
                                    think=think,
                                    keep_alive=-1,
                                    options=options)

        return response['message']['content']