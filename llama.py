import random
import transformers
import torch


class Llama:
    def __init__(self, model_id, seed=42, max_tokens=4096, temperature=1.0):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)

        self.max_tokens = max_tokens
        self.temperature = temperature

        print("CUDA: " + str(torch.cuda.is_available()))

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def prompt_llama(self, system_prompt="", user_prompt=""):
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        return outputs[0]["generated_text"][-1]