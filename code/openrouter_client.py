import os
from openai import OpenAI

class OpenRouterClient(OpenAI):
    def __init__(self, api_key=None):
        super().__init__(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.headers.update({
            "HTTP-Referer": "https://github.com/tencent-ailab/persona-hub",
            "X-Title": "Persona Hub"
        })

    def chat_create(self, model, messages, temperature=0.7, **kwargs):
        return self.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            headers=self.headers,
            **kwargs
        )
