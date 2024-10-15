import os
import requests

class OpenRouterClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.chat = ChatCompletions(self)

class ChatCompletions:
    def __init__(self, client):
        self.client = client

    def create(self, model, messages, temperature=0.7):
        headers = {
            "Authorization": f"Bearer {self.client.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tencent-ailab/persona-hub",
            "X-Title": "Persona Hub"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        response = requests.post(f"{self.client.base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()
