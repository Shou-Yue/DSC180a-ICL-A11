import anthropic
import openai
import google.generativeai as genai
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def predict(self, prompt: str) -> str:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def predict(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()

class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def predict(self, prompt: str) -> str:
        message = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()

class GPTProvider(LLMProvider):
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
    
    def predict(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        return response.choices[0].message.content.strip()

def get_provider(provider_name: str, api_key: str) -> LLMProvider:
    """Factory function to get the appropriate provider"""
    providers = {
        "gemini": GeminiProvider,
        "claude": ClaudeProvider,
        "gpt": GPTProvider
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    return providers[provider_name](api_key)