import anthropic
import openai
from google import genai
from abc import ABC, abstractmethod
import time
import random


def retry_with_backoff(func, max_retries=5, initial_delay=1.0, backoff_factor=2.0, jitter=True):
    """
    Retry a function with exponential backoff on rate limit errors.
    
    Args:
        func: Function to call
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Multiply delay by this factor each retry
        jitter: Add random jitter to delay
        
    Returns:
        Result of function call
        
    Raises:
        Exception: If all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check for rate limit errors (429, quota exceeded, etc.)
            is_rate_limit = (
                '429' in str(e) or
                'rate_limit' in error_str or
                'quota' in error_str or
                'too_many_requests' in error_str or
                'resource_exhausted' in error_str
            )
            
            if not is_rate_limit or attempt == max_retries - 1:
                # Not a rate limit error, or last attempt - raise immediately
                raise
            
            # Calculate delay with exponential backoff
            delay = initial_delay * (backoff_factor ** attempt)
            if jitter:
                delay = delay * (0.5 + random.random())  # Add 0-50% jitter
            
            print(f"  ⏳ Rate limited. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    
    # Should not reach here
    raise last_error


class LLMProvider(ABC):
    """Base class for LLM providers with rate limiting support"""
    
    def __init__(self):
        # Configurable rate limiting parameters per provider
        self.max_retries = 5
        self.initial_delay = 1.0
        self.backoff_factor = 2.0
        self.between_call_delay = 0.0  # Delay between calls in seconds
    
    @abstractmethod
    def _predict_impl(self, prompt: str) -> str:
        """Implementation of predict - to be overridden by subclasses"""
        pass
    
    def predict(self, prompt: str) -> str:
        """Public predict with retry logic and rate limiting"""
        # Add inter-call delay
        if self.between_call_delay > 0:
            time.sleep(self.between_call_delay)
        
        # Call with retry logic
        return retry_with_backoff(
            lambda: self._predict_impl(prompt),
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            backoff_factor=self.backoff_factor
        )


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = genai.Client(api_key=api_key)
        # Gemini has tight rate limits - add delay and longer backoff
        self.max_retries = 5
        self.initial_delay = 3.0  # Start with 3s delay
        self.backoff_factor = 2.0
        self.between_call_delay = 1.0  # 1s between calls
    
    def _predict_impl(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                'max_output_tokens': 5,
                'temperature': 0.0,
            }
        )
        result = response.text.strip()
        # Extract only first character if it's 0 or 1
        if result and result[0] in ['0', '1']:
            return result[0]
        return result


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = anthropic.Anthropic(api_key=api_key)
        # Claude has moderate rate limits
        self.max_retries = 5
        self.initial_delay = 2.0  # Start with 2s delay
        self.backoff_factor = 2.0
        self.between_call_delay = 0.8  # 0.8s between calls
    
    def _predict_impl(self, prompt: str) -> str:
        message = self.client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=5,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        result = message.content[0].text.strip()
        # Extract only first character if it's 0 or 1
        if result and result[0] in ['0', '1']:
            return result[0]
        return result


class GPTProvider(LLMProvider):
    def __init__(self, api_key: str):
        super().__init__()
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        # GPT has good rate limits
        self.max_retries = 3
        self.initial_delay = 1.0  # Start with 1s delay
        self.backoff_factor = 2.0
        self.between_call_delay = 0.1  # 0.1s between calls
    
    def _predict_impl(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0
        )
        result = response.choices[0].message.content.strip()
        # Extract only first character if it's 0 or 1
        if result and result[0] in ['0', '1']:
            return result[0]
        return result


class MockLLMProvider(LLMProvider):
    """Mock provider for testing without API keys.
    
    Returns deterministic binary predictions based on prompt hash.
    Useful for unit tests and CI/CD pipelines.
    """
    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed
        self.random = random.Random(seed)
        self.between_call_delay = 0.0  # No delay for mocks
    
    def _predict_impl(self, prompt: str) -> str:
        """Return deterministic 0 or 1 based on prompt"""
        # Use hash of prompt to ensure consistency
        hash_val = hash(prompt) ^ self.seed
        return '0' if hash_val % 2 == 0 else '1'


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