from openai import OpenAI
import logging
import ollama
from typing import List, Dict, Any, Generator, Union
from django.conf import settings
from ..models import AIVendor

logger = logging.getLogger(__name__)

_OPENAPI_KEY=None

def set_openai_key(key):
    global _OPENAPI_KEY
    _OPENAPI_KEY = key

def create_openai():
    """Used by Whisper/STT and any legacy OpenAI chat paths."""
    global _OPENAPI_KEY
    return OpenAI(api_key=_OPENAPI_KEY)


# ---- Chat provider abstraction ----
Message = Dict[str, str]  # {"role": "system|user|assistant", "content": "..."}

class LLMProvider:
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[str, Generator[str, None, None]]:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = create_openai()

    def chat(self, messages, temperature=0.7, stream=False, **kwargs):
        max_tokens = kwargs.get("max_tokens")
        # (streaming optional later; keep behavior identical to current non-stream)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
'''
class OllamaProvider(LLMProvider):
    def __init__(self, host: str, model: str):
        self.model = model
        self.client = ollama.Client(host=host)

    def chat(self, messages, temperature=0.7, stream=False, **kwargs):
        # map OpenAI-style max_tokens to Ollama num_predict
        num_predict = kwargs.get("max_tokens")
        options = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = num_predict

        payload = {
            "model": self.model,
            "messages": messages,
            "options": options,
            "stream": stream,
        }
        if stream:
            def gen():
                for chunk in self.client.chat(**payload):
                    delta = (chunk.get("message") or {}).get("content", "")
                    if delta:
                        yield delta
            return gen()
        else:
            resp = self.client.chat(**payload)
            return (resp.get("message") or {}).get("content", "")
'''


'''
class OpenAIProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = create_openai()

    def chat(self, messages, temperature=0.7, stream=False, **kwargs):
        max_tokens = kwargs.get("max_tokens")
        if stream:
            # stream token deltas
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            def gen():
                for chunk in resp:
                    # v1 OpenAI client returns .choices[0].delta.content per chunk
                    choice = chunk.choices[0]
                    delta = getattr(choice.delta, "content", None)
                    if delta:
                        yield delta
            return gen()
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
'''
class OllamaProvider(LLMProvider):
    def __init__(self, host: str, model: str):
        self.model = model
        import ollama
        self.client = ollama.Client(host=host)

    def chat(self, messages, temperature=0.7, stream=False, **kwargs):
        num_predict = kwargs.get("max_tokens")
        options = {"temperature": temperature}
        #if num_predict is not None:
        #    options["num_predict"] = num_predict

        if isinstance(num_predict, int) and num_predict > 0:
            options["num_predict"] = num_predict
        # else: omit -> unlimited

        payload = {
            "model": self.model,
            "messages": messages,
            "options": options,
            "stream": stream,
        }
        if stream:
            def gen():
                for chunk in self.client.chat(**payload):
                    msg = chunk.get("message") or {}
                    delta = msg.get("content", "")
                    if delta:
                        yield delta
            return gen()
        else:
            resp = self.client.chat(**payload)
            return (resp.get("message") or {}).get("content", "")






def get_llm_provider_from_vendor(vendor: AIVendor, model: str) -> LLMProvider:
    """
    Create a chat provider based on DB-selected vendor enum.
    - vendor: AIVendor.OPEN_AI or AIVendor.OLLAMA
    - model: model name stored with the chat (e.g., "gpt-4o-mini" or "llama3")
    """
    # normalize in case an int slipped through
    if not isinstance(vendor, AIVendor):
        vendor = AIVendor(int(vendor))

    if vendor == AIVendor.OLLAMA:
        host = getattr(settings, "OLLAMA_HOST", "http://127.0.0.1:11434")
        fallback = getattr(settings, "OLLAMA_MODEL", "llama3")
        return OllamaProvider(host=host, model=(model or fallback))

    # default OPEN_AI
    fallback = getattr(settings, "OPENAI_MODEL", "gpt-3.5-turbo")
    return OpenAIProvider(model=(model or fallback))