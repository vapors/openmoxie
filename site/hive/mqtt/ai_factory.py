from openai import OpenAI
import logging
import ollama
from typing import List, Dict, Any, Generator, Union
from django.conf import settings
from ..models import AIVendor
# NEW: xai-sdk (sync)
try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import system as xai_system, user as xai_user, assistant as xai_assistant
    _XAI_SDK_OK = True
except Exception as _e:
    _XAI_SDK_OK = False

logger = logging.getLogger(__name__)

_OPENAPI_KEY=None
_XAI_API_KEY = None  # <— NEW

def set_openai_key(key):
    global _OPENAPI_KEY
    _OPENAPI_KEY = key


def set_xai_key(key):  # <— NEW
    global _XAI_API_KEY
    _XAI_API_KEY = key



def create_openai():
    """Used by Whisper/STT and any legacy OpenAI chat paths."""
    global _OPENAPI_KEY
    return OpenAI(api_key=_OPENAPI_KEY)




#def create_xai():
#    """xAI Grok: OpenAI-compatible API, different base_url."""
#    base = getattr(settings, "XAI_BASE_URL", "https://api.x.ai/v1")
#    return OpenAI(api_key=_XAI_API_KEY, base_url=base)


def create_xai():
    """Create xAI SDK client (sync)."""
    if not _XAI_SDK_OK:
        raise RuntimeError("xai-sdk not installed. Run `pip install xai-sdk`.")
    base = getattr(settings, "XAI_BASE_URL", None)  # usually None; SDK has default
    if base:
        return XAIClient(api_key=_XAI_API_KEY, base_url=base)
    return XAIClient(api_key=_XAI_API_KEY)



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

# --- xAI provider (Grok) ---
'''
class XAIProvider(LLMProvider):  # <— NEW
    def __init__(self, model: str):
        self.model = model
        self.client = create_xai()

    def chat(self, messages, temperature=0.7, stream=False, **kwargs):
        max_tokens = kwargs.get("max_tokens")
        resp = self.client.chat.completions.create(
            model=self.model,                # e.g., "grok-2" or "grok-2-mini"
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,                    # keep final-only
        )
        return resp.choices[0].message.content
'''
# --- xAI provider (Grok via xai-sdk) ----------------------------------------
def _to_xai_messages(messages: List[Message]):
    """Map OpenAI-style dicts to xai_sdk.chat message objects."""
    out = []
    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "")
        if role == "system":
            out.append(xai_system(content))
        elif role == "assistant":
            out.append(xai_assistant(content))
        else:
            # Treat anything else (including 'user') as user
            out.append(xai_user(content))
    return out

class XAIProvider(LLMProvider):
    def __init__(self, model: str):
        if not _XAI_SDK_OK:
            raise RuntimeError("xai-sdk not installed. Run `pip install xai-sdk`.")
        self.model = model
        self.client = create_xai()

    def chat(self, messages, temperature=0.7, stream=False, **kwargs):
        # NOTE: We run final-only (no partials), to match your client behavior.
        max_tokens = kwargs.get("max_tokens")

        xai_messages = _to_xai_messages(messages)

        # Some versions of xai-sdk accept `sampling_params=...` at chat.create.
        # We set both common keys defensively; SDK will ignore unknown fields.
        sampling_params = {
            "temperature": float(temperature),
        }
        if isinstance(max_tokens, int) and max_tokens > 0:
            sampling_params["max_tokens"] = max_tokens
            sampling_params["max_output_tokens"] = max_tokens

        try:
            chat = self.client.chat.create(
                model=self.model,
                messages=xai_messages,
                sampling_params=sampling_params
            )
        except TypeError:
            # Older SDKs may not support sampling_params at create-time.
            chat = self.client.chat.create(
                model=self.model,
                messages=xai_messages
            )

        # Final-only sample
        try:
            resp = chat.sample()
        except TypeError:
            # Fallback for SDKs that accept params at sample-time
            resp = chat.sample(temperature=float(temperature))

        # xai-sdk returns an object with .content (string). Be robust just in case.
        text = getattr(resp, "content", None)
        if isinstance(text, list):
            # Rare case: if content is chunked parts, concatenate text fields.
            text = "".join(
                (p.get("text", "") if isinstance(p, dict) else str(p))
                for p in text
            )
        return text or ""


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





# ---- Factory ----------------------------------------------------------------

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

    if vendor == AIVendor.XAI:
        fallback = getattr(settings, "XAI_MODEL", "grok-3-mini")
        return XAIProvider(model=(model or fallback))


    # default OPEN_AI
    fallback = getattr(settings, "OPENAI_MODEL", "gpt-3.5-turbo")
    return OpenAIProvider(model=(model or fallback))