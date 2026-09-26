import base64
import json
import os
import urllib.error
import urllib.request
from types import SimpleNamespace


class OpenRouterError(RuntimeError):
    pass


def _response_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        values = []
        for item in content:
            if isinstance(item, dict):
                values.append(str(item.get("text", "")))
            else:
                values.append(str(item))
        return "".join(values)
    return str(content or "")


def _part_blocks(part):
    text = getattr(part, "text", None)
    if text:
        return str(text)
    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None:
        data = getattr(inline_data, "data", None)
        mime_type = getattr(inline_data, "mime_type", "audio/wav") or "audio/wav"
        if data is None:
            raise OpenRouterError("OpenRouter audio input is missing data")
        audio_format = str(mime_type).split("/")[-1].replace("mpeg", "mp3")
        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(data).decode("ascii"),
                "format": audio_format,
            },
        }
    return ""


def _contents_messages(contents, config):
    messages = []
    system_instruction = getattr(config, "system_instruction", None)
    if system_instruction:
        messages.append({"role": "system", "content": str(system_instruction)})
    if isinstance(contents, str):
        contents = [SimpleNamespace(role="user", parts=[SimpleNamespace(text=contents)])]
    for content in contents or []:
        blocks = []
        for part in getattr(content, "parts", []) or []:
            block = _part_blocks(part)
            if block:
                blocks.append(block)
        if not blocks:
            continue
        text = "\n".join(block for block in blocks if isinstance(block, str))
        message_content = text if text and len(blocks) == 1 else blocks
        messages.append({"role": getattr(content, "role", "user") or "user", "content": message_content})
    return messages


class _OpenRouterResponse:
    def __init__(self, text, usage):
        self.prompt_feedback = None
        self.text = text
        part = SimpleNamespace(text=text, thought=False)
        self.candidates = [SimpleNamespace(content=SimpleNamespace(parts=[part]))]
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=int(usage.get("prompt_tokens") or 0),
            thoughts_token_count=int(usage.get("completion_tokens_details", {}).get("reasoning_tokens") or 0),
            candidates_token_count=int(usage.get("completion_tokens") or 0),
            total_token_count=int(usage.get("total_tokens") or 0),
        )


class _OpenRouterModels:
    ROUTER_MODELS = ("openrouter/free", "openrouter/auto", "openrouter/auto-beta", "openrouter/fusion")
    ROUTER_CONTEXT_LIMIT = 32768

    def __init__(self, client):
        self.client = client

    def list(self):
        payload = self.client._request("GET", "/models")
        models = []
        for item in payload.get("data", []):
            model_id = str(item.get("id") or "")
            if not model_id:
                continue
            if self.client.only_free and not self._is_free(model_id):
                continue
            context_length = int(item.get("context_length") or 8192)
            if model_id in self.ROUTER_MODELS:
                context_length = min(context_length, self.ROUTER_CONTEXT_LIMIT)
            models.append(
                SimpleNamespace(
                    name=model_id,
                    supported_actions=["generateContent"],
                    output_token_limit=context_length,
                    context_length=context_length,
                )
            )
        return models

    def _is_free(self, model_id):
        return model_id.endswith(":free") or model_id in self.ROUTER_MODELS

    def get(self, model):
        model_id = str(model or "")
        for item in self.list():
            if item.name == model_id:
                return item
        raise OpenRouterError(f"OpenRouter model not found: {model_id}")

    def count_tokens(self, model, contents):
        text = json.dumps(self._contents_text(contents), ensure_ascii=False)
        return SimpleNamespace(total_tokens=max(1, len(text) // 4))

    @staticmethod
    def _contents_text(contents):
        if isinstance(contents, str):
            return contents
        values = []
        for content in contents or []:
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    values.append(str(text))
        return "\n".join(values)

    def generate_content(self, model, contents, config):
        body = {
            "model": str(model),
            "messages": _contents_messages(contents, config),
            "temperature": getattr(config, "temperature", None),
            "top_p": getattr(config, "top_p", None),
        }
        if getattr(config, "top_k", None) is not None:
            body["top_k"] = config.top_k
        payload = self.client._request("POST", "/chat/completions", body)
        message = (payload.get("choices") or [{}])[0].get("message") or {}
        return _OpenRouterResponse(_response_text(message.get("content", "")), payload.get("usage") or {})

    def generate_content_stream(self, model, contents, config):
        yield self.generate_content(model, contents, config)


class OpenRouterClient:
    def __init__(self, api_key=None, base_url="https://openrouter.ai/api/v1", only_free=False, app_title="gemini-srt-translator"):
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not self.api_key:
            raise OpenRouterError("OpenRouter API key is missing; set OPENROUTER_API_KEY or pass --openrouter-key")
        self.base_url = str(base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self.only_free = bool(only_free)
        self.app_title = app_title
        self.models = _OpenRouterModels(self)

    def _request(self, method, path, body=None):
        url = f"{self.base_url}/{path.lstrip('/')}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "X-Title": self.app_title,
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise OpenRouterError(f"OpenRouter HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise OpenRouterError(f"OpenRouter request failed: {error}") from error
