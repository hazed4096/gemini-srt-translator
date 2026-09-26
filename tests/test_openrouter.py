import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gemini_srt_translator as gst
from gemini_srt_translator.main import GeminiSRTTranslator
from gemini_srt_translator.openrouter import (
    OpenRouterClient,
    OpenRouterError,
    _contents_messages,
    _response_text,
)


class TestResponseParsing(unittest.TestCase):
    def test_response_text_from_string(self):
        self.assertEqual(_response_text("hello"), "hello")

    def test_response_text_from_content_blocks(self):
        self.assertEqual(
            _response_text([{"type": "text", "text": "he"}, {"type": "text", "text": "llo"}]),
            "hello",
        )

    def test_response_text_from_none(self):
        self.assertEqual(_response_text(None), "")

    def test_messages_include_system_and_user_roles(self):
        config = SimpleNamespace(system_instruction="be a translator")
        contents = [
            SimpleNamespace(role="user", parts=[SimpleNamespace(text="first")]),
            SimpleNamespace(role="model", parts=[SimpleNamespace(text="second")]),
        ]
        messages = _contents_messages(contents, config)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "be a translator"},
                {"role": "user", "content": "first"},
                {"role": "model", "content": "second"},
            ],
        )

    def test_audio_part_becomes_input_audio_block(self):
        part = SimpleNamespace(
            text=None,
            inline_data=SimpleNamespace(data=b"\x00\x01", mime_type="audio/mpeg"),
        )
        contents = [SimpleNamespace(role="user", parts=[part])]
        messages = _contents_messages(contents, SimpleNamespace(system_instruction=None))
        block = messages[0]["content"][0]
        self.assertEqual(block["type"], "input_audio")
        self.assertEqual(block["input_audio"]["format"], "mp3")
        self.assertEqual(block["input_audio"]["data"], "AAE=")


class TestOpenRouterClient(unittest.TestCase):
    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(OpenRouterError):
                OpenRouterClient(api_key=None)

    def test_list_models_filters_free_only(self):
        client = OpenRouterClient(api_key="key", only_free=True)
        with patch.object(
            client,
            "_request",
            return_value={
                "data": [
                    {"id": "vendor/model-a:free", "context_length": 4096},
                    {"id": "openrouter/free", "context_length": 200000},
                    {"id": "vendor/model-b", "context_length": 8192},
                ]
            },
        ):
            models = client.models.list()
        self.assertEqual([model.name for model in models], ["vendor/model-a:free", "openrouter/free"])
        self.assertEqual(models[0].output_token_limit, 4096)

    def test_router_context_limit_is_capped(self):
        client = OpenRouterClient(api_key="key")
        with patch.object(
            client,
            "_request",
            return_value={"data": [{"id": "openrouter/free", "context_length": 200000}]},
        ):
            model = client.models.get(model="openrouter/free")
        self.assertEqual(model.output_token_limit, 32768)

    def test_get_model_raises_for_unknown_model(self):
        client = OpenRouterClient(api_key="key")
        with patch.object(client, "_request", return_value={"data": [{"id": "vendor/model-a:free"}]}):
            with self.assertRaises(OpenRouterError):
                client.models.get(model="vendor/missing")

    def test_generate_content_returns_genai_shaped_response(self):
        client = OpenRouterClient(api_key="key")
        payload = {
            "choices": [{"message": {"content": '{"lines": []}'}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
        }
        with patch.object(client, "_request", return_value=payload) as request:
            response = client.models.generate_content(
                model="vendor/model-a:free",
                contents=[SimpleNamespace(role="user", parts=[SimpleNamespace(text="hi")])],
                config=SimpleNamespace(system_instruction="sys", temperature=0.2, top_p=0.9, top_k=None),
            )
        body = request.call_args.args[2]
        self.assertEqual(body["model"], "vendor/model-a:free")
        self.assertEqual(body["temperature"], 0.2)
        self.assertNotIn("top_k", body)
        self.assertIsNone(response.prompt_feedback)
        self.assertEqual(response.text, '{"lines": []}')
        self.assertEqual(response.usage_metadata.prompt_token_count, 11)
        self.assertEqual(response.usage_metadata.candidates_token_count, 3)
        self.assertEqual(response.usage_metadata.total_token_count, 14)
        self.assertFalse(response.candidates[0].content.parts[0].thought)

    def test_generate_content_stream_yields_single_chunk(self):
        client = OpenRouterClient(api_key="key")
        payload = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        with patch.object(client, "_request", return_value=payload):
            chunks = list(
                client.models.generate_content_stream(
                    model="vendor/model-a:free",
                    contents=[SimpleNamespace(role="user", parts=[SimpleNamespace(text="hi")])],
                    config=SimpleNamespace(system_instruction=None, temperature=None, top_p=None, top_k=None),
                )
            )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "ok")

    def test_count_tokens_is_positive(self):
        client = OpenRouterClient(api_key="key")
        tokens = client.models.count_tokens(
            model="vendor/model-a:free",
            contents=[SimpleNamespace(role="user", parts=[SimpleNamespace(text="hello world")])],
        )
        self.assertGreaterEqual(tokens.total_tokens, 1)


class TestTranslatorOpenRouterIntegration(unittest.TestCase):
    def test_getmodels_returns_raw_openrouter_ids(self):
        with patch.object(GeminiSRTTranslator, "_get_client") as mock_get_client:
            client = OpenRouterClient(api_key="key")
            with patch.object(
                client,
                "_request",
                return_value={"data": [{"id": "vendor/model-a:free"}, {"id": "vendor/model-b"}]},
            ):
                mock_get_client.return_value = client
                translator = GeminiSRTTranslator(provider="openrouter", openrouter_api_key="key")
                models = translator.getmodels()
        self.assertEqual(models, ["vendor/model-a:free", "vendor/model-b"])

    def test_default_model_is_free_router_when_omitted(self):
        translator = GeminiSRTTranslator(provider="openrouter", openrouter_api_key="key", model_name=None)
        self.assertEqual(translator.model_name, "openrouter/free")

    def test_free_only_accepts_free_router(self):
        translator = GeminiSRTTranslator(
            provider="openrouter",
            openrouter_api_key="key",
            model_name="openrouter/free",
            openrouter_only_free=True,
        )
        self.assertEqual(translator.model_name, "openrouter/free")

    def test_free_only_rejects_paid_model(self):
        with self.assertRaises(OpenRouterError):
            GeminiSRTTranslator(
                provider="openrouter",
                openrouter_api_key="key",
                model_name="google/gemini-2.5-flash-lite",
                openrouter_only_free=True,
            )

    def test_get_client_returns_openrouter_client(self):
        translator = GeminiSRTTranslator(provider="openrouter", openrouter_api_key="key")
        self.assertIsInstance(translator._get_client(), OpenRouterClient)

    def test_gemini_provider_still_uses_genai(self):
        with patch("gemini_srt_translator.main.genai.Client") as mock_client:
            translator = GeminiSRTTranslator(gemini_api_key="test-key", model_name="gemini-2.5-flash")
            translator._get_client()
        mock_client.assert_called_once_with(api_key="test-key")


class TestModuleGlobals(unittest.TestCase):
    def test_translate_passes_openrouter_settings(self):
        gst.provider = "openrouter"
        gst.openrouter_api_key = "key"
        gst.openrouter_base_url = "https://openrouter.ai/api/v1"
        gst.openrouter_app_title = "test-app"
        gst.openrouter_only_free = True
        gst.input_file = None
        gst.video_file = None
        gst.output_file = None
        gst.skip_upgrade = True
        try:
            with patch("gemini_srt_translator.main.GeminiSRTTranslator") as mock_translator:
                gst.translate()
            kwargs = mock_translator.call_args.kwargs
            self.assertEqual(kwargs["provider"], "openrouter")
            self.assertEqual(kwargs["openrouter_api_key"], "key")
            self.assertTrue(kwargs["openrouter_only_free"])
        finally:
            gst.provider = "gemini"
            gst.openrouter_api_key = None
            gst.openrouter_only_free = False


if __name__ == "__main__":
    unittest.main()
