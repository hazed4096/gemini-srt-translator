import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import gemini_srt_translator as gst
from gemini_srt_translator.main import GeminiSRTTranslator


class MockModel:
    def __init__(self, name, supported_actions=None):
        self.name = name
        self.supported_actions = supported_actions


class TestGeminiSRTTranslatorGetModels(unittest.TestCase):
    @patch.object(GeminiSRTTranslator, "_get_client")
    def test_getmodels_non_enterprise_filters_generate_content(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.models.list.return_value = [
            MockModel("models/gemini-2.5-flash", supported_actions=["generateContent"]),
            MockModel("models/gemini-2.5-pro", supported_actions=["generateContent", "countTokens"]),
            MockModel("models/text-embedding-004", supported_actions=["embedContent"]),
            MockModel("models/imagen-3.0", supported_actions=None),
        ]
        mock_get_client.return_value = mock_client

        translator = GeminiSRTTranslator(gemini_api_key="test-key", use_enterprise=False)
        models = translator.getmodels()

        self.assertEqual(models, ["gemini-2.5-flash", "gemini-2.5-pro"])
        mock_client.models.list.assert_called_once()

    @patch.object(GeminiSRTTranslator, "_get_client")
    def test_getmodels_enterprise_strips_publisher_prefix(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.models.list.return_value = [
            MockModel("publishers/google/models/gemini-2.5-flash"),
            MockModel("publishers/google/models/gemini-2.5-pro"),
            MockModel("publishers/google/models/custom-model"),
        ]
        mock_get_client.return_value = mock_client

        translator = GeminiSRTTranslator(
            gemini_api_key="test-key",
            use_enterprise=True,
            cloud_project="test-project",
            cloud_location="us-central1",
        )
        models = translator.getmodels()

        self.assertEqual(models, ["gemini-2.5-flash", "gemini-2.5-pro", "custom-model"])
        mock_client.models.list.assert_called_once()

    @patch.object(GeminiSRTTranslator, "_get_client")
    def test_getmodels_empty_list(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_get_client.return_value = mock_client

        translator = GeminiSRTTranslator(gemini_api_key="test-key")
        models = translator.getmodels()

        self.assertEqual(models, [])


class TestModuleListModels(unittest.TestCase):
    def setUp(self):
        self.orig_skip_upgrade = gst.skip_upgrade
        self.orig_api_key = gst.gemini_api_key
        self.orig_enterprise = gst.use_enterprise
        self.orig_cloud_key = gst.cloud_api_key
        self.orig_cloud_proj = gst.cloud_project
        self.orig_cloud_loc = gst.cloud_location
        gst.skip_upgrade = True

    def tearDown(self):
        gst.skip_upgrade = self.orig_skip_upgrade
        gst.gemini_api_key = self.orig_api_key
        gst.use_enterprise = self.orig_enterprise
        gst.cloud_api_key = self.orig_cloud_key
        gst.cloud_project = self.orig_cloud_proj
        gst.cloud_location = self.orig_cloud_loc

    @patch("gemini_srt_translator.main.GeminiSRTTranslator.getmodels")
    def test_listmodels_prints_available_models(self, mock_getmodels):
        mock_getmodels.return_value = ["gemini-2.5-flash", "gemini-2.5-pro"]
        gst.gemini_api_key = "test-api-key"

        captured = io.StringIO()
        with redirect_stdout(captured):
            gst.listmodels()

        output = captured.getvalue()
        self.assertIn("Available models:", output)
        self.assertIn("gemini-2.5-flash", output)
        self.assertIn("gemini-2.5-pro", output)

    @patch("gemini_srt_translator.main.GeminiSRTTranslator.getmodels")
    def test_listmodels_prints_message_when_no_models_found(self, mock_getmodels):
        mock_getmodels.return_value = []
        gst.gemini_api_key = "test-api-key"

        captured = io.StringIO()
        with redirect_stdout(captured):
            gst.listmodels()

        output = captured.getvalue()
        self.assertIn("No models available or an error occurred while fetching models.", output)

    @patch("gemini_srt_translator.main.GeminiSRTTranslator.__init__", return_value=None)
    @patch("gemini_srt_translator.main.GeminiSRTTranslator.getmodels", return_value=[])
    def test_listmodels_passes_configuration_to_translator(self, mock_getmodels, mock_init):
        gst.gemini_api_key = "test-key"
        gst.use_enterprise = True
        gst.cloud_api_key = "cloud-key"
        gst.cloud_project = "my-proj"
        gst.cloud_location = "europe-west1"

        captured = io.StringIO()
        with redirect_stdout(captured):
            gst.listmodels()

        mock_init.assert_called_once_with(
            gemini_api_key="test-key",
            use_enterprise=True,
            cloud_api_key="cloud-key",
            cloud_project="my-proj",
            cloud_location="europe-west1",
        )

    @patch("gemini_srt_translator.upgrade_package")
    @patch("gemini_srt_translator.main.GeminiSRTTranslator.getmodels", return_value=["gemini-2.5-flash"])
    def test_listmodels_checks_upgrade_when_not_skipped(self, mock_getmodels, mock_upgrade):
        gst.skip_upgrade = False
        captured = io.StringIO()
        with redirect_stdout(captured):
            gst.listmodels()

        mock_upgrade.assert_called_once_with("gemini-srt-translator", use_colors=gst.use_colors)

    @patch("gemini_srt_translator.main.GeminiSRTTranslator.getmodels")
    def test_getmodels_returns_model_list(self, mock_getmodels):
        mock_getmodels.return_value = ["gemini-2.5-flash", "gemini-2.5-pro"]
        gst.gemini_api_key = "test-api-key"

        result = gst.getmodels()
        self.assertEqual(result, ["gemini-2.5-flash", "gemini-2.5-pro"])


if __name__ == "__main__":
    unittest.main()
