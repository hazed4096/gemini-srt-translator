import json
import unittest
from unittest.mock import MagicMock, patch

from gemini_srt_translator.utils import (
    get_installed_version,
    get_latest_pypi_version,
    upgrade_package,
)


class TestUtils(unittest.TestCase):
    @patch("gemini_srt_translator.utils.get_installed_version", return_value="3.8.0b2")
    @patch("gemini_srt_translator.utils.get_latest_pypi_version", return_value="3.8.0")
    @patch("gemini_srt_translator.utils.input_prompt", return_value="n")
    @patch("gemini_srt_translator.utils.info")
    def test_upgrade_package_detects_beta_to_final_upgrade(
        self, mock_info, mock_input, mock_latest, mock_installed
    ):
        upgrade_package("gemini-srt-translator", use_colors=False)
        mock_input.assert_called_once_with(
            "Do you want to upgrade gemini-srt-translator from version 3.8.0b2 to 3.8.0? (y/n): "
        )

    @patch("gemini_srt_translator.utils.get_installed_version", return_value="3.8.0")
    @patch("gemini_srt_translator.utils.get_latest_pypi_version", return_value="3.8.0")
    @patch("gemini_srt_translator.utils.input_prompt")
    def test_upgrade_package_same_version_does_not_prompt(
        self, mock_input, mock_latest, mock_installed
    ):
        upgrade_package("gemini-srt-translator", use_colors=False)
        mock_input.assert_not_called()

    @patch("gemini_srt_translator.utils.get_installed_version", return_value="3.8.1")
    @patch("gemini_srt_translator.utils.get_latest_pypi_version", return_value="3.8.0")
    @patch("gemini_srt_translator.utils.input_prompt")
    def test_upgrade_package_newer_installed_does_not_prompt(
        self, mock_input, mock_latest, mock_installed
    ):
        upgrade_package("gemini-srt-translator", use_colors=False)
        mock_input.assert_not_called()

    @patch("gemini_srt_translator.utils.get_installed_version", return_value=None)
    @patch("gemini_srt_translator.utils.get_latest_pypi_version", return_value="3.8.0")
    @patch("gemini_srt_translator.utils.input_prompt")
    def test_upgrade_package_none_installed_graceful(
        self, mock_input, mock_latest, mock_installed
    ):
        upgrade_package("gemini-srt-translator", use_colors=False)
        mock_input.assert_not_called()

    @patch("gemini_srt_translator.utils.get_installed_version", return_value="3.8.0")
    @patch("gemini_srt_translator.utils.get_latest_pypi_version", return_value=None)
    @patch("gemini_srt_translator.utils.input_prompt")
    def test_upgrade_package_none_latest_graceful(
        self, mock_input, mock_latest, mock_installed
    ):
        upgrade_package("gemini-srt-translator", use_colors=False)
        mock_input.assert_not_called()

    @patch("gemini_srt_translator.utils.urllib.request.urlopen")
    def test_get_latest_pypi_version_ignores_prereleases(self, mock_urlopen):
        pypi_response = {
            "releases": {
                "3.7.0": [],
                "3.8.0a1": [],
                "3.8.0b1": [],
                "3.8.0b2": [],
                "3.8.0rc1": [],
                "3.8.0": [],
                "3.9.0a1": [],
                "3.9.0b1": [],
            }
        }
        mock_cm = MagicMock()
        mock_cm.status = 200
        mock_cm.read.return_value = json.dumps(pypi_response).encode("utf-8")
        mock_cm.__enter__.return_value = mock_cm
        mock_urlopen.return_value = mock_cm

        latest = get_latest_pypi_version("gemini-srt-translator")
        self.assertEqual(latest, "3.8.0")


if __name__ == "__main__":
    unittest.main()
