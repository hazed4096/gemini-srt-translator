import unittest
from unittest.mock import MagicMock, patch

from gemini_srt_translator.logger import set_quiet_mode
from gemini_srt_translator.main import GeminiSRTTranslator


class TestAdaptiveBatchSize(unittest.TestCase):
    def setUp(self):
        set_quiet_mode(True)

    def tearDown(self):
        set_quiet_mode(False)

    def test_default_and_custom_error_steps(self):
        # Default steps
        t1 = GeminiSRTTranslator(gemini_api_key="fake_key", batch_size=1000, audio_chunk_size=300)
        self.assertEqual(t1.batch_size_error_step, 100)
        self.assertEqual(t1.audio_chunk_error_step, 60)
        self.assertEqual(t1._original_batch_size, 1000)
        self.assertEqual(t1._original_audio_chunk_size, 300)

        # Custom steps
        t2 = GeminiSRTTranslator(
            gemini_api_key="fake_key",
            batch_size=500,
            audio_chunk_size=240,
            batch_size_error_step=50,
            audio_chunk_error_step=30,
        )
        self.assertEqual(t2.batch_size_error_step, 50)
        self.assertEqual(t2.audio_chunk_error_step, 30)
        self.assertEqual(t2._original_batch_size, 500)
        self.assertEqual(t2._original_audio_chunk_size, 240)

    def test_reduce_and_restore_batch_size_no_audio(self):
        t = GeminiSRTTranslator(gemini_api_key="fake_key", batch_size=1000, batch_size_error_step=100)
        self.assertEqual(t.batch_size, 1000)

        # 1st error
        t.consecutive_error_count = 1
        t._reduce_batch_size()
        self.assertEqual(t.batch_size, 900)

        # 2nd error
        t.consecutive_error_count = 2
        t._reduce_batch_size()
        self.assertEqual(t.batch_size, 800)

        # Multiple errors down to floor (100)
        t.consecutive_error_count = 15
        t._reduce_batch_size()
        self.assertEqual(t.batch_size, 100)

        # Restore on success
        t._restore_batch_size()
        self.assertEqual(t.batch_size, 1000)

    def test_reduce_and_restore_batch_size_with_audio(self):
        t = GeminiSRTTranslator(
            gemini_api_key="fake_key",
            batch_size=500,
            audio_file="mock_audio.mp3",
            audio_chunk_size=300,
            batch_size_error_step=100,
            audio_chunk_error_step=60,
        )
        mock_session = MagicMock()
        mock_session.audio_chunk_size = 300
        t.session = mock_session

        # 1st error
        t.consecutive_error_count = 1
        t._reduce_batch_size()
        self.assertEqual(t.batch_size, 400)
        self.assertEqual(t.audio_chunk_size, 240)
        self.assertEqual(mock_session.audio_chunk_size, 240)

        # 2nd error
        t.consecutive_error_count = 2
        t._reduce_batch_size()
        self.assertEqual(t.batch_size, 300)
        self.assertEqual(t.audio_chunk_size, 180)
        self.assertEqual(mock_session.audio_chunk_size, 180)

        # Error down to floor (60s)
        t.consecutive_error_count = 10
        t._reduce_batch_size()
        self.assertEqual(t.batch_size, 100)
        self.assertEqual(t.audio_chunk_size, 60)
        self.assertEqual(mock_session.audio_chunk_size, 60)

        # Restore on success
        t._restore_batch_size()
        self.assertEqual(t.batch_size, 500)
        self.assertEqual(t.audio_chunk_size, 300)
        self.assertEqual(mock_session.audio_chunk_size, 300)

    def test_reduce_and_restore_transcribe_chunk_size(self):
        t = GeminiSRTTranslator(
            gemini_api_key="fake_key",
            audio_file="mock_audio.mp3",
            audio_chunk_size=600,
            audio_chunk_error_step=60,
        )
        mock_session = MagicMock()
        mock_session.audio_chunk_size = 600
        t.transcription_session = mock_session
        t._original_audio_chunk_size = 600

        # 1st error
        t._reduce_transcribe_chunk_size(error_count=1)
        self.assertEqual(mock_session.audio_chunk_size, 540)
        self.assertEqual(t.audio_chunk_size, 540)

        # 2nd error
        t._reduce_transcribe_chunk_size(error_count=2)
        self.assertEqual(mock_session.audio_chunk_size, 480)
        self.assertEqual(t.audio_chunk_size, 480)

        # Many errors down to floor (60s)
        t._reduce_transcribe_chunk_size(error_count=12)
        self.assertEqual(mock_session.audio_chunk_size, 60)
        self.assertEqual(t.audio_chunk_size, 60)

        # Restore on success
        t._restore_transcribe_chunk_size()
        self.assertEqual(mock_session.audio_chunk_size, 600)
        self.assertEqual(t.audio_chunk_size, 600)

    def test_small_initial_batch_floor(self):
        # Initial batch size smaller than 100
        t = GeminiSRTTranslator(gemini_api_key="fake_key", batch_size=50, batch_size_error_step=100)
        t._original_batch_size = 50
        t.consecutive_error_count = 1
        t._reduce_batch_size()
        # Should not go below initial 50
        self.assertEqual(t.batch_size, 50)

    def test_quota_error_does_not_reduce_batch_size(self):
        t = GeminiSRTTranslator(gemini_api_key="fake_key", batch_size=1000, batch_size_error_step=100)
        t._original_batch_size = 1000
        t.consecutive_error_count = 1
        # In translate(), quota branch does NOT call _reduce_batch_size
        self.assertEqual(t.batch_size, 1000)


if __name__ == "__main__":
    unittest.main()
