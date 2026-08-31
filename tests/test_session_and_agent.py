import json
import os
import tempfile
import unittest

from pydub import AudioSegment

from gemini_srt_translator.agent_cli import (
    cmd_agent_commit,
    cmd_agent_next,
    cmd_agent_reset,
    cmd_agent_start,
    cmd_agent_status,
)
from gemini_srt_translator.session import SubtitleSession, TranscriptionSession

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:03,000
Hello, world!

2
00:00:04,000 --> 00:00:06,000
How are you today?

3
00:00:07,000 --> 00:00:09,000
This is a test subtitle.

4
00:00:10,000 --> 00:00:12,000
Goodbye!
"""


class TestSubtitleSession(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.srt_path = os.path.join(self.temp_dir.name, "test.srt")
        self.out_path = os.path.join(self.temp_dir.name, "test_translated.srt")
        with open(self.srt_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_SRT)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_session_initialization(self):
        session = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
            output_file=self.out_path,
            batch_size=2,
        )
        self.assertEqual(session.total_lines, 4)
        self.assertEqual(session.current_line, 1)
        self.assertFalse(session.is_complete())

    def test_get_next_batch_and_commit(self):
        session = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
            output_file=self.out_path,
            batch_size=2,
            context_size=2,
        )

        # Batch 1
        batch1 = session.get_next_batch()
        self.assertIsNotNone(batch1)
        self.assertEqual(batch1["start_line"], 1)
        self.assertEqual(batch1["end_line"], 2)
        self.assertEqual(len(batch1["batch"]), 2)
        self.assertEqual(batch1["batch"][0]["text"], "Hello, world!")
        self.assertNotIn("original_context", batch1)
        self.assertNotIn("translated_context", batch1)

        # Commit Batch 1
        trans1 = [
            {"index": "0", "text": "Bonjour le monde !"},
            {"index": "1", "text": "Comment allez-vous aujourd'hui ?"},
        ]
        res1 = session.commit_batch(trans1)
        self.assertTrue(res1["success"])
        self.assertFalse(res1["is_complete"])
        self.assertEqual(session.current_line, 3)

        # Batch 2
        batch2 = session.get_next_batch()
        self.assertIsNotNone(batch2)
        self.assertEqual(batch2["start_line"], 3)
        self.assertEqual(batch2["end_line"], 4)
        self.assertEqual(len(batch2["batch"]), 2)
        self.assertEqual(batch2["batch"][0]["text"], "This is a test subtitle.")
        self.assertEqual(len(batch2["original_context"]), 2)
        self.assertEqual(batch2["original_context"][0]["text"], "Hello, world!")
        self.assertEqual(len(batch2["translated_context"]), 2)
        self.assertEqual(batch2["translated_context"][0]["text"], "Bonjour le monde !")

        # Commit Batch 2
        trans2 = json.dumps(
            [
                {"index": "2", "text": "Ceci est un sous-titre de test."},
                {"index": "3", "text": "Au revoir !"},
            ]
        )
        res2 = session.commit_batch(trans2)
        self.assertTrue(res2["success"])
        self.assertTrue(res2["is_complete"])
        self.assertTrue(session.is_complete())
        self.assertTrue(os.path.exists(self.out_path))

    def test_commit_item_count_validation(self):
        session = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
            output_file=self.out_path,
            batch_size=2,
        )
        # Pass 1 item when 2 are expected
        bad_trans = [{"index": "0", "text": "Only one item"}]
        res = session.commit_batch(bad_trans)
        self.assertFalse(res["success"])
        self.assertIn("Item count mismatch", res["error"])

    def test_resume_progress(self):
        session1 = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
            output_file=self.out_path,
            batch_size=2,
        )
        session1.commit_batch(
            [
                {"index": "0", "text": "Bonjour"},
                {"index": "1", "text": "Comment allez-vous"},
            ]
        )

        # Create new session loading the same file
        session2 = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
            output_file=self.out_path,
            batch_size=2,
            resume=True,
        )
        self.assertEqual(session2.current_line, 3)
        self.assertEqual(session2.get_status()["completed_lines"], 2)

    def test_progress_file_created_on_init(self):
        progress_path = os.path.join(self.temp_dir.name, "test.progress")
        session = SubtitleSession(
            input_file=self.srt_path,
            target_language="Portuguese (Brazil)",
            output_file=self.out_path,
            batch_size=2,
            description="Test Series",
        )
        self.assertTrue(os.path.exists(progress_path))
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["line"], 1)
        self.assertEqual(data["target_language"], "Portuguese (Brazil)")
        self.assertEqual(data["batch_size"], 2)
        self.assertEqual(data["description"], "Test Series")

    def test_metadata_restoration_across_sessions(self):
        # Initialize session with custom parameters
        session1 = SubtitleSession(
            input_file=self.srt_path,
            target_language="Portuguese (Brazil)",
            output_file=self.out_path,
            batch_size=2,
            context_size=1,
            description="Test Notes",
        )
        # Commit batch 1
        trans1 = [
            {"index": "0", "text": "Ola mundo"},
            {"index": "1", "text": "Como vai voce"},
        ]
        res1 = session1.commit_batch(trans1)
        self.assertTrue(res1["success"])

        # Simulate separate CLI call for commit / next / status without passing -l, -b
        session2 = SubtitleSession(
            input_file=self.srt_path,
            resume=True,
        )
        self.assertEqual(session2.current_line, 3)
        self.assertEqual(session2.target_language, "Portuguese (Brazil)")
        self.assertEqual(session2.batch_size, 2)
        self.assertEqual(session2.context_size, 1)
        self.assertEqual(session2.description, "Test Notes")

        # Verify strict validation uses restored batch_size=2
        # Passing 1 item fails
        res_fail = session2.commit_batch([{"index": "2", "text": "So um item"}])
        self.assertFalse(res_fail["success"])
        self.assertIn("expected 2 items", res_fail["error"])

        # Passing 2 items succeeds
        res_pass = session2.commit_batch(
            [
                {"index": "2", "text": "Isto e um teste"},
                {"index": "3", "text": "Adeus"},
            ]
        )
        self.assertTrue(res_pass["success"])
        self.assertTrue(session2.is_complete())
        progress_path = os.path.join(self.temp_dir.name, "test.progress")
        self.assertFalse(os.path.exists(progress_path))

    def test_no_resume_cleans_progress_file_and_resets(self):
        # Create initial session and advance
        session1 = SubtitleSession(
            input_file=self.srt_path,
            target_language="Portuguese (Brazil)",
            output_file=self.out_path,
            batch_size=2,
        )
        session1.commit_batch(
            [
                {"index": "0", "text": "Ola"},
                {"index": "1", "text": "Mundo"},
            ]
        )
        progress_path = os.path.join(self.temp_dir.name, "test.progress")
        self.assertTrue(os.path.exists(progress_path))

        # Start new session with resume=False
        session2 = SubtitleSession(
            input_file=self.srt_path,
            target_language="Spanish",
            output_file=self.out_path,
            batch_size=3,
            resume=False,
        )
        self.assertEqual(session2.current_line, 1)
        self.assertEqual(session2.target_language, "Spanish")
        self.assertEqual(session2.batch_size, 3)
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["line"], 1)
        self.assertEqual(data["target_language"], "Spanish")
        self.assertEqual(data["batch_size"], 3)

    def test_default_output_file_naming(self):
        # Standalone subtitle: should append _translated
        session_sub = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
        )
        self.assertEqual(session_sub.output_file, os.path.join(self.temp_dir.name, "test_translated.srt"))

        # Video file + extracted subtitle: should NOT append _translated, matching video basename
        video_path = os.path.join(self.temp_dir.name, "movie.mkv")
        session_vid = SubtitleSession(
            input_file=self.srt_path,
            video_file=video_path,
            target_language="French",
        )
        self.assertEqual(session_vid.output_file, os.path.join(self.temp_dir.name, "movie.srt"))

        from gemini_srt_translator.main import GeminiSRTTranslator
        t_sub = GeminiSRTTranslator(input_file=self.srt_path, target_language="French")
        self.assertEqual(t_sub.output_file, os.path.join(self.temp_dir.name, "test_translated.srt"))

        t_vid = GeminiSRTTranslator(video_file=video_path, target_language="French")
        self.assertEqual(t_vid.output_file, os.path.join(self.temp_dir.name, "movie.srt"))

    def test_agent_cli_defaults_context_size_to_zero(self):
        from gemini_srt_translator.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["agent", "start", self.srt_path, "-l", "French"])
        self.assertEqual(args.context_size, 0)

    def test_agent_cli_command_workflow_persists_settings(self):
        from gemini_srt_translator.cli import create_parser

        parser = create_parser()

        # Step 1: gst agent start with custom language and batch size 2
        start_args = parser.parse_args(["agent", "start", self.srt_path, "-l", "Portuguese (Brazil)", "-b", "2"])
        code = cmd_agent_start(start_args)
        self.assertEqual(code, 0)

        # Verify progress file exists immediately on disk
        progress_path = os.path.join(self.temp_dir.name, "test.progress")
        self.assertTrue(os.path.exists(progress_path))

        # Step 2: gst agent commit with --data-file without re-passing -l or -b
        batch_file = os.path.join(self.temp_dir.name, "step2_batch.json")
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump([
                {"index": "0", "text": "Ola mundo"},
                {"index": "1", "text": "Como vai voce"},
            ], f)
        commit_args = parser.parse_args(["agent", "commit", self.srt_path, "--data-file", batch_file])
        code = cmd_agent_commit(commit_args)
        self.assertEqual(code, 0)

        # Step 3: gst agent status without passing -l or -b
        status_args = parser.parse_args(["agent", "status", self.srt_path])
        code = cmd_agent_status(status_args)
        self.assertEqual(code, 0)

    def test_agent_cli_commit_with_data_file(self):
        from gemini_srt_translator.cli import create_parser

        parser = create_parser()
        start_args = parser.parse_args(["agent", "start", self.srt_path, "-l", "French", "-b", "2"])
        self.assertEqual(cmd_agent_start(start_args), 0)

        # Write data to a temp file
        batch_file = os.path.join(self.temp_dir.name, "batch_1.json")
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump([
                {"index": "0", "text": "Bonjour"},
                {"index": "1", "text": "Comment allez-vous"},
            ], f)

        commit_args = parser.parse_args(["agent", "commit", self.srt_path, "--data-file", batch_file])
        code = cmd_agent_commit(commit_args)
        self.assertEqual(code, 0)

    def test_agent_cli_commit_without_data_does_not_hang(self):
        import time
        from gemini_srt_translator.cli import create_parser

        parser = create_parser()
        start_args = parser.parse_args(["agent", "start", self.srt_path, "-l", "French"])
        self.assertEqual(cmd_agent_start(start_args), 0)

        # Commit without --data or --data-file: should quickly return error code 1 without hanging
        t0 = time.time()
        commit_args = parser.parse_args(["agent", "commit", self.srt_path])
        code = cmd_agent_commit(commit_args)
        t1 = time.time()

        self.assertEqual(code, 1)
        self.assertLess(t1 - t0, 2.0)

    def test_get_next_batch_include_system_prompt_flag(self):
        session = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
            output_file=self.out_path,
            batch_size=2,
        )
        batch_with_prompt = session.get_next_batch(include_system_prompt=True)
        self.assertIn("system_prompt", batch_with_prompt)
        self.assertIsNotNone(batch_with_prompt["system_prompt"])

        batch_without_prompt = session.get_next_batch(include_system_prompt=False)
        self.assertNotIn("system_prompt", batch_without_prompt)

    def test_agent_cli_commit_returns_streamlined_json(self):
        import io
        from unittest.mock import patch
        from gemini_srt_translator.cli import create_parser

        parser = create_parser()

        # 1. Start session
        start_args = parser.parse_args(["agent", "start", self.srt_path, "-l", "French", "-b", "2"])
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = cmd_agent_start(start_args)
            self.assertEqual(code, 0)
            start_data = json.loads(fake_out.getvalue().strip())
            self.assertEqual(start_data["status"], "ready")
            self.assertIn("session", start_data)
            self.assertIn("system_prompt", start_data)
            self.assertEqual(start_data["next_batch"]["start_line"], 1)
            self.assertEqual(start_data["next_batch"]["end_line"], 2)

        # 2. Commit batch 1
        batch1_file = os.path.join(self.temp_dir.name, "b1.json")
        with open(batch1_file, "w", encoding="utf-8") as f:
            json.dump([
                {"index": "0", "text": "Bonjour"},
                {"index": "1", "text": "Comment allez-vous"},
            ], f)

        commit1_args = parser.parse_args(["agent", "commit", self.srt_path, "--data-file", batch1_file])
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = cmd_agent_commit(commit1_args)
            self.assertEqual(code, 0)
            commit1_data = json.loads(fake_out.getvalue().strip())

            # Verify streamlined structure
            self.assertEqual(commit1_data["status"], "committed")
            self.assertIn("progress", commit1_data)
            self.assertEqual(commit1_data["progress"]["batch"], 2)
            self.assertEqual(commit1_data["progress"]["completed_lines"], 2)
            self.assertEqual(commit1_data["progress"]["total_lines"], 4)
            self.assertEqual(commit1_data["progress"]["percent"], 50.0)

            self.assertIn("next_batch", commit1_data)
            # Verify system_prompt is omitted on commit
            self.assertNotIn("system_prompt", commit1_data["next_batch"])
            # Verify duplicate metadata is not present
            self.assertNotIn("commit_result", commit1_data)
            self.assertNotIn("session", commit1_data)

        # 3. Commit batch 2 (final)
        batch2_file = os.path.join(self.temp_dir.name, "b2.json")
        with open(batch2_file, "w", encoding="utf-8") as f:
            json.dump([
                {"index": "2", "text": "Test"},
                {"index": "3", "text": "Au revoir"},
            ], f)

        commit2_args = parser.parse_args(["agent", "commit", self.srt_path, "--data-file", batch2_file])
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = cmd_agent_commit(commit2_args)
            self.assertEqual(code, 0)
            commit2_data = json.loads(fake_out.getvalue().strip())

            # Verify completion structure
            self.assertEqual(commit2_data["status"], "completed")
            self.assertEqual(commit2_data["progress"]["completed_lines"], 4)
            self.assertEqual(commit2_data["progress"]["percent"], 100.0)
            self.assertEqual(commit2_data["output_file"], self.out_path)
            self.assertNotIn("next_batch", commit2_data)


class TestTranscriptionSession(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mp3_path = os.path.join(self.temp_dir.name, "test_audio.mp3")
        self.out_path = os.path.join(self.temp_dir.name, "test_transcribed.srt")

        # Generate a 20-second silent mp3 for testing
        seg = AudioSegment.silent(duration=20000)
        seg.export(self.mp3_path, format="mp3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_transcription_initialization(self):
        session = TranscriptionSession(
            audio_file=self.mp3_path,
            output_file=self.out_path,
            audio_chunk_size=10,
        )
        self.assertEqual(session.total_seconds, 20)
        self.assertEqual(session.current_seconds, 0)
        self.assertFalse(session.is_complete())

    def test_transcription_chunk_and_commit(self):
        session = TranscriptionSession(
            audio_file=self.mp3_path,
            output_file=self.out_path,
            audio_chunk_size=10,
        )

        # Chunk 1 (0 to 10s)
        chunk1 = session.get_next_chunk()
        self.assertIsNotNone(chunk1)
        self.assertEqual(chunk1["start_seconds"], 0)
        self.assertEqual(chunk1["end_seconds"], 10)
        self.assertIsNotNone(chunk1["audio_bytes"])

        # Commit chunk 1
        items1 = [
            {"text": "Hello world from 0 to 4s", "time_start": "00:00", "time_end": "00:04"},
            {"text": "Next sentence from 5 to 9s", "time_start": "00:05", "time_end": "00:09"},
        ]
        res1 = session.commit_chunk(items1)
        self.assertTrue(res1["success"])
        self.assertEqual(res1["added_subtitles"], 2)
        self.assertEqual(session.current_seconds, 10)
        self.assertFalse(session.is_complete())
        self.assertTrue(os.path.exists(self.out_path))

        # Chunk 2 (10 to 20s)
        chunk2 = session.get_next_chunk()
        self.assertIsNotNone(chunk2)
        self.assertEqual(chunk2["start_seconds"], 10)
        self.assertEqual(chunk2["end_seconds"], 20)

        # Commit chunk 2
        items2 = [
            {"text": "Ending part from 11 to 15s", "time_start": "00:01", "time_end": "00:05"},
        ]
        res2 = session.commit_chunk(items2)
        self.assertTrue(res2["success"])
        self.assertTrue(res2["is_complete"])
        self.assertTrue(session.is_complete())
        self.assertEqual(len(session.transcribed_subtitles), 3)

    def test_transcription_default_output_name(self):
        session = TranscriptionSession(audio_file=self.mp3_path)
        expected_name = os.path.join(self.temp_dir.name, "test_audio.srt")
        self.assertEqual(session.output_file, expected_name)

    def test_transcription_get_next_chunk_include_system_prompt_flag(self):
        session = TranscriptionSession(
            audio_file=self.mp3_path,
            output_file=self.out_path,
            audio_chunk_size=10,
        )
        chunk_with_prompt = session.get_next_chunk(include_system_prompt=True)
        self.assertIn("system_prompt", chunk_with_prompt)
        self.assertIsNotNone(chunk_with_prompt["system_prompt"])

        chunk_without_prompt = session.get_next_chunk(include_system_prompt=False)
        self.assertNotIn("system_prompt", chunk_without_prompt)

    def test_transcription_cleanup_extracted_mp3_and_chunks(self):
        # Create a mock extracted mp3
        extracted_mp3 = os.path.join(self.temp_dir.name, "video_extracted.mp3")
        seg = AudioSegment.silent(duration=10000)
        seg.export(extracted_mp3, format="mp3")

        session = TranscriptionSession(audio_file=extracted_mp3, video_file="video.mp4")
        self.assertTrue(session.audio_extracted)
        chunk = session.get_next_chunk()
        self.assertIsNotNone(chunk)
        chunk_file = chunk["audio_chunk_path"]
        self.assertTrue(os.path.exists(chunk_file))
        self.assertTrue(os.path.exists(extracted_mp3))

        session.cleanup()
        self.assertFalse(os.path.exists(extracted_mp3))
        self.assertFalse(os.path.exists(chunk_file))


class TestSkillManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_skill_content(self):
        from gemini_srt_translator.skill import get_skill_content

        content = get_skill_content()
        self.assertIn("subtitle-translator", content)
        self.assertIn("gst agent start", content)

    def test_install_skill_default(self):
        from gemini_srt_translator.skill import install_skill

        installed = install_skill(is_global=False, cwd=self.temp_dir.name)
        self.assertEqual(len(installed), 1)
        expected_file = os.path.join(self.temp_dir.name, ".agents", "skills", "subtitle-translator", "SKILL.md")
        self.assertTrue(os.path.exists(expected_file))
        self.assertEqual(installed[0], expected_file)

    def test_install_skill_antigravity(self):
        from gemini_srt_translator.skill import install_skill

        installed = install_skill(target="antigravity", is_global=False, cwd=self.temp_dir.name)
        self.assertEqual(len(installed), 1)
        expected_file = os.path.join(self.temp_dir.name, ".gemini", "skills", "subtitle-translator", "SKILL.md")
        self.assertTrue(os.path.exists(expected_file))
        self.assertEqual(installed[0], expected_file)

    def test_install_skill_all_targets(self):
        from gemini_srt_translator.skill import install_skill

        installed = install_skill(target="all", is_global=False, cwd=self.temp_dir.name)
        self.assertGreaterEqual(len(installed), 3)
        for p in installed:
            self.assertTrue(os.path.exists(p))


if __name__ == "__main__":
    unittest.main()
