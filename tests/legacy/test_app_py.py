from __future__ import annotations

import io
import sys
from pathlib import Path
import unittest
import wave


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import clean_model_text, ready_sentences, wav_bytes


class CoreTests(unittest.TestCase):
    def test_removes_reasoning_blocks(self) -> None:
        self.assertEqual(clean_model_text("<think>hidden</think>你好"), "你好")

    def test_splits_complete_sentences(self) -> None:
        complete, pending = ready_sentences("第一句话。第二句还没结束")
        self.assertEqual(complete, ["第一句话。"])
        self.assertEqual(pending, "第二句还没结束")

    def test_wav_wrapper(self) -> None:
        data = wav_bytes(b"\x00\x00" * 160, 16000)
        with wave.open(io.BytesIO(data), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 16000)


if __name__ == "__main__":
    unittest.main()
