import hashlib
import importlib
import io
import os
import sys
import tempfile
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import ModuleType
from unittest.mock import patch

from fastapi.testclient import TestClient

import chunk_uploads


class ChunkUploadTests(unittest.TestCase):
    def test_existing_upload_and_resumed_upload_share_transcription_pipeline(self):
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "PCM"))
            wav.writeframes(b"\0\0" * 40000)
        data = audio.getvalue()
        part_bytes = 16384
        parts = [data[offset:offset + part_bytes] for offset in range(0, len(data), part_bytes)]
        expected = hashlib.sha256(data).hexdigest()

        def normalize(source, destination):
            Path(source).replace(destination)
            return destination

        async def transcribe(path, *_args):
            try:
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()
            finally:
                Path(path).unlink()

        model = ModuleType("model_service")
        model.FunASRService = type("FakeService", (), {"load": lambda self: None})
        with tempfile.TemporaryDirectory() as directory:
            env = {"ASR_CHUNK_UPLOAD_ENABLED": "1", "ASR_FAST_ENABLED": "0",
                   "ASR_CHUNK_UPLOAD_DIR": directory, "DATA_DIR": directory,
                   "LEADER_DB_PATH": str(Path(directory) / "leaders.json")}
            modules = {"model_service": model}
            spec = importlib.util.find_spec("main")
            app = importlib.util.module_from_spec(spec)
            with patch.dict(os.environ, env), patch.dict(sys.modules, modules):
                with patch.multiple("config", LEADER_DB_PATH=Path(env["LEADER_DB_PATH"])):
                    spec.loader.exec_module(app)
            self.addCleanup(app.audio_io_executor.shutdown)
            options = {"action": "transcribe", "return_full_text": "true"}
            with TestClient(app.app) as client, patch.object(app, "_normalize_audio_file", normalize), \
                    patch.object(app, "_run_audio_worker", transcribe), patch.object(chunk_uploads, "PART_BYTES", part_bytes):
                def post(action, upload_id=None, **extra):
                    fields = {"action": action}
                    if upload_id:
                        fields["upload_id"] = upload_id
                    return client.post("/transcribe", data={**fields, **extra})

                def send(number, content, upload_id):
                    return client.post("/transcribe", data={"action": "upload_part", "upload_id": upload_id,
                                                           "part_number": number}, files={"file": ("part.bin", content)})

                def initialize(digest):
                    response = post("upload_init", upload_filename="sample.wav", upload_size=len(data), upload_sha256=digest)
                    self.assertEqual(response.status_code, 200, response.text)
                    return response.json()["data"]["upload_id"]

                store = app.chunk_uploads
                with patch.object(app, "chunk_uploads", None):
                    self.assertEqual(post("upload_init").status_code, 503)
                    legacy = client.post("/transcribe", data=options, files={"file": ("sample.wav", data)})
                    self.assertEqual(legacy.status_code, 200, legacy.text)
                self.assertEqual(client.post("/transcribe", data=options, files={"file": ("sample.wav", data)}).json(), legacy.json())

                upload_id = initialize(expected)
                self.assertEqual(post("upload_complete", upload_id).status_code, 409)
                self.assertEqual(send(1, b"partial", upload_id).status_code, 400)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    replies = pool.map(lambda n: send(n, parts[n-1], upload_id), [5, 1, 4, 2, 3, 4, 1])
                    self.assertEqual({reply.status_code for reply in replies}, {200})
                self.assertEqual(send(1, b"X" + parts[0][1:], upload_id).status_code, 409)

                reopened = chunk_uploads.ChunkUploads(store.root, store.max_file_size, store.capacity, store.ttl, store.extensions)
                with patch.object(app, "chunk_uploads", reopened):
                    status = post("upload_status", upload_id).json()["data"]
                    self.assertEqual(status["received_parts"], [1, 2, 3, 4, 5])
                    for _ in range(2):
                        self.assertTrue(post("upload_complete", upload_id).json()["data"]["ready"])
                    self.assertEqual(send(5, parts[4], upload_id).status_code, 200)
                    result = post("transcribe", upload_id, return_full_text="true")
                    self.assertEqual(result.json(), legacy.json())

                corrupt = initialize("0" * 64)
                for number, part in enumerate(parts, 1):
                    self.assertEqual(send(number, part, corrupt).status_code, 200)
                self.assertEqual(post("upload_complete", corrupt).status_code, 422)
                self.assertEqual(post("transcribe", corrupt, return_full_text="true").status_code, 409)

                raw = [{"speaker": "0", "start_time": 0.0, "end_time": 1.0, "text": "示例"}]

                async def speaker_rows(path, *_args):
                    Path(path).unlink()
                    return raw, {"status": "ok"}, []

                release = Event()
                with ThreadPoolExecutor(max_workers=1) as correction_pool, \
                        patch.object(app, "_run_audio_worker", speaker_rows), \
                        patch.object(app, "correction_executor", correction_pool), \
                        patch.object(app, "POSTPROCESS_AUTO_WAIT_SECONDS", 0.02), \
                        patch.object(app.post_processor, "correct_segments", lambda rows: [dict(row, text="已纠错") for row in rows]):
                    blocker = correction_pool.submit(release.wait, 5)
                    task_ids = []
                    try:
                        for fields in ({}, {"correct_text": "true"}, {"correct_text": "false"}):
                            reply = client.post("/transcribe", data=fields, files={"file": ("sample.wav", data)})
                            self.assertEqual(reply.status_code, 200, reply.text)
                            task_id = reply.json().get("correction_task_id")
                            if fields.get("correct_text") == "false":
                                self.assertIsNone(task_id)
                                self.assertEqual(reply.json()["data"], raw)
                            else:
                                self.assertIsNotNone(task_id)
                                self.assertEqual(app.get_correction_task(task_id)["status"], "queued")
                                task_ids.append(task_id)
                    finally:
                        release.set()
                    blocker.result(timeout=2)
                    for task_id in task_ids:
                        app.correction_tasks[task_id]["future"].result(timeout=2)
                        task = post("correction_result", task_id=task_id).json()["data"]
                        self.assertEqual(task["status"], "completed")
                        self.assertEqual(task["result"][0]["text"], "已纠错")


if __name__ == "__main__":
    unittest.main()
