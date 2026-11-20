from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext, suppress
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

import config
from chunk_uploads import ChunkUploads, UploadError
try:
    from leader_store import LeaderAlreadyExistsError, LeaderStore, cosine_similarity
except ImportError:
    from leader_store import LeaderStore, cosine_similarity

    LeaderAlreadyExistsError = ValueError
from model_service import FunASRService
from post_processor import TranscriptPostProcessor


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac"}
DEFAULT_ALLOWED_AUDIO_FORMATS = ["WAV", "MP3", "M4A", "AAC"]

CHUNK_UPLOAD_ENABLED = os.getenv("ASR_CHUNK_UPLOAD_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
FAST_ASR_ENABLED = os.getenv("ASR_FAST_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
POSTPROCESS_AUTO_WAIT_SECONDS = max(0.0, float(os.getenv(
    "ASR_POSTPROCESS_AUTO_WAIT_SECONDS", str(config.POSTPROCESS_SYNC_TIMEOUT_SECONDS),
)))
fast_runtime = None
chunk_uploads = None
if CHUNK_UPLOAD_ENABLED:
    chunk_uploads = ChunkUploads(
        root=Path(os.getenv("ASR_CHUNK_UPLOAD_DIR", str(Path(config.LEADER_DB_PATH).parent / "chunk_uploads"))),
        max_file_size=config.MAX_FILE_SIZE,
        capacity=int(os.getenv("ASR_CHUNK_UPLOAD_CAPACITY_BYTES", str(4 * config.MAX_FILE_SIZE))),
        ttl=int(os.getenv("ASR_CHUNK_UPLOAD_TTL_SECONDS", "86400")),
        extensions=getattr(config, "ALLOWED_AUDIO_EXTENSIONS", DEFAULT_ALLOWED_AUDIO_EXTENSIONS),
    )

executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
correction_executor = ThreadPoolExecutor(max_workers=config.CORRECTION_TASK_WORKERS)
correction_tasks: dict[str, dict[str, Any]] = {}
correction_tasks_lock = threading.Lock()
asr_model_lock = threading.RLock()
asr_service = FunASRService()
leader_store = LeaderStore(config.LEADER_DB_PATH)
post_processor = TranscriptPostProcessor()

PENDING_VOICEPRINT_DB_PATH = Path(
    os.getenv(
        "PENDING_VOICEPRINT_DB_PATH",
        str(Path(getattr(config, "DATA_DIR", Path(config.LEADER_DB_PATH).parent)) / "pending_voiceprints.json"),
    )
).resolve()
PENDING_VOICEPRINT_TTL_DAYS = max(1, int(os.getenv("PENDING_VOICEPRINT_TTL_DAYS", "30")))
PENDING_VOICEPRINT_MATCH_THRESHOLD = float(os.getenv("PENDING_VOICEPRINT_MATCH_THRESHOLD", "0.70"))
PENDING_VOICEPRINT_SAMPLE_DEDUPE_THRESHOLD = float(
    os.getenv("PENDING_VOICEPRINT_SAMPLE_DEDUPE_THRESHOLD", "0.97")
)
PENDING_VOICEPRINT_MIN_SAMPLES = 3
PENDING_VOICEPRINT_MAX_SAMPLES = min(
    5,
    max(PENDING_VOICEPRINT_MIN_SAMPLES, int(os.getenv("PENDING_VOICEPRINT_MAX_SAMPLES", "5"))),
)
PENDING_VOICEPRINT_MIN_SAMPLE_SECONDS = max(
    5.0,
    float(os.getenv("PENDING_VOICEPRINT_MIN_SAMPLE_SECONDS", "5.0")),
)
PENDING_VOICEPRINT_MIN_TOTAL_SPEECH_SECONDS = max(
    PENDING_VOICEPRINT_MIN_SAMPLES * PENDING_VOICEPRINT_MIN_SAMPLE_SECONDS,
    float(os.getenv("PENDING_VOICEPRINT_MIN_TOTAL_SPEECH_SECONDS", "15.0")),
)
PENDING_VOICEPRINT_MIN_SAMPLE_SIMILARITY = float(
    os.getenv(
        "PENDING_VOICEPRINT_MIN_SAMPLE_SIMILARITY",
        str(max(float(getattr(config, "LEADER_ENROLLMENT_MIN_SIMILARITY", 0.60)), 0.68)),
    )
)
PENDING_VOICEPRINT_MIN_ACTIVE_RATIO = min(
    1.0,
    max(0.0, float(os.getenv("PENDING_VOICEPRINT_MIN_ACTIVE_RATIO", "0.70"))),
)
PENDING_VOICEPRINT_MIN_TEXT_CHARS_PER_SECOND = max(
    0.0,
    float(os.getenv("PENDING_VOICEPRINT_MIN_TEXT_CHARS_PER_SECOND", "0.50")),
)
pending_voiceprints_lock = threading.RLock()


import bisect
import sys
import traceback
from contextlib import contextmanager


ASR_MAIN_BUILD = "large-audio-20261112-v2"


audio_io_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr-audio-io")


def _progress_logger() -> logging.Logger:
    progress_logger = logging.getLogger("asr.progress")
    progress_logger.disabled = False
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False
    if not progress_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        progress_logger.addHandler(handler)
    return progress_logger


class TranscriptionProgress:
    """Observe slow workers without cancelling an in-flight model call."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.logger = _progress_logger()
        self.started = time.monotonic()
        self.state_lock = threading.Lock()
        self.state = ("queued", self.started, None)
        self.stopped = threading.Event()
        self.monitor = threading.Thread(target=self._watch, name=f"asr-watch-{request_id}", daemon=True)

    def start(self) -> None:
        self.logger.info("ASR request=%s build=%s stage=queued", self.request_id, ASR_MAIN_BUILD)
        self.monitor.start()

    @contextmanager
    def stage(self, name: str):
        started = time.monotonic()
        with self.state_lock:
            previous = self.state
            self.state = (name, started, threading.get_ident())
        self.logger.info("ASR request=%s stage=%s started", self.request_id, name)
        try:
            yield
        except BaseException:
            self.logger.exception("ASR request=%s stage=%s failed", self.request_id, name)
            raise
        finally:
            self.logger.info("ASR request=%s stage=%s elapsed=%.3fs", self.request_id, name, time.monotonic() - started)
            with self.state_lock:
                self.state = previous

    def _watch(self) -> None:
        while not self.stopped.wait(60):
            with self.state_lock:
                name, started, thread_id = self.state
            elapsed = time.monotonic() - started
            if elapsed < 60:
                continue
            frame = sys._current_frames().get(thread_id) if thread_id is not None else None
            stack = "".join(traceback.format_stack(frame)) if frame is not None else "worker not started"
            self.logger.warning("ASR request=%s stage=%s waiting=%.1fs total=%.1fs\n%s", self.request_id, name, elapsed, time.monotonic() - self.started, stack)
            del frame

    def stop(self) -> None:
        self.stopped.set()


def _remove_audio_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


async def _run_audio_worker(path: str, progress: TranscriptionProgress, function, *args):
    # A cancelled HTTP request must not delete audio still being read by a worker.
    try:
        future = asyncio.get_running_loop().run_in_executor(executor, _transcription_worker, path, progress, function, *args)
    except BaseException:
        _remove_audio_file(path)
        progress.stop()
        raise
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        def completed(task):
            try:
                task.result()
            except BaseException:
                logger.exception("Detached ASR worker failed after request cancellation")
        future.add_done_callback(completed)
        raise


def _transcription_worker(path: str, progress: TranscriptionProgress, function, *args):
    try:
        with progress.stage("model_lock"):
            asr_model_lock.acquire()
        try:
            return function(path, progress, *args)
        finally:
            asr_model_lock.release()
    finally:
        try:
            with progress.stage("audio_cleanup"):
                _remove_audio_file(path)
        finally:
            progress.stop()


def _overlap_free_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Prefix maxima for two distinct speakers preserve the strict overlap rule.
    intervals = []
    for segment in segments:
        start = float(segment.get("start_time", 0))
        end = float(segment.get("end_time", 0))
        if not math.isnan(start) and not math.isnan(end):
            intervals.append((start, end, str(segment.get("speaker", ""))))
    intervals.sort(key=lambda item: item[0])
    starts = [item[0] for item in intervals]
    prefixes = []
    best: list[tuple[float, str]] = []
    for _, end, speaker in intervals:
        previous = next((value for value, owner in best if owner == speaker), -math.inf)
        best = [(value, owner) for value, owner in best if owner != speaker]
        best.append((max(previous, end), speaker))
        best.sort(reverse=True)
        best = best[:2]
        prefixes.append(tuple(best))
    clean = []
    for segment in segments:
        start = float(segment.get("start_time", 0))
        end = float(segment.get("end_time", 0))
        speaker = str(segment.get("speaker", "")).strip()
        if math.isnan(start) or math.isnan(end):
            clean.append(segment)
            continue
        index = bisect.bisect_left(starts, end) - 1
        if index < 0 or not any(value > start and owner != speaker for value, owner in prefixes[index]):
            clean.append(segment)
    return clean


def _leader_exists_compatible(leader_id: str) -> bool:
    """Support both the current LeaderStore API and the production old base."""
    exists = getattr(leader_store, "exists", None)
    if callable(exists):
        return bool(exists(leader_id))
    read_store = getattr(leader_store, "_read", None)
    if callable(read_store):
        data = read_store()
        return leader_id in data.get("leaders", {})
    list_leaders = getattr(leader_store, "list", None)
    if callable(list_leaders):
        return any(str(item.get("leader_id")) == leader_id for item in list_leaders())
    return False


def _plain_transcription_pipeline(path: str, progress: TranscriptionProgress, hotwords: str | None) -> str:
    with progress.stage("plain_transcription"):
        return transcribe_plain_text_compatible(path, hotwords)


def _transcription_pipeline(
    path: str,
    progress: TranscriptionProgress,
    num_speakers: int | None,
    hotwords: str | None,
    identify_leaders: bool,
    leader_threshold: float,
    return_leader_scores: bool,
    collect_unknown_voiceprints: bool,
    source_name: str | None,
    meeting_id: str | None,
    early_correction=None,
):
    audio_duration = audio_duration_seconds(path)
    with progress.stage("transcribe_diarization"):
        result = transcribe_compatible(path, num_speakers, hotwords)
    if identify_leaders:
        if early_correction is not None and leader_store.has_samples():
            early_correction.start(result, merge_adjacent_segments_compatible)
        with progress.stage("identify_leaders"):
            result = annotate_leaders_compatible(path, result, leader_threshold, return_leader_scores)
    with progress.stage("diarization_quality"):
        result = merge_adjacent_segments_compatible(result)
        asr_quality = transcript_quality_compatible(result, audio_duration)
        should_retry = should_retry_diarization_compatible(asr_quality, num_speakers, audio_duration)
    if should_retry:
        initial_quality = asr_quality
        fallback_speakers = config.ASR_FALLBACK_SPEAKER_NUM
        logger.warning("Retrying ASR with fallback speaker count %s after diarization warning: %s", fallback_speakers, initial_quality)
        with progress.stage("retry_transcribe_diarization"):
            result = transcribe_compatible(path, fallback_speakers, hotwords)
        if identify_leaders:
            with progress.stage("retry_identify_leaders"):
                result = annotate_leaders_compatible(path, result, leader_threshold, return_leader_scores)
        result = merge_adjacent_segments_compatible(result)
        asr_quality = transcript_quality_compatible(result, audio_duration)
        asr_quality["retry"] = {
            "reason": "bad_diarization", "fallback_speaker_num": fallback_speakers,
            "initial_quality": initial_quality,
        }
    if getattr(config, "ASR_RESCUE_ENABLED", False) and callable(getattr(asr_service, "rescue_segments", None)):
        with progress.stage("rescue_segments"):
            result, rescue_summary = rescue_segments_compatible(path, result, hotwords)
        asr_quality["rescue"] = rescue_summary
    if asr_quality.get("status") != "ok":
        logger.warning("ASR quality warnings: %s", asr_quality)
    unknown_speakers: list[dict[str, Any]] = []
    if collect_unknown_voiceprints:
        with progress.stage("collect_unknown_voiceprints"):
            unknown_speakers = collect_unknown_voiceprints_compatible(path, result, source_name, meeting_id, leader_threshold)
    progress.logger.info("ASR request=%s pipeline_completed segments=%s unknown_speakers=%s total=%.3fs", progress.request_id, len(result), len(unknown_speakers), time.monotonic() - progress.started)
    return result, asr_quality, unknown_speakers


def _effective_speech_seconds_legacy(path: str) -> float:
    chunk_ms = 20
    active_chunks = 0
    with wave.open(path, "rb") as audio:
        sample_rate = audio.getframerate()
        frames_per_chunk = max(1, sample_rate * chunk_ms // 1000)
        while data := audio.readframes(frames_per_chunk):
            samples = array("h")
            samples.frombytes(data)
            if not samples:
                continue
            rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
            dbfs = 20 * math.log10(max(rms, 1.0) / 32768.0)
            if dbfs >= getattr(config, "LEADER_ENROLLMENT_SPEECH_DBFS", -50.0):
                active_chunks += 1
    return round(active_chunks * chunk_ms / 1000.0, 2)


def _normalize_audio_file(source_path: str, normalized_path: str) -> str:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        normalized_path,
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True,
            timeout=float(os.getenv("ASR_FFMPEG_TIMEOUT_SECONDS", "900")),
        )
        if completed.stderr:
            logger.warning("Audio normalization warning: %s", completed.stderr.strip())
        return normalized_path
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="audio normalization timed out") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="服务器未安装 ffmpeg，无法处理音频") from exc
    except subprocess.CalledProcessError as exc:
        error = (exc.stderr or "unsupported or damaged audio").strip()
        logger.warning("Audio normalization failed: %s", error)
        raise HTTPException(status_code=400, detail="音频文件无效、已损坏或内容与扩展名不一致") from exc
    finally:
        _remove_audio_file(source_path)
        if sys.exc_info()[0] is not None:
            _remove_audio_file(normalized_path)


def _empty_pending_voiceprint_store() -> dict[str, Any]:
    return {"version": 1, "candidates": {}}


def _read_pending_voiceprint_store() -> dict[str, Any]:
    if not PENDING_VOICEPRINT_DB_PATH.exists():
        return _empty_pending_voiceprint_store()
    try:
        with PENDING_VOICEPRINT_DB_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read pending voiceprint store: {PENDING_VOICEPRINT_DB_PATH}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("candidates", {}), dict):
        raise RuntimeError(f"invalid pending voiceprint store: {PENDING_VOICEPRINT_DB_PATH}")
    data.setdefault("version", 1)
    data.setdefault("candidates", {})
    return data


def _write_pending_voiceprint_store(data: dict[str, Any]) -> None:
    PENDING_VOICEPRINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = PENDING_VOICEPRINT_DB_PATH.with_name(
        f".{PENDING_VOICEPRINT_DB_PATH.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, PENDING_VOICEPRINT_DB_PATH)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _cleanup_expired_pending_candidates(data: dict[str, Any], current_time: float | None = None) -> bool:
    now = time.time() if current_time is None else current_time
    expired = [
        candidate_id
        for candidate_id, candidate in data["candidates"].items()
        if candidate.get("status") == "pending" and float(candidate.get("expires_at", 0)) <= now
    ]
    for candidate_id in expired:
        del data["candidates"][candidate_id]
    return bool(expired)


def _public_pending_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"samples"}
    }


def list_pending_voiceprints() -> list[dict[str, Any]]:
    with pending_voiceprints_lock:
        data = _read_pending_voiceprint_store()
        if _cleanup_expired_pending_candidates(data):
            _write_pending_voiceprint_store(data)
        candidates = [
            _public_pending_candidate(item)
            for item in data["candidates"].values()
            if item.get("status") == "pending"
        ]
    return sorted(candidates, key=lambda item: float(item.get("created_at", 0)), reverse=True)


def delete_pending_voiceprint(candidate_id: str) -> bool:
    with pending_voiceprints_lock:
        data = _read_pending_voiceprint_store()
        deleted = data["candidates"].pop(candidate_id, None) is not None
        if deleted:
            _write_pending_voiceprint_store(data)
        return deleted


def _validated_embedding(value: Any) -> list[float]:
    try:
        embedding = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("voiceprint embedding is not numeric") from exc
    if not embedding or any(not math.isfinite(item) for item in embedding):
        raise ValueError("voiceprint embedding is empty or contains non-finite values")
    return embedding


def _mean_embedding(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        raise ValueError("voiceprint samples are empty")
    dimension = len(embeddings[0])
    if dimension == 0 or any(len(item) != dimension for item in embeddings):
        raise ValueError("voiceprint sample dimensions are inconsistent")
    averaged = [sum(item[index] for item in embeddings) / len(embeddings) for index in range(dimension)]
    norm = math.sqrt(sum(value * value for value in averaged))
    if norm == 0:
        raise ValueError("voiceprint average cannot be a zero vector")
    return [value / norm for value in averaged]


def _candidate_similarity(existing_samples: list[dict[str, Any]], embeddings: list[list[float]]) -> float:
    stored = [
        _validated_embedding(sample.get("embedding", []))
        for sample in existing_samples
        if isinstance(sample, dict)
    ]
    if not stored:
        return -1.0
    best_scores: list[float] = []
    for embedding in embeddings:
        compatible = [item for item in stored if len(item) == len(embedding)]
        if not compatible:
            return -1.0
        best_scores.append(max(cosine_similarity(embedding, item) for item in compatible))
    return min(best_scores) if best_scores else -1.0


def save_pending_voiceprint(
    speaker: str,
    embeddings: list[list[float]],
    speech_seconds: float,
    source_name: str | None,
    meeting_id: str | None,
    sample_groups: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    embeddings = [_validated_embedding(item) for item in embeddings]
    if not PENDING_VOICEPRINT_MIN_SAMPLES <= len(embeddings) <= PENDING_VOICEPRINT_MAX_SAMPLES:
        raise ValueError("three to five voiceprint samples are required")
    if len(sample_groups) != len(embeddings) or any(
        len(group) != 1
        or float(group[0]["end_time"]) - float(group[0]["start_time"]) < PENDING_VOICEPRINT_MIN_SAMPLE_SECONDS - 1e-6
        or float(group[0].get("speech_seconds", 0)) < PENDING_VOICEPRINT_MIN_SAMPLE_SECONDS
        for group in sample_groups
    ):
        raise ValueError("each voiceprint sample must map to one qualified continuous interval")
    sample_similarity = min(
        cosine_similarity(embeddings[left], embeddings[right])
        for left in range(len(embeddings))
        for right in range(left + 1, len(embeddings))
    )
    if sample_similarity < PENDING_VOICEPRINT_MIN_SAMPLE_SIMILARITY:
        raise ValueError("voiceprint samples are not similar enough")
    now = time.time()
    expires_at = now + PENDING_VOICEPRINT_TTL_DAYS * 86400
    source = (source_name or "meeting_audio").replace("\\", "/").rsplit("/", 1)[-1][:255]
    meeting_ref = (meeting_id or "").strip() or None
    with pending_voiceprints_lock:
        data = _read_pending_voiceprint_store()
        _cleanup_expired_pending_candidates(data, now)
        # Keep each observation independent until a user explicitly binds its identity.

        samples = [
            {
                "embedding": embedding,
                "source": source,
                "created_at": now,
                "dimension": len(embedding),
                "voiceprint_segments": _voiceprint_segments([sample_groups[index]], index + 1),
            }
            for index, embedding in enumerate(embeddings)
        ]
        current_voiceprint_segments = _voiceprint_segments(sample_groups)
        candidate_id = f"vp_{uuid.uuid4().hex}"
        candidate = {
            "candidate_id": candidate_id,
            "collection_version": 2,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "expires_at": expires_at,
            "last_speaker": speaker,
            "speech_seconds": round(speech_seconds, 2),
            "sample_count": len(samples),
            "sample_similarity": round(sample_similarity, 5),
            "sources": [source],
            "meeting_ids": [meeting_ref] if meeting_ref else [],
            "voiceprint_segments": current_voiceprint_segments,
            "samples": samples,
        }
        data["candidates"][candidate_id] = candidate
        _write_pending_voiceprint_store(data)
        return _public_pending_candidate(candidate)


def transcribe_compatible(audio_path: str, num_speakers: int | None, hotwords: str | None):
    with asr_model_lock:
        try:
            parameter_count = len(inspect.signature(asr_service.transcribe).parameters)
        except (TypeError, ValueError):
            parameter_count = 3
        if parameter_count >= 3:
            return asr_service.transcribe(audio_path, num_speakers, hotwords)
        if parameter_count >= 2:
            return asr_service.transcribe(audio_path, num_speakers)
        return asr_service.transcribe(audio_path)


def transcribe_plain_text_compatible(audio_path: str, hotwords: str | None) -> str:
    """Run ASR without speaker diarization or any post-processing workflow."""
    with asr_model_lock:
        model = getattr(asr_service, "model", None)
        generate = getattr(model, "generate", None)
        if not callable(generate) or not hasattr(model, "spk_model"):
            return full_transcript_text(transcribe_compatible(audio_path, None, hotwords))

        original_spk_model = model.spk_model
        model.spk_model = None
        try:
            kwargs: dict[str, Any] = {
                "input": audio_path,
                "batch_size_s": getattr(config, "BATCH_SIZE_S", 300),
            }
            selected_hotwords = getattr(config, "ASR_HOTWORDS", "") if hotwords is None else hotwords.strip()
            if selected_hotwords:
                kwargs["hotword"] = selected_hotwords
            raw_results = generate(**kwargs)
        finally:
            model.spk_model = original_spk_model

    return "".join(
        str(result.get("text", "")).strip()
        for result in raw_results or []
        if str(result.get("text", "")).strip()
    )


def annotate_leaders_compatible(
    audio_path: str,
    segments: list[dict[str, Any]],
    threshold: float,
    return_scores: bool,
):
    with asr_model_lock:
        annotate = getattr(asr_service, "annotate_leaders", None)
        if not callable(annotate):
            return segments
        try:
            parameter_count = len(inspect.signature(annotate).parameters)
        except (TypeError, ValueError):
            parameter_count = 5
        if parameter_count >= 5:
            return annotate(audio_path, segments, leader_store, threshold, return_scores)
        if parameter_count >= 4:
            return annotate(audio_path, segments, leader_store, threshold)
        if parameter_count >= 3:
            return annotate(audio_path, segments, leader_store)
        return segments


def extract_voiceprint_compatible(audio_path: str) -> list[float]:
    with asr_model_lock:
        return asr_service.extract_voiceprint(audio_path)


def rescue_segments_compatible(audio_path: str, segments: list[dict[str, Any]], hotwords: str | None):
    with asr_model_lock:
        return asr_service.rescue_segments(audio_path, segments, hotwords)


def _extract_speaker_audio_compatible(audio_path: str, segments: list[dict[str, Any]]) -> str:
    extract_speaker_audio = getattr(asr_service, "_extract_speaker_audio", None)
    if callable(extract_speaker_audio):
        return extract_speaker_audio(audio_path, segments)

    clip_paths: list[str] = []
    list_path = str(Path(tempfile.gettempdir()) / f"pending_voiceprint_{uuid.uuid4().hex}.txt")
    output_path = str(Path(tempfile.gettempdir()) / f"pending_voiceprint_{uuid.uuid4().hex}.wav")
    try:
        for segment in sorted(segments, key=lambda item: float(item.get("start_time", 0))):
            start = max(0.0, float(segment.get("start_time", 0)))
            end = max(start, float(segment.get("end_time", start)))
            duration = end - start
            if duration <= 0:
                continue
            clip_path = str(Path(tempfile.gettempdir()) / f"pending_clip_{uuid.uuid4().hex}.wav")
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    str(start),
                    "-t",
                    str(duration),
                    "-i",
                    audio_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    clip_path,
                ],
                check=True,
                capture_output=True,
            )
            clip_paths.append(clip_path)
        if not clip_paths:
            raise ValueError("speaker has no usable audio segments")
        with open(list_path, "w", encoding="utf-8") as f:
            for clip_path in clip_paths:
                escaped = clip_path.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return output_path
    finally:
        for temp_path in [*clip_paths, list_path]:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _extract_pending_sample_audio(audio_path: str, group: list[dict[str, Any]]) -> str:
    if len(group) != 1:
        raise ValueError("pending sample must contain one continuous interval")
    # The ASR helper pads clips; crop the normalized WAV directly for exact provenance.
    segment = group[0]
    output_path = None
    try:
        with wave.open(audio_path, "rb") as source:
            rate = source.getframerate()
            start = int(round(float(segment["start_time"]) * rate))
            end = int(round(float(segment["end_time"]) * rate))
            if start < 0 or end <= start or end > source.getnframes():
                raise ValueError("pending sample timestamps are outside the audio")
            with tempfile.NamedTemporaryFile(prefix="pending_exact_", suffix=".wav", delete=False) as temp:
                output_path = temp.name
            source.setpos(start)
            with wave.open(output_path, "wb") as target:
                target.setparams(source.getparams())
                remaining = end - start
                frame_size = source.getnchannels() * source.getsampwidth()
                while remaining:
                    chunk = source.readframes(min(remaining, rate * 10))
                    if not chunk:
                        raise ValueError("incomplete audio while extracting pending sample")
                    target.writeframesraw(chunk)
                    remaining -= len(chunk) // frame_size
        return output_path
    except Exception:
        if output_path:
            try:
                os.remove(output_path)
            except OSError:
                pass
        raise


def _voiceprint_text_char_count(value: Any) -> int:
    return sum(1 for char in str(value or "") if char.isalnum())


def _voiceprint_segments(
    groups: list[list[dict[str, Any]]],
    first_sample_index: int = 1,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for offset, group in enumerate(groups):
        for segment in sorted(group, key=lambda item: float(item.get("start_time", 0))):
            start = max(0.0, float(segment.get("start_time", 0)))
            end = max(start, float(segment.get("end_time", start)))
            if end <= start:
                continue
            result.append(
                {
                    "sample_index": first_sample_index + offset,
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "duration": round(end - start, 2),
                    **{
                        key: segment[key] for key in ("speech_seconds", "active_ratio")
                        if key in segment
                    },
                }
            )
    return result


def _eligible_voiceprint_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: float(item.get("start_time", 0))):
        start = max(0.0, float(segment.get("start_time", 0)))
        end = max(start, float(segment.get("end_time", start)))
        duration = end - start
        if not math.isfinite(start) or not math.isfinite(end) or duration < PENDING_VOICEPRINT_MIN_SAMPLE_SECONDS:
            continue
        text_density = _voiceprint_text_char_count(segment.get("text")) / duration
        if text_density >= PENDING_VOICEPRINT_MIN_TEXT_CHARS_PER_SECOND:
            eligible.append(segment)
    return eligible


def _speaker_sample_groups(segments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    minimum = PENDING_VOICEPRINT_MIN_SAMPLE_SECONDS
    eligible = _eligible_voiceprint_segments(segments)
    groups: list[list[dict[str, Any]]] = []
    previous_end = 0.0
    for segment in eligible:
        start = max(previous_end, float(segment["start_time"]))
        end = float(segment["end_time"])
        duration = end - start
        if duration < minimum:
            continue
        # Divide only a continuous turn; never join speech across other speakers.
        target_duration = minimum / max(PENDING_VOICEPRINT_MIN_ACTIVE_RATIO, 0.1)
        minimum_count = PENDING_VOICEPRINT_MIN_SAMPLES if duration >= PENDING_VOICEPRINT_MIN_TOTAL_SPEECH_SECONDS else 1
        count = min(PENDING_VOICEPRINT_MAX_SAMPLES, max(minimum_count, int(duration // target_duration)))
        for index in range(count):
            groups.append([{
                "start_time": round(start + duration * index / count, 2),
                "end_time": round(start + duration * (index + 1) / count, 2),
            }])
        previous_end = end
    return groups


def collect_unknown_voiceprints_compatible(
    audio_path: str,
    segments: list[dict[str, Any]],
    source_name: str | None,
    meeting_id: str | None,
    leader_threshold: float,
) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        speaker = str(segment.get("speaker", "")).strip()
        if speaker:
            by_speaker.setdefault(speaker, []).append(segment)

    clean_ids = {id(segment) for segment in _overlap_free_segments(segments)}
    unknown_speakers: list[dict[str, Any]] = []
    for speaker, speaker_segments in by_speaker.items():
        if any(segment.get("leader_id") for segment in speaker_segments):
            continue
        speech_seconds = round(
            sum(
                max(0.0, float(segment.get("end_time", 0)) - float(segment.get("start_time", 0)))
                for segment in speaker_segments
            ),
            2,
        )
        # Exclude turns that overlap another diarized speaker.
        clean_segments = [segment for segment in speaker_segments if id(segment) in clean_ids]
        groups = _speaker_sample_groups(clean_segments)
        if len(groups) < PENDING_VOICEPRINT_MIN_SAMPLES:
            eligible_speech_seconds = round(
                sum(
                    max(0.0, float(segment.get("end_time", 0)) - float(segment.get("start_time", 0)))
                    for segment in _eligible_voiceprint_segments(clean_segments)
                ),
                2,
            )
            rejected_for_quality = speech_seconds > PENDING_VOICEPRINT_MIN_TOTAL_SPEECH_SECONDS
            unknown_speakers.append(
                {
                    "speaker": speaker,
                    "candidate_id": None,
                    "speech_seconds": speech_seconds,
                    "eligible_speech_seconds": eligible_speech_seconds,
                    "sample_count": 0,
                    "status": "quality_rejected" if rejected_for_quality else "insufficient_audio",
                    "required_speech_seconds": PENDING_VOICEPRINT_MIN_TOTAL_SPEECH_SECONDS,
                    "reason": "fewer_than_three_continuous_samples",
                }
            )
            continue

        sample_paths: list[str] = []
        try:
            active_seconds: list[float] = []
            active_ratios: list[float] = []
            qualified = []
            # Bound audio probing and prefer longer turns with room for pauses.
            ordered_groups = sorted(groups, key=lambda group: (-(group[0]["end_time"] - group[0]["start_time"]), group[0]["start_time"]))
            for group in ordered_groups[:20]:
                sample_path = _extract_pending_sample_audio(audio_path, group)
                sample_paths.append(sample_path)
                current_active_seconds = effective_speech_seconds(sample_path)
                sample_duration = audio_duration_seconds(sample_path)
                active_ratio = current_active_seconds / max(sample_duration, 0.01)
                if current_active_seconds < PENDING_VOICEPRINT_MIN_SAMPLE_SECONDS or active_ratio < PENDING_VOICEPRINT_MIN_ACTIVE_RATIO:
                    continue
                qualified.append((group, sample_path, current_active_seconds, active_ratio))
                if len(qualified) == PENDING_VOICEPRINT_MAX_SAMPLES:
                    break
            qualified.sort(key=lambda item: item[0][0]["start_time"])
            if len(qualified) < PENDING_VOICEPRINT_MIN_SAMPLES:
                unknown_speakers.append({
                    "speaker": speaker, "candidate_id": None, "sample_count": 0,
                    "eligible_sample_count": len(qualified),
                    "status": "quality_rejected", "reason": "fewer_than_three_clear_samples",
                    "required_speech_seconds": PENDING_VOICEPRINT_MIN_TOTAL_SPEECH_SECONDS,
                })
                continue
            groups = [item[0] for item in qualified]
            active_seconds = [item[2] for item in qualified]
            active_ratios = [item[3] for item in qualified]
            for index, group in enumerate(groups):
                group[0]["speech_seconds"] = round(active_seconds[index], 2)
                group[0]["active_ratio"] = round(active_ratios[index], 5)

            total_active_seconds = sum(active_seconds)
            voiceprint_segments = _voiceprint_segments(groups)
            if (
                total_active_seconds < PENDING_VOICEPRINT_MIN_TOTAL_SPEECH_SECONDS
                or min(active_seconds) < PENDING_VOICEPRINT_MIN_SAMPLE_SECONDS
            ):
                unknown_speakers.append(
                    {
                        "speaker": speaker,
                        "candidate_id": None,
                        "speech_seconds": round(total_active_seconds, 2),
                        "sample_count": 0,
                        "status": "insufficient_audio",
                        "required_speech_seconds": PENDING_VOICEPRINT_MIN_TOTAL_SPEECH_SECONDS,
                    }
                )
                continue

            minimum_active_ratio = min(active_ratios)
            if minimum_active_ratio < PENDING_VOICEPRINT_MIN_ACTIVE_RATIO:
                unknown_speakers.append(
                    {
                        "speaker": speaker,
                        "candidate_id": None,
                        "speech_seconds": round(total_active_seconds, 2),
                        "sample_count": 0,
                        "minimum_active_ratio": round(minimum_active_ratio, 5),
                        "status": "quality_rejected",
                    }
                )
                continue

            embeddings = [
                _validated_embedding(extract_voiceprint_compatible(sample_path))
                for _, sample_path, _, _ in qualified
            ]
            sample_similarity = min(
                cosine_similarity(embeddings[left], embeddings[right])
                for left in range(len(embeddings))
                for right in range(left + 1, len(embeddings))
            )
            if sample_similarity < PENDING_VOICEPRINT_MIN_SAMPLE_SIMILARITY:
                unknown_speakers.append(
                    {
                        "speaker": speaker,
                        "candidate_id": None,
                        "speech_seconds": round(sum(active_seconds), 2),
                        "sample_count": len(embeddings),
                        "sample_similarity": round(sample_similarity, 5),
                        "minimum_active_ratio": round(minimum_active_ratio, 5),
                        "status": "quality_rejected",
                    }
                )
                continue

            existing_matches = [leader_store.identify(item, leader_threshold) for item in embeddings]
            existing_ids = {
                str(match.get("leader_id"))
                for match in existing_matches
                if isinstance(match, dict) and match.get("leader_id")
            }
            if len(existing_ids) == 1 and all(existing_matches):
                existing_id = existing_ids.pop()
                for segment in speaker_segments:
                    segment["is_leader"] = True
                    segment["leader_id"] = existing_id
                continue

            if existing_ids:
                unknown_speakers.append({
                    "speaker": speaker, "candidate_id": None, "sample_count": len(embeddings),
                    "voiceprint_segments": voiceprint_segments,
                    "status": "identity_conflict",
                    "matched_leader_ids": sorted(existing_ids),
                    "reason": "samples_do_not_agree_on_registered_identity",
                })
                continue

            candidate = save_pending_voiceprint(
                speaker,
                embeddings,
                sum(active_seconds),
                source_name,
                meeting_id,
                groups,
            )
            candidate_id = candidate["candidate_id"]
            for segment in speaker_segments:
                segment["candidate_id"] = candidate_id
            unknown_speakers.append(
                {
                    "speaker": speaker,
                    "candidate_id": candidate_id,
                    "speech_seconds": round(sum(active_seconds), 2),
                    "sample_count": len(embeddings),
                    "collection_version": 2,
                    "sample_similarity": round(sample_similarity, 5),
                    "minimum_active_ratio": round(minimum_active_ratio, 5),
                    "voiceprint_segments": voiceprint_segments,
                    "status": "pending",
                }
            )
        except Exception as exc:
            logger.exception("Could not collect pending voiceprint for speaker %s", speaker)
            unknown_speakers.append(
                {
                    "speaker": speaker,
                    "candidate_id": None,
                    "speech_seconds": speech_seconds,
                    "sample_count": 0,
                    "status": "collection_failed",
                    "message": "voiceprint collection failed; see server logs",
                }
            )
        finally:
            for sample_path in sample_paths:
                try:
                    os.remove(sample_path)
                except OSError:
                    pass
    return unknown_speakers


def _leader_id_for_sample_source(source: str) -> str | None:
    read_store = getattr(leader_store, "_read", None)
    if not callable(read_store):
        return None
    data = read_store()
    for leader_id, leader in data.get("leaders", {}).items():
        if any(sample.get("source") == source for sample in leader.get("samples", [])):
            return str(leader_id)
    return None


def _validate_candidate_binding(user_id: str, embedding: list[float], user_exists: bool) -> None:
    score_all = getattr(leader_store, "score_all", None)
    if not callable(score_all):
        raise HTTPException(status_code=500, detail="current leader store cannot validate voiceprint binding")
    scores = score_all(embedding)
    best = scores[0] if scores else None
    target = next((item for item in scores if str(item.get("leader_id")) == user_id), None)
    threshold = float(getattr(config, "LEADER_THRESHOLD", 0.45))
    margin = float(getattr(config, "LEADER_SCORE_MARGIN", 0.03))
    if user_exists:
        if target is None or float(target.get("score", -1)) < threshold:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "candidate voiceprint does not match the existing user_id",
                    "user_id": user_id,
                    "score": target.get("score") if target else None,
                    "required_score": threshold,
                },
            )
        if (
            best
            and str(best.get("leader_id")) != user_id
            and float(best.get("score", -1)) - float(target.get("score", -1)) >= margin
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "candidate voiceprint matches another registered user more closely",
                    "requested_user_id": user_id,
                    "matched_user_id": best.get("leader_id"),
                },
            )
    elif best and float(best.get("score", -1)) >= threshold:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "candidate voiceprint already matches a registered user",
                "requested_user_id": user_id,
                "matched_user_id": best.get("leader_id"),
                "score": best.get("score"),
            },
        )


def bind_pending_voiceprint(candidate_id: str, user_id: str) -> dict[str, Any]:
    now = time.time()
    with pending_voiceprints_lock:
        data = _read_pending_voiceprint_store()
        changed = _cleanup_expired_pending_candidates(data, now)
        candidate = data["candidates"].get(candidate_id)
        if candidate is None:
            if changed:
                _write_pending_voiceprint_store(data)
            raise HTTPException(status_code=404, detail="pending voiceprint candidate not found or expired")
        if candidate.get("status") == "bound":
            if candidate.get("bound_user_id") != user_id:
                raise HTTPException(status_code=409, detail="candidate is already bound to another user_id")
            return {
                "candidate_id": candidate_id,
                "user_id": user_id,
                "leader_id": user_id,
                "status": "bound",
                "idempotent": True,
            }
        if candidate.get("status") != "pending":
            raise HTTPException(status_code=409, detail="candidate is not pending")

        embeddings = [
            _validated_embedding(sample.get("embedding", []))
            for sample in candidate.get("samples", [])
            if isinstance(sample, dict)
        ]
        if not PENDING_VOICEPRINT_MIN_SAMPLES <= len(embeddings) <= PENDING_VOICEPRINT_MAX_SAMPLES:
            raise HTTPException(status_code=409, detail="candidate must contain three to five voiceprint samples")
        source = f"pending:{candidate_id}"
        source_user_id = _leader_id_for_sample_source(source)
        if source_user_id and source_user_id != user_id:
            raise HTTPException(status_code=409, detail="candidate was already applied to another user_id")
        if source_user_id == user_id:
            item = {"leader_id": user_id}
            operation = "already_applied"
        elif leader_store.exists(user_id):
            aggregate_embedding = _mean_embedding(embeddings)
            _validate_candidate_binding(user_id, aggregate_embedding, True)
            add_sample = getattr(leader_store, "add_sample", None)
            if not callable(add_sample):
                raise HTTPException(status_code=500, detail="current leader store cannot append voiceprint samples")
            item = add_sample(user_id, None, aggregate_embedding, source)
            operation = "sample_appended"
        else:
            _validate_candidate_binding(user_id, _mean_embedding(embeddings), False)
            try:
                item = _create_leader_compatible(user_id, [(embedding, source) for embedding in embeddings])
            except LeaderAlreadyExistsError:
                add_sample = getattr(leader_store, "add_sample", None)
                if not callable(add_sample):
                    raise HTTPException(status_code=409, detail="user_id already exists")
                item = add_sample(user_id, None, _mean_embedding(embeddings), source)
                operation = "sample_appended"
            else:
                operation = "created"

        candidate["status"] = "bound"
        candidate["updated_at"] = now
        candidate["bound_at"] = now
        candidate["bound_user_id"] = user_id
        candidate["bound_sample_count"] = len(embeddings)
        candidate.pop("samples", None)
        _write_pending_voiceprint_store(data)
        return {
            "candidate_id": candidate_id,
            "user_id": user_id,
            "leader_id": user_id,
            "status": "bound",
            "operation": operation,
            "leader": item,
            "idempotent": False,
        }


def merge_adjacent_segments_compatible(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merge = getattr(asr_service, "_merge_adjacent_segments", None)
    if callable(merge):
        return merge(segments)
    return segments


def transcript_quality_compatible(
    segments: list[dict[str, Any]],
    audio_duration: float | None,
) -> dict[str, Any]:
    transcript_quality = getattr(asr_service, "transcript_quality", None)
    if callable(transcript_quality):
        return transcript_quality(segments, audio_duration)
    return {"status": "ok", "segments": len(segments), "audio_duration": audio_duration}


def should_retry_diarization_compatible(
    quality: dict[str, Any],
    requested_speakers: int | None,
    audio_duration: float | None,
) -> bool:
    should_retry = getattr(asr_service, "should_retry_diarization", None)
    if callable(should_retry):
        return should_retry(quality, requested_speakers, audio_duration)
    return False


async def reap_chunk_uploads(store: ChunkUploads):
    while True:
        try:
            await run_in_threadpool(store.expire)
        except Exception:
            logger.exception("Failed to remove expired chunk uploads")
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fast_runtime
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, asr_service.load)
    if config.POSTPROCESS_ENABLED and config.POSTPROCESS_PRELOAD:
        await loop.run_in_executor(executor, post_processor.load)
    fast = nullcontext()
    if FAST_ASR_ENABLED:
        from fast_asr import install

        fast = install(sys.modules[__name__])
    with fast as fast_runtime:
        cleanup = asyncio.create_task(reap_chunk_uploads(chunk_uploads)) if chunk_uploads is not None else None
        try:
            yield
        finally:
            fast_runtime = None
            if cleanup is not None:
                cleanup.cancel()
                with suppress(asyncio.CancelledError):
                    await cleanup


app = FastAPI(
    title="FunASR Leader ASR",
    description="Speech recognition, speaker diarization, and leader voiceprint identification.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def error_response(status_code: int, detail: Any, headers: dict[str, str] | None = None) -> JSONResponse:
    if isinstance(detail, dict):
        message = str(detail.get("message") or "request failed")
        data = {**detail, "status": "rejected"}
        data.pop("message", None)
    else:
        message = str(detail)
        data = {"status": "rejected"}
    return JSONResponse(
        status_code=status_code,
        content={"code": status_code, "message": message, "data": data},
        headers=headers,
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return error_response(exc.status_code, exc.detail, exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"location": list(item.get("loc", ())), "message": item.get("msg", ""), "type": item.get("type", "")}
        for item in exc.errors()
    ]
    return error_response(422, {"message": "request validation failed", "errors": errors})


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error")
    return error_response(500, "internal server error")


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    for schema in openapi_schema.get("components", {}).get("schemas", {}).values():
        properties = schema.get("properties", {})
        for prop in properties.values():
            items = prop.get("items")
            if isinstance(items, dict) and items.get("contentMediaType") == "application/octet-stream":
                items.pop("contentMediaType", None)
                items["format"] = "binary"
            if prop.get("contentMediaType") == "application/octet-stream":
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def ok(data: Any = None) -> dict[str, Any]:
    return {"code": 0, "message": "success", "data": data}


def transcribe_ok(data: Any = None, **extra: Any) -> dict[str, Any]:
    response = {"code": 200, "message": "success", "data": data}
    response.update(extra)
    return response


def now_ts() -> float:
    return round(time.time(), 3)


def public_correction_task(task: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in task.items() if key != "future"}


def get_correction_task(task_id: str) -> dict[str, Any] | None:
    with correction_tasks_lock:
        task = correction_tasks.get(task_id)
        if task is None:
            return None
        return public_correction_task(task)


def segment_text_chars(segments: list[dict[str, Any]]) -> int:
    return sum(len(str(segment.get("text", ""))) for segment in segments)


def full_transcript_text(segments: list[dict[str, Any]]) -> str:
    """Join final ASR segments for clients that also need a plain transcript."""
    return "".join(
        str(segment.get("text", "")).strip()
        for segment in segments
        if str(segment.get("text", "")).strip()
    )


def transcribe_response(
    segments: list[dict[str, Any]],
    return_full_text: bool,
    **extra: Any,
) -> dict[str, Any]:
    response = transcribe_ok(segments, **extra)
    if return_full_text:
        response["full_text"] = full_transcript_text(segments)
    return response


def run_correction_task(task_id: str, segments: list[dict[str, Any]], early_correction=None) -> None:
    with correction_tasks_lock:
        task = correction_tasks.get(task_id)
        if task is not None:
            task["status"] = "running"
            task["updated_at"] = now_ts()
    try:
        correct = post_processor.correct_segments
        rows = [dict(segment) for segment in segments]
        result = correct(rows) if early_correction is None else early_correction.result(rows, correct)
        with correction_tasks_lock:
            task = correction_tasks.get(task_id)
            if task is not None:
                task["status"] = "completed"
                task["updated_at"] = now_ts()
                task["result"] = result
    except Exception as exc:
        logger.exception("ASR correction task failed: %s", task_id)
        with correction_tasks_lock:
            task = correction_tasks.get(task_id)
            if task is not None:
                task["status"] = "failed"
                task["updated_at"] = now_ts()
                task["error"] = str(exc)


def submit_correction_task(segments: list[dict[str, Any]], mode: str, early_correction=None) -> tuple[str, Any]:
    task_id = uuid.uuid4().hex
    task = {
        "task_id": task_id,
        "status": "queued",
        "mode": mode,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "result": None,
        "error": None,
    }
    with correction_tasks_lock:
        correction_tasks[task_id] = task
    future = correction_executor.submit(run_correction_task, task_id, [dict(segment) for segment in segments], early_correction)
    with correction_tasks_lock:
        correction_tasks[task_id]["future"] = future
    return task_id, future


def correction_pending_fields(task_id: str, mode: str) -> dict[str, Any]:
    return {
        "correction_status": "running",
        "correction_mode": mode,
        "correction_task_id": task_id,
        "poll_action": "correction_result",
        "poll_url": f"/transcribe?action=correction_result&task_id={task_id}",
    }


def normalize_correction_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode in {"", "auto"}:
        return "auto"
    if mode in {"sync", "async", "none"}:
        return mode
    raise HTTPException(status_code=400, detail="correction_mode must be auto, sync, async, or none")


def validate_similarity_threshold(value: float, field_name: str = "threshold") -> float:
    if not 0.0 <= value <= 1.0:
        raise HTTPException(status_code=400, detail=f"{field_name} must be between 0 and 1")
    return value


def normalize_leader_id(value: str) -> str:
    leader_id = value.strip()
    if not leader_id:
        raise HTTPException(status_code=400, detail="id不能为空")
    if len(leader_id) > 128:
        raise HTTPException(status_code=400, detail="id长度不能超过128个字符")
    if any(ord(char) < 32 for char in leader_id):
        raise HTTPException(status_code=400, detail="id不能包含控制字符")
    return leader_id


def effective_speech_seconds(path: str) -> float:
    try:
        import numpy as np
    except ImportError:
        return _effective_speech_seconds_legacy(path)
    threshold = getattr(config, "LEADER_ENROLLMENT_SPEECH_DBFS", -50.0)
    active_chunks = 0
    with wave.open(path, "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            return _effective_speech_seconds_legacy(path)
        frames_per_chunk = max(1, audio.getframerate() * 20 // 1000)
        while data := audio.readframes(frames_per_chunk * 1024):
            samples = np.frombuffer(data, dtype="<i2").astype(np.int64)
            count, remaining = divmod(len(samples), frames_per_chunk)
            if count:
                blocks = samples[:count * frames_per_chunk].reshape(count, frames_per_chunk)
                powers = np.einsum("ij,ij->i", blocks, blocks)
                levels = 20 * np.log10(np.maximum(np.sqrt(powers / frames_per_chunk), 1.0) / 32768.0)
                accepted = levels >= threshold
                # Retain Python's original comparison at floating-point boundaries.
                for index in np.flatnonzero(np.abs(levels - threshold) < 1e-10):
                    rms = math.sqrt(int(powers[index]) / frames_per_chunk)
                    accepted[index] = 20 * math.log10(max(rms, 1.0) / 32768.0) >= threshold
                active_chunks += int(np.count_nonzero(accepted))
            if remaining:
                tail = samples[count * frames_per_chunk:]
                rms = math.sqrt(sum(int(value) * int(value) for value in tail) / len(tail))
                active_chunks += int(20 * math.log10(max(rms, 1.0) / 32768.0) >= threshold)
    return round(active_chunks * 20 / 1000.0, 2)


def audio_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as audio:
        return round(audio.getnframes() / max(audio.getframerate(), 1), 2)


def validate_enrollment_samples(
    samples: list[tuple[list[float], str | None, float]],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    short_samples = [
        {"source": source, "speech_seconds": speech_seconds}
        for _, source, speech_seconds in samples
        if speech_seconds < getattr(config, "LEADER_ENROLLMENT_MIN_SPEECH_SECONDS", 5.0)
    ]
    if short_samples:
        issues.append(
            {
                "message": "声纹样本中的有效语音时长不足",
                "minimum_speech_seconds": getattr(config, "LEADER_ENROLLMENT_MIN_SPEECH_SECONDS", 5.0),
                "samples": short_samples,
            }
        )

    pair_scores: list[dict[str, Any]] = []
    best_by_sample = [-1.0] * len(samples)
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            score = cosine_similarity(samples[left][0], samples[right][0])
            best_by_sample[left] = max(best_by_sample[left], score)
            best_by_sample[right] = max(best_by_sample[right], score)
            pair_scores.append(
                {
                    "left": samples[left][1],
                    "right": samples[right][1],
                    "score": round(score, 5),
                }
            )

    minimum_best_score = min(best_by_sample)
    if minimum_best_score < getattr(config, "LEADER_ENROLLMENT_MIN_SIMILARITY", 0.60):
        issues.append(
            {
                "message": "上传的声纹样本相似度过低，可能不是同一个人，或录音设备、环境差异过大",
                "minimum_similarity": getattr(config, "LEADER_ENROLLMENT_MIN_SIMILARITY", 0.60),
                "minimum_best_score": round(minimum_best_score, 5),
                "pairs": pair_scores,
            }
        )
    if issues:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "声纹注册质量校验失败",
                "issues": issues,
            },
        )
    return {
        "minimum_best_score": round(minimum_best_score, 5),
        "pairs": pair_scores,
    }


async def save_upload(file: UploadFile, prefix: str) -> str:
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    allowed_extensions = getattr(config, "ALLOWED_AUDIO_EXTENSIONS", DEFAULT_ALLOWED_AUDIO_EXTENSIONS)
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "不支持的音频格式",
                "file_name": filename or None,
                "allowed_formats": getattr(config, "ALLOWED_AUDIO_FORMATS", DEFAULT_ALLOWED_AUDIO_FORMATS),
            },
        )

    upload_id = uuid.uuid4().hex
    source_path = os.path.join(tempfile.gettempdir(), f"{prefix}_{upload_id}{ext}")
    normalized_path = os.path.join(tempfile.gettempdir(), f"{prefix}_{upload_id}_16k.wav")
    try:
        size = 0
        with open(source_path, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > config.MAX_FILE_SIZE:
                    raise HTTPException(status_code=400, detail=f"音频文件超过大小限制，最大允许 {config.MAX_FILE_SIZE} 字节")
                output.write(chunk)
                await asyncio.sleep(0)
        if size == 0:
            raise HTTPException(status_code=400, detail="上传的音频文件为空")
        future = asyncio.get_running_loop().run_in_executor(
            audio_io_executor, _normalize_audio_file, source_path, normalized_path,
        )
    except BaseException:
        _remove_audio_file(source_path)
        raise
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        def completed(task):
            try:
                task.result()
            except BaseException:
                logger.exception("Audio normalization failed after request cancellation")
            finally:
                _remove_audio_file(normalized_path)
        future.add_done_callback(completed)
        raise


def parse_embedding(raw: str) -> list[float]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="embedding must be valid JSON") from exc
    if isinstance(value, dict):
        value = value.get("embedding")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="embedding must be a numeric array")
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="embedding contains non-numeric values") from exc


@app.get("/health")
async def health():
    postprocess_enabled = bool(getattr(config, "POSTPROCESS_ENABLED", False))
    postprocess_provider = getattr(config, "POSTPROCESS_PROVIDER", None)
    rescue_provider_model_name = getattr(asr_service, "_rescue_provider_model_name", lambda provider: None)
    for name, default in {
        "BATCH_SIZE_S": None,
        "VAD_MAX_SINGLE_SEGMENT_TIME_MS": None,
        "ASR_HOTWORDS": "",
        "ASR_QUALITY_MAX_SEGMENT_SECONDS": None,
        "ASR_QUALITY_DOMINANT_SPEAKER_RATIO": None,
        "ASR_QUALITY_MIN_EXPECTED_SPEAKERS": None,
        "ASR_QUALITY_MIN_SEGMENTS_PER_HOUR": None,
        "ASR_AUTO_RETRY_BAD_DIARIZATION": False,
        "ASR_FALLBACK_SPEAKER_NUM": None,
        "ASR_RESCUE_ENABLED": False,
        "ASR_RESCUE_MAX_SEGMENTS": 0,
        "ASR_RESCUE_MAX_TOTAL_AUDIO_SECONDS": 0,
        "ASR_RESCUE_MAX_SEGMENT_SECONDS": 0,
        "ASR_RESCUE_MIN_TEXT_CHARS_PER_SECOND": 0,
        "ASR_RESCUE_LONG_SEGMENT_SECONDS": 0,
        "ASR_RESCUE_PADDING_MS": 0,
        "ASR_RESCUE_PRESET_SPEAKER_NUM": None,
        "ASR_RESCUE_MIN_SIMILARITY": None,
        "ASR_RESCUE_NOISE_MIN_SIMILARITY": None,
        "ASR_RESCUE_PROVIDER": "none",
        "ASR_RESCUE_MODEL": None,
        "ASR_RESCUE_LANGUAGE": None,
        "ASR_RESCUE_AUDIT_SAMPLES": 0,
        "ASR_FIRERED_RESCUE_MODEL": None,
        "ASR_FIRERED_RESCUE_MODEL_TYPE": None,
        "ASR_FIRERED_RESCUE_RUNTIME_PATH": None,
        "ASR_FIRERED_RESCUE_USE_GPU": False,
        "ASR_FIRERED_RESCUE_USE_HALF": False,
        "ASR_FIRERED_RESCUE_BEAM_SIZE": 0,
        "ASR_FIRERED_RESCUE_NBEST": 0,
        "ASR_FIRERED_RESCUE_RETURN_TIMESTAMP": False,
        "ASR_FIRERED_RESCUE_ALLOW_DIRECT_REPLACE": False,
        "ASR_QWEN_RESCUE_MODEL": None,
        "ASR_QWEN_RESCUE_RUNTIME_PATH": None,
        "ASR_QWEN_RESCUE_LANGUAGE": None,
        "ASR_QWEN_RESCUE_MIN_SEGMENT_SECONDS": 0,
        "ASR_QWEN_RESCUE_ALLOW_DIRECT_REPLACE": False,
        "ASR_QWEN_RESCUE_FORBIDDEN_TERMS": (),
        "ASR_QWEN_RESCUE_HINT_MIN_SIMILARITY": None,
        "ASR_QWEN_RESCUE_HINT_MIN_LENGTH_RATIO": None,
        "ASR_QWEN_RESCUE_HINT_MAX_LENGTH_RATIO": None,
        "ASR_QWEN_RESCUE_REASON_FILTER": (),
    }.items():
        if not hasattr(config, name):
            setattr(config, name, default)
    return {
        "status": "ok",
        "model_loaded": asr_service.model is not None,
        "fast_asr_active": fast_runtime is not None,
        "chunk_upload_enabled": chunk_uploads is not None,
        "device": asr_service.device,
        "leader_count": len(leader_store.list()),
        "models_dir": str(config.MODELS_DIR),
        "postprocess_enabled": postprocess_enabled,
        "postprocess_provider": postprocess_provider if postprocess_enabled else None,
        "postprocess_model": getattr(config, "POSTPROCESS_MODEL", None) if postprocess_enabled else None,
        "postprocess_device": getattr(config, "POSTPROCESS_DEVICE", None) if postprocess_enabled else None,
        "postprocess_workers": getattr(config, "POSTPROCESS_WORKERS", 0) if postprocess_enabled else 0,
        "postprocess_sync_max_chars": getattr(config, "POSTPROCESS_SYNC_MAX_CHARS", 0) if postprocess_enabled else 0,
        "postprocess_sync_timeout_seconds": getattr(config, "POSTPROCESS_SYNC_TIMEOUT_SECONDS", 0) if postprocess_enabled else 0,
        "postprocess_min_similarity": getattr(config, "POSTPROCESS_MIN_SIMILARITY", None) if postprocess_enabled else None,
        "postprocess_max_length_ratio": getattr(config, "POSTPROCESS_MAX_LENGTH_RATIO", None) if postprocess_enabled else None,
        "postprocess_max_length_extra": getattr(config, "POSTPROCESS_MAX_LENGTH_EXTRA", None) if postprocess_enabled else None,
        "deepseek_configured": bool(getattr(config, "DEEPSEEK_API_KEY", "")) if postprocess_enabled else False,
        "deepseek_model": getattr(config, "DEEPSEEK_MODEL", None)
        if postprocess_enabled and postprocess_provider == "deepseek"
        else None,
        "deepseek_context_chars": getattr(config, "DEEPSEEK_CONTEXT_CHARS", 0)
        if postprocess_enabled and postprocess_provider == "deepseek"
        else 0,
        "correction_task_workers": getattr(config, "CORRECTION_TASK_WORKERS", 0) if postprocess_enabled else 0,
        "batch_size_s": config.BATCH_SIZE_S,
        "vad_max_single_segment_time_ms": config.VAD_MAX_SINGLE_SEGMENT_TIME_MS,
        "default_hotwords_enabled": bool(config.ASR_HOTWORDS.strip()),
        "quality_thresholds": {
            "max_segment_seconds": config.ASR_QUALITY_MAX_SEGMENT_SECONDS,
            "dominant_speaker_ratio": config.ASR_QUALITY_DOMINANT_SPEAKER_RATIO,
            "min_expected_speakers": config.ASR_QUALITY_MIN_EXPECTED_SPEAKERS,
            "min_segments_per_hour": config.ASR_QUALITY_MIN_SEGMENTS_PER_HOUR,
        },
        "auto_retry_bad_diarization": config.ASR_AUTO_RETRY_BAD_DIARIZATION,
        "fallback_speaker_num": config.ASR_FALLBACK_SPEAKER_NUM,
        "rescue_asr": {
            "enabled": config.ASR_RESCUE_ENABLED,
            "max_segments": config.ASR_RESCUE_MAX_SEGMENTS,
            "max_total_audio_seconds": config.ASR_RESCUE_MAX_TOTAL_AUDIO_SECONDS,
            "max_segment_seconds": config.ASR_RESCUE_MAX_SEGMENT_SECONDS,
            "min_text_chars_per_second": config.ASR_RESCUE_MIN_TEXT_CHARS_PER_SECOND,
            "long_segment_seconds": config.ASR_RESCUE_LONG_SEGMENT_SECONDS,
            "padding_ms": config.ASR_RESCUE_PADDING_MS,
            "preset_speaker_num": config.ASR_RESCUE_PRESET_SPEAKER_NUM,
            "min_similarity": config.ASR_RESCUE_MIN_SIMILARITY,
            "noise_min_similarity": config.ASR_RESCUE_NOISE_MIN_SIMILARITY,
            "provider": config.ASR_RESCUE_PROVIDER,
            "model": rescue_provider_model_name(config.ASR_RESCUE_PROVIDER),
            "language": config.ASR_RESCUE_LANGUAGE,
            "audit_samples": config.ASR_RESCUE_AUDIT_SAMPLES,
            "firered": {
                "model": config.ASR_FIRERED_RESCUE_MODEL,
                "model_type": config.ASR_FIRERED_RESCUE_MODEL_TYPE,
                "runtime_path": config.ASR_FIRERED_RESCUE_RUNTIME_PATH,
                "use_gpu": config.ASR_FIRERED_RESCUE_USE_GPU,
                "use_half": config.ASR_FIRERED_RESCUE_USE_HALF,
                "beam_size": config.ASR_FIRERED_RESCUE_BEAM_SIZE,
                "nbest": config.ASR_FIRERED_RESCUE_NBEST,
                "return_timestamp": config.ASR_FIRERED_RESCUE_RETURN_TIMESTAMP,
                "allow_direct_replace": config.ASR_FIRERED_RESCUE_ALLOW_DIRECT_REPLACE,
            },
            "qwen": {
                "model": config.ASR_QWEN_RESCUE_MODEL,
                "runtime_path": config.ASR_QWEN_RESCUE_RUNTIME_PATH,
                "language": config.ASR_QWEN_RESCUE_LANGUAGE,
                "min_segment_seconds": config.ASR_QWEN_RESCUE_MIN_SEGMENT_SECONDS,
                "allow_direct_replace": config.ASR_QWEN_RESCUE_ALLOW_DIRECT_REPLACE,
                "forbidden_terms": list(config.ASR_QWEN_RESCUE_FORBIDDEN_TERMS),
                "hint_min_similarity": config.ASR_QWEN_RESCUE_HINT_MIN_SIMILARITY,
                "hint_length_ratio": [
                    config.ASR_QWEN_RESCUE_HINT_MIN_LENGTH_RATIO,
                    config.ASR_QWEN_RESCUE_HINT_MAX_LENGTH_RATIO,
                ],
                "reason_filter": sorted(config.ASR_QWEN_RESCUE_REASON_FILTER),
            },
        },
    }


TRANSCRIBE_ACTION_ALIASES = {
    "": "transcribe",
    "asr": "transcribe",
    "audio_transcription": "transcribe",
    "audio_transcriptions": "transcribe",
    "whisper": "transcribe",
    "whisper_transcribe": "transcribe",
    "correct": "correction_result",
    "correction": "correction_result",
    "corrections": "correction_result",
    "get_correction": "correction_result",
    "leader_enroll": "leader_enroll",
    "enroll_leader": "leader_enroll",
    "leaders_enroll": "leader_enroll",
    "leader_list": "leader_list",
    "leaders": "leader_list",
    "list_leaders": "leader_list",
    "leader_delete": "leader_delete",
    "delete_leader": "leader_delete",
    "voiceprint": "voiceprint_extract",
    "voiceprint_extract": "voiceprint_extract",
    "extract_voiceprint": "voiceprint_extract",
    "voiceprint_verify": "voiceprint_verify",
    "verify_voiceprint": "voiceprint_verify",
    "voiceprint_pending_list": "voiceprint_pending_list",
    "pending_voiceprints": "voiceprint_pending_list",
    "list_pending_voiceprints": "voiceprint_pending_list",
    "voiceprint_bind": "voiceprint_bind",
    "bind_voiceprint": "voiceprint_bind",
    "bind_pending_voiceprint": "voiceprint_bind",
    "voiceprint_pending_delete": "voiceprint_pending_delete",
    "delete_pending_voiceprint": "voiceprint_pending_delete",
}

TRANSCRIBE_ACTIONS = {
    "transcribe",
    "correction_result",
    "leader_enroll",
    "leader_list",
    "leader_delete",
    "voiceprint_extract",
    "voiceprint_verify",
    "voiceprint_pending_list",
    "voiceprint_bind",
    "voiceprint_pending_delete",
}
CHUNK_UPLOAD_ACTIONS = {"upload_init", "upload_part", "upload_status", "upload_complete"}


def normalize_action(value: str | None) -> str:
    key = (value or "transcribe").strip().lower().replace("-", "_").replace(".", "_")
    action = TRANSCRIBE_ACTION_ALIASES.get(key, key)
    if action in CHUNK_UPLOAD_ACTIONS and chunk_uploads is None:
        raise HTTPException(status_code=503, detail="分片上传未启用")
    allowed_actions = TRANSCRIBE_ACTIONS | CHUNK_UPLOAD_ACTIONS if chunk_uploads is not None else TRANSCRIBE_ACTIONS
    if action not in allowed_actions:
        allowed = ", ".join(sorted(allowed_actions))
        raise HTTPException(status_code=400, detail=f"unsupported action: {value}. allowed: {allowed}")
    return action


def required_text(value: str | None, field_name: str) -> str:
    text = (value or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    return text


def normalize_optional_reference(value: str | None, field_name: str, max_length: int = 128) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) > max_length:
        raise HTTPException(status_code=400, detail=f"{field_name} length cannot exceed {max_length}")
    if any(ord(char) < 32 for char in text):
        raise HTTPException(status_code=400, detail=f"{field_name} cannot contain control characters")
    return text


def normalize_candidate_id(value: str | None) -> str:
    candidate_id = required_text(value, "candidate_id")
    if len(candidate_id) > 64 or any(ord(char) < 32 for char in candidate_id):
        raise HTTPException(status_code=400, detail="invalid candidate_id")
    return candidate_id


def require_file(file: UploadFile | None, action: str, field_name: str = "file") -> UploadFile:
    if file is None:
        raise HTTPException(status_code=400, detail=f"{field_name} is required for action={action}")
    return file


async def run_upload_operation(operation, *args):
    try:
        return await run_in_threadpool(operation, *args)
    except UploadError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@app.post("/transcribe")
async def transcribe(
    action: str = Form(default="transcribe"),
    file: Optional[UploadFile] = File(default=None),
    files: Optional[List[UploadFile]] = File(default=None),
    audio_a: Optional[UploadFile] = File(default=None),
    audio_b: Optional[UploadFile] = File(default=None),
    model: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default=None),
    temperature: Optional[float] = Form(default=None),
    num_speakers: Optional[int] = Form(default=None),
    hotwords: Optional[str] = Form(default=None),
    identify_leaders: bool = Form(default=True),
    leader_threshold: float = Form(default=config.LEADER_THRESHOLD),
    return_leader_scores: bool = Form(default=False),
    return_full_text: bool = Form(default=False),
    collect_unknown_voiceprints: bool = Form(default=False),
    meeting_id: Optional[str] = Form(default=None),
    post_process: bool = Form(default=config.POSTPROCESS_ENABLED),
    correct_text: Optional[bool] = Form(default=None),
    correction_mode: str = Form(default="auto"),
    task_id: Optional[str] = Form(default=None),
    leader_id: Optional[str] = Form(default=None),
    id_: Optional[str] = Form(default=None, alias="id"),
    user_id: Optional[str] = Form(default=None),
    candidate_id: Optional[str] = Form(default=None),
    embedding: Optional[str] = Form(default=None),
    threshold: float = Form(default=config.LEADER_THRESHOLD),
    upload_id: Optional[str] = Form(default=None),
    upload_filename: Optional[str] = Form(default=None),
    upload_size: Optional[int] = Form(default=None),
    upload_sha256: Optional[str] = Form(default=None),
    part_number: Optional[int] = Form(default=None),
):
    _ = (model, language, response_format, temperature)
    normalized_action = normalize_action(action)
    if normalized_action in CHUNK_UPLOAD_ACTIONS:
        if normalized_action == "upload_init":
            data = await run_upload_operation(chunk_uploads.initialize, upload_filename, upload_size, upload_sha256)
        elif normalized_action == "upload_part":
            part = require_file(file, normalized_action)
            data = await run_upload_operation(chunk_uploads.store_part, upload_id, part_number, part.file)
        elif normalized_action == "upload_status":
            data = await run_upload_operation(chunk_uploads.status, upload_id)
        else:
            data = await run_upload_operation(chunk_uploads.complete, upload_id)
        return ok(data)
    if normalized_action == "correction_result":
        return await correction_result(required_text(task_id, "task_id"))
    if normalized_action == "leader_enroll":
        return await enroll_leader(required_text(leader_id or id_, "leader_id"), files or [])
    if normalized_action == "leader_list":
        return await list_leaders()
    if normalized_action == "leader_delete":
        return await delete_leader(required_text(leader_id or id_, "leader_id"))
    if normalized_action == "voiceprint_extract":
        return await voiceprint_extract(require_file(file, normalized_action))
    if normalized_action == "voiceprint_verify":
        return await voiceprint_verify(
            require_file(audio_a or file, normalized_action, "audio_a or file"),
            audio_b,
            embedding,
            threshold,
        )
    if normalized_action == "voiceprint_pending_list":
        return ok(list_pending_voiceprints())
    if normalized_action == "voiceprint_bind":
        normalized_candidate_id = normalize_candidate_id(candidate_id)
        supplied_user_ids = {
            value.strip()
            for value in (user_id, leader_id, id_)
            if value is not None and value.strip()
        }
        if len(supplied_user_ids) > 1:
            raise HTTPException(status_code=400, detail="user_id, leader_id, and id must not conflict")
        normalized_user_id = normalize_leader_id(required_text(user_id or leader_id or id_, "user_id"))
        return ok(bind_pending_voiceprint(normalized_candidate_id, normalized_user_id))
    if normalized_action == "voiceprint_pending_delete":
        normalized_candidate_id = normalize_candidate_id(candidate_id)
        if not delete_pending_voiceprint(normalized_candidate_id):
            raise HTTPException(status_code=404, detail="pending voiceprint candidate not found")
        return ok({"candidate_id": normalized_candidate_id, "deleted": True})

    if collect_unknown_voiceprints and return_full_text:
        raise HTTPException(
            status_code=400,
            detail="collect_unknown_voiceprints=true cannot be used with return_full_text=true",
        )
    if collect_unknown_voiceprints and not identify_leaders:
        raise HTTPException(
            status_code=400,
            detail="collect_unknown_voiceprints=true requires identify_leaders=true",
        )
    leader_threshold = validate_similarity_threshold(leader_threshold, "leader_threshold")
    normalized_meeting_id = normalize_optional_reference(meeting_id, "meeting_id")
    if upload_id is not None and file is None and chunk_uploads is None:
        raise HTTPException(status_code=503, detail="分片上传未启用")
    staged_input = chunk_uploads is not None and upload_id is not None
    if staged_input:
        if file is not None:
            raise HTTPException(status_code=400, detail="file 和 upload_id 只能使用其中一个")
        staged_name, staged_file = await run_upload_operation(chunk_uploads.open_complete, upload_id)
        transcribe_file = UploadFile(file=staged_file, filename=staged_name)
    else:
        transcribe_file = require_file(file, normalized_action)
    source_name = transcribe_file.filename
    progress = TranscriptionProgress(uuid.uuid4().hex[:12])
    progress.start()
    try:
        with progress.stage("upload_normalization"):
            path = await save_upload(transcribe_file, "asr")
    except BaseException:
        progress.stop()
        raise
    finally:
        if staged_input:
            await transcribe_file.close()
    try:
        effective_hotwords = hotwords if hotwords is not None else prompt
        if return_full_text:
            full_text = await _run_audio_worker(
                path, progress, _plain_transcription_pipeline, effective_hotwords,
            )
            return transcribe_ok({"full_text": full_text})
        should_correct_text = post_process if correct_text is None else correct_text
        early_correction = None
        if fast_runtime is not None and should_correct_text and correction_mode.strip().lower() in {"", "auto", "sync", "async"}:
            early_correction = fast_runtime.new_request()
        result, asr_quality, unknown_speakers = await _run_audio_worker(
            path, progress, _transcription_pipeline, num_speakers, effective_hotwords,
            identify_leaders, leader_threshold, return_leader_scores,
            collect_unknown_voiceprints, source_name, normalized_meeting_id, early_correction,
        )
        voiceprint_response_fields = (
            {"unknown_speakers": unknown_speakers}
            if collect_unknown_voiceprints
            else {}
        )
        mode = normalize_correction_mode(correction_mode)
        if mode == "none":
            should_correct_text = False
        if should_correct_text:
            if mode == "async" or (
                mode == "auto" and segment_text_chars(result) > config.POSTPROCESS_SYNC_MAX_CHARS
            ):
                task_id, _ = submit_correction_task(result, mode, early_correction)
                return transcribe_response(
                    result,
                    return_full_text,
                    asr_quality=asr_quality,
                    **voiceprint_response_fields,
                    **correction_pending_fields(task_id, mode),
                )

            task_id, future = submit_correction_task(result, mode, early_correction)
            try:
                timeout = config.POSTPROCESS_SYNC_TIMEOUT_SECONDS
                if mode == "auto":
                    timeout = min(timeout, POSTPROCESS_AUTO_WAIT_SECONDS)
                # The correction task is still pollable after a request times out or disconnects.
                await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout=timeout)
                task = get_correction_task(task_id)
                if task and task.get("status") == "completed":
                    return transcribe_response(
                        task.get("result", result),
                        return_full_text,
                        asr_quality=asr_quality,
                        **voiceprint_response_fields,
                    )
                if task and task.get("status") == "failed":
                    logger.warning("ASR correction task failed, returning raw ASR: %s", task.get("error"))
                    return transcribe_response(
                        result,
                        return_full_text,
                        asr_quality=asr_quality,
                        **voiceprint_response_fields,
                    )
            except asyncio.TimeoutError:
                return transcribe_response(
                    result,
                    return_full_text,
                    asr_quality=asr_quality,
                    **voiceprint_response_fields,
                    **correction_pending_fields(task_id, "timeout" if mode == "sync" else mode),
                )
        return transcribe_response(
            result,
            return_full_text,
            asr_quality=asr_quality,
            **voiceprint_response_fields,
        )
    finally:
        progress.logger.info("ASR request=%s response_handler_finished elapsed=%.3fs", progress.request_id, time.monotonic() - progress.started)


async def correction_result(task_id: str):
    task = get_correction_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="correction task not found")
    return ok(task)



def _create_leader_compatible(leader_id: str, embeddings: list[tuple[list[float], str | None]]) -> dict[str, Any]:
    create_leader = getattr(leader_store, "create_leader", None)
    if not callable(create_leader):
        raise HTTPException(status_code=500, detail="current leader store cannot create voiceprint")
    try:
        parameter_count = len(__import__("inspect").signature(create_leader).parameters)
    except (TypeError, ValueError):
        parameter_count = 2
    if parameter_count >= 2:
        return create_leader(leader_id, embeddings)
    add_sample = getattr(leader_store, "add_sample", None)
    if not callable(add_sample):
        raise HTTPException(status_code=500, detail="current leader store cannot create voiceprint")
    item = None
    for embedding, source in embeddings:
        item = add_sample(leader_id, None, embedding, source)
    return item or {"leader_id": leader_id, "sample_count": 0}

async def enroll_leader(
    leader_id: str = Form(..., alias="id", description="业务系统中的用户唯一ID。"),
    files: List[UploadFile] = File(..., description="请至少上传两个声纹音频文件。"),
):
    leader_id = normalize_leader_id(leader_id)
    if leader_store.exists(leader_id):
        raise HTTPException(status_code=409, detail="当前用户的声纹信息已存在")
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="至少需要上传两个声纹音频文件")

    paths: list[str] = []
    try:
        loop = asyncio.get_event_loop()
        samples: list[tuple[list[float], str | None, float]] = []
        for file in files:
            path = await save_upload(file, "leader")
            paths.append(path)
            speech_seconds = effective_speech_seconds(path)
            embedding = await loop.run_in_executor(executor, extract_voiceprint_compatible, path)
            samples.append((embedding, file.filename, speech_seconds))
        quality = validate_enrollment_samples(samples)
        embeddings = [(embedding, source) for embedding, source, _ in samples]
        try:
            item = _create_leader_compatible(leader_id, embeddings)
        except LeaderAlreadyExistsError as exc:
            raise HTTPException(status_code=409, detail="当前用户的声纹信息已存在") from exc
        item["quality"] = {
            "samples": [
                {"source": source, "speech_seconds": speech_seconds}
                for _, source, speech_seconds in samples
            ],
            **quality,
        }
        return ok(item)
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


async def list_leaders():
    return ok(leader_store.list())


async def delete_leader(leader_id: str):
    deleted = leader_store.delete(leader_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="leader not found")
    return ok({"deleted": True, "leader_id": leader_id})


async def voiceprint_extract(file: UploadFile = File(...)):
    path = await save_upload(file, "voiceprint")
    try:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(executor, extract_voiceprint_compatible, path)
        return ok({"embedding": embedding, "dimension": len(embedding)})
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


async def voiceprint_verify(
    audio_a: UploadFile = File(...),
    audio_b: Optional[UploadFile] = File(default=None),
    embedding: Optional[str] = Form(default=None),
    threshold: float = Form(default=config.LEADER_THRESHOLD),
):
    threshold = validate_similarity_threshold(threshold)
    if audio_b is None and embedding is None:
        raise HTTPException(status_code=400, detail="provide audio_b or embedding")
    if audio_b is not None and embedding is not None:
        raise HTTPException(status_code=400, detail="audio_b and embedding cannot both be provided")

    paths: list[str] = []
    try:
        loop = asyncio.get_event_loop()
        path_a = await save_upload(audio_a, "voiceprint")
        paths.append(path_a)
        embedding_a = await loop.run_in_executor(executor, extract_voiceprint_compatible, path_a)

        if audio_b is not None:
            path_b = await save_upload(audio_b, "voiceprint")
            paths.append(path_b)
            embedding_b = await loop.run_in_executor(executor, extract_voiceprint_compatible, path_b)
        else:
            embedding_b = parse_embedding(embedding or "")

        score = cosine_similarity(embedding_a, embedding_b)
        return ok(
            {
                "score": round(score, 5),
                "threshold": threshold,
                "same_speaker": score >= threshold,
                "dimension": len(embedding_a),
            }
        )
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT, workers=1)
