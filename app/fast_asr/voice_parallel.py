"""CPU leader matching for the isolated ASR timing run (one service instance)."""
import copy
import threading
import time
import wave
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torchaudio.compliance import kaldi


@torch.inference_mode()
def feature(path):
    with wave.open(str(path), "rb") as src:
        if (src.getsampwidth(), src.getnchannels(), src.getframerate()) != (2, 1, 16000):
            raise ValueError("leader matching requires normalized PCM16 mono at 16 kHz")
        pcm = np.frombuffer(src.readframes(src.getnframes()), dtype="<i2").copy()
    fbank = kaldi.fbank(torch.from_numpy(pcm).float().div_(32768).unsqueeze(0), num_mel_bins=80)
    return fbank - fbank.mean(dim=0, keepdim=True)


class ParallelVoice:
    def __init__(self, service, network, reuse_reader, minimum, workers):
        self.network = network
        self.scorer = copy.copy(service)
        self.minimum = minimum
        self.reuse_reader = reuse_reader
        self.workers = workers
        self.history = deque(maxlen=64)
        self.lock = threading.Lock()

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.history)

    @torch.inference_mode()
    def _encode(self, batch):
        vectors = self.network(torch.stack([item[1] for item in batch]))
        if len(vectors) != len(batch):
            raise ValueError("speaker encoder row count mismatch")
        return [(item[0], row.tolist()) for item, row in zip(batch, vectors)]

    def annotate(self, audio_path, segments, leader_store, threshold, return_scores=False):
        start = time.time()
        if not leader_store.has_samples():
            raise RuntimeError("full benchmark requires an enrolled leader store")
        scorer = copy.copy(self.scorer)
        paths, eligible = [], {}
        by_speaker, aggregate = defaultdict(list), {}
        try:
            # One mutable WAV cursor is shared only on this thread while slicing.
            with self.reuse_reader(audio_path):
                for index, row in enumerate(segments):
                    if row["end_time"] - row["start_time"] < self.minimum:
                        continue
                    path = scorer._extract_clip(audio_path, row["start_time"], row["end_time"])
                    eligible[index] = path
                    paths.append(path)
                    by_speaker[row["speaker"]].append(row)
                for speaker, rows in by_speaker.items():
                    path = scorer._extract_speaker_audio(audio_path, rows)
                    aggregate[speaker] = path
                    paths.append(path)

            with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="leader-cpu") as pool:
                buckets = defaultdict(list)
                for path, data in zip(paths, pool.map(feature, paths)):
                    buckets[data.shape[0]].append((path, data))
                # CAMPPlus statistics pooling uses every frame. Only stack equal lengths.
                batches = [group[i:i + 4] for group in buckets.values() for i in range(0, len(group), 4)]
                vectors = dict(entry for block in pool.map(self._encode, batches) for entry in block)

            scores = {i: leader_store.score_all(vectors[eligible[i]]) if i in eligible else []
                      for i in range(len(segments))}
            scorer._extract_speaker_audio = lambda _, rows: aggregate[rows[0]["speaker"]]
            scorer.extract_voiceprint = vectors.__getitem__
            matches = scorer._identify_speakers(audio_path, segments, leader_store, scores)
            for i, row in enumerate(segments):
                if return_scores:
                    row["leader_candidates"] = scores[i]
                if row["speaker"] in matches:
                    scorer._mark_leader(row, matches[row["speaker"]])
            with self.lock:
                self.history.append(dict(start_epoch=start, end_epoch=time.time(), rows=len(segments),
                    workers=self.workers, device="cpu", scored=sum(bool(s) for s in scores.values()),
                    winners=len(matches), speaker_aggregates=len(aggregate), batches=len(batches)))
            return segments
        finally:
            for path in paths:
                Path(path).unlink(missing_ok=True)


def install_voice(service, workers=8):
    import config
    from funasr.models.campplus.model import CAMPPlus
    from model_service import _reuse_pcm_clips
    if workers < 1 or not isinstance(service.model.spk_model, CAMPPlus):
        raise ValueError("CAMPPlus production speaker model is required")
    model = CAMPPlus()
    model.load_state_dict(service.model.spk_model.state_dict(), strict=True)
    probe = ParallelVoice(service, model.cpu().eval(), _reuse_pcm_clips, config.LEADER_MIN_SEGMENT_SECONDS, workers)
    service.annotate_leaders = probe.annotate
    return probe
