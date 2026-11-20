"""Process-local overlap of fixed-window Kaldi features with Paraformer on the same ASR batch."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
import copy
import hashlib
import inspect
import threading
import time
import textwrap
from types import FunctionType, MethodType

import numpy as np
import torch


def chunks(speech, regions, limit, sv_chunk, sizes, timings):
    pending = []
    for waveform, (region, _) in zip(speech, regions):
        windows = sv_chunk([[region[0] / 1000., region[1] / 1000., np.array(waveform)]])
        if not windows:
            raise ValueError("empty VAD group")
        sizes.append(len(windows))
        for start, end, audio in windows:
            if audio.shape != (24000,):
                raise ValueError("expected a fixed 24000-sample CAMPPlus window")
            timings.append((start, end))
            pending.append(audio)
            if len(pending) == limit:
                yield pending
                pending = []
    if pending:
        yield pending


class PreparedEncoder:
    """Adapt one tile to the original inference wrapper, including any inner splits."""
    def __init__(self, speaker, windows, prepared, fs):
        self.speaker = speaker
        self.cursor = cursor = [0]
        self.eval = speaker.eval
        self.parameters = speaker.parameters
        original = speaker.inference.__func__
        features, lengths, times = prepared
        if len(features) != len(windows) or len(lengths) != len(windows) or len(times) != len(windows):
            raise RuntimeError("speaker feature row count changed")

        def loaded(data, **options):
            if options != dict(fs=16000, audio_fs=fs, data_type="sound"):
                raise RuntimeError("speaker input options changed after ASR")
            return data

        def cached(items):
            offset = cursor[0]
            end = offset + len(items)
            if end > len(windows) or any(item is not windows[i] for i, item in enumerate(items, offset)):
                raise RuntimeError("speaker sub-batch order changed")
            cursor[0] = end
            return features[offset:end], lengths[offset:end], times[offset:end]

        # The verified production method still executes its own dtype/device
        # conversion and original forward. Each tile owns the two replacements.
        namespace = dict(original.__globals__, load_audio_text_image_video=loaded, extract_feature=cached)
        private = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
        private.__kwdefaults__ = original.__kwdefaults__
        self.inference = MethodType(private, speaker)


class Prefetch:
    def __init__(self, instance, limit, depth, callback):
        self.instance, self.limit, self.depth, self.callback = instance, limit, depth, callback
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camp-host-prefetch")
        self.lock = threading.Lock()
        self.runs = deque(maxlen=64)
        self.previous = instance.inference_with_vad
        self.installed = None
        self.closed = False
        speaker = instance.spk_model.inference.__func__
        self.loader = speaker.__globals__["load_audio_text_image_video"]
        self.extract = speaker.__globals__["extract_feature"]

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.runs)

    def uninstall(self):
        if self.instance.inference_with_vad is self.installed:
            self.instance.inference_with_vad = self.previous
        self.pool.shutdown(wait=True)
        self.closed = True

    @torch.no_grad()
    def prepare(self, windows, fs):
        begin = time.perf_counter()
        audio = self.loader(windows, fs=16000, audio_fs=fs, data_type="sound")
        value = self.extract(audio)
        if value[0].device.type != "cpu":
            raise RuntimeError("CAMPPlus feature worker must stay on the host")
        return value, begin, time.perf_counter()

    def run(self, instance, speech, regions, model, kwargs, cfg, timings, sv_chunk):
        def recognize():
            return instance.inference(speech, input_len=None, model=model, kwargs=kwargs, **cfg)

        if instance.spk_model is None:
            return recognize()
        if len(regions) != len(speech):
            raise ValueError("ASR waveform/region count mismatch")
        sizes, blocks, pending = [], [], deque()
        stream = chunks(speech,regions,self.limit,sv_chunk,sizes,timings)
        fs = kwargs.get("fs",16000)
        row = dict(start_epoch=time.time(), scopes="host walls on the caller; no device synchronization",
                   depth=self.depth, pre_asr_tiles=0, windows=0, tiles=0, feature_seconds=0.,
                   feature_asr_overlap_seconds=0., feature_wait_seconds=0., speaker_host_seconds=0.,
                   max_queued_tiles=0, asr_host_seconds=0., completed=False)
        def fill():
            while len(pending) < self.depth:
                tile = next(stream,None)
                if tile is None:
                    break
                future = self.pool.submit(self.prepare, tile, fs)
                pending.append((tile,future))
                row["max_queued_tiles"] = max(row["max_queued_tiles"], len(pending))

        try:
            preparation_error = None
            try:
                fill()
            except Exception as exc:
                preparation_error = exc
            row["pre_asr_tiles"] = len(pending)
            asr_begin = time.perf_counter()
            try:
                results = recognize()
            finally:
                asr_end = time.perf_counter()
                row["asr_host_seconds"] = asr_end-asr_begin
            if self.callback:
                self.callback()
            if preparation_error:
                raise preparation_error
            if len(results) != len(speech):
                raise ValueError("ASR result count differs from VAD input count")
            while pending:
                tile, future = pending.popleft()
                start = time.perf_counter()
                value, beg_feat, end_feat = future.result()
                row["feature_wait_seconds"] += time.perf_counter()-start
                row["feature_seconds"] += end_feat-beg_feat
                row["feature_asr_overlap_seconds"] += max(0.,min(end_feat,asr_end)-max(beg_feat,asr_begin))
                # Reserve one tile for the current forward; together with pending,
                # at most depth+1 tiles are resident on the caller/worker.
                fill()
                proxy = PreparedEncoder(instance.spk_model,tile,value,fs)
                start = time.perf_counter()
                response = instance.inference(tile,input_len=None,model=proxy,kwargs=kwargs,**cfg)
                row["speaker_host_seconds"] += time.perf_counter()-start
                matrices = [item["spk_embedding"] for item in response]
                block = matrices[0] if len(matrices)==1 else torch.cat(matrices)
                if proxy.cursor[0] != len(tile) or block.ndim!=2 or block.shape[0]!=len(tile):
                    raise RuntimeError("speaker encoder did not consume every window in order")
                blocks.append(block)
                row["tiles"] += 1
                row["windows"] += len(tile)
                del proxy, value, future, tile
            if blocks:
                joined = blocks[0] if len(blocks)==1 else torch.cat(blocks)
                for result, block in zip(results,joined.split(sizes)):
                    result["spk_embedding"] = block
            row["completed"] = True
            return results
        finally:
            futures = [entry[1] for entry in pending]
            for f in futures:
                f.cancel()
            wait(futures)
            row["end_epoch"] = time.time()
            with self.lock:
                self.runs.append(row)


def install_prefetch(instance, limit=32, depth=2):
    """Install after window_batches.install on this isolated model; one handle per instance."""
    from .window_batches import patched_method
    current = instance.inference_with_vad.__func__
    if (not 1 <= depth <= 4 or current.__code__.co_filename != "<shadow fixed-window speaker batches>"
            or current.__globals__.get("_window_limit") != limit):
        raise ValueError("install matching fixed-window patch first and use a depth from 1 to 4")
    speaker = instance.spk_model.inference.__func__
    digest = hashlib.sha256((textwrap.dedent(inspect.getsource(speaker)).rstrip()+"\n").encode()).hexdigest()
    if digest != "24c9146da2f49d0403c739ecdb8dfc9397ec68b20f49b541c56d0b18f1fa7779":
        raise ValueError("unsupported installed CAMPPlus.inference")
    attach = current.__globals__["_attach_speaker_embeddings"]
    callback = inspect.getclosurevars(attach).nonlocals["on_speaker_batch"]
    source = patched_method(inspect.getsource(type(instance).inference_with_vad))
    first = source.index("            results = self.inference(\n                speech_j")
    last = source.index("            beg_idx = end_idx", first)
    between = source[first:last]
    if between.count("_attach_speaker_embeddings")!=1 or between.count("self.inference(")!=1:
        raise ValueError("unrecognized installed fixed-window ASR batch")
    source = (source[:first] + "            results = _run_prefetch(self, speech_j, sorted_data[beg_idx:end_idx],\n"
              "                                    model, kwargs, cfg, all_segments, sv_chunk)\n" + source[last:])
    code = compile(source, "<same-batch CAMPPlus host prefetch>", "exec")
    handle = Prefetch(instance,limit,depth,callback)
    namespace = dict(current.__globals__, _run_prefetch=handle.run)
    exec(code,namespace)
    handle.installed = MethodType(namespace["inference_with_vad"],instance)
    instance.inference_with_vad = handle.installed
    return handle
