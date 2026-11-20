from __future__ import annotations

import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from threading import BoundedSemaphore


logger = logging.getLogger(__name__)
SEGMENT_KEYS = ("speaker", "start_time", "end_time", "text")


def segment_keys(rows):
    return [tuple(row[k] for k in SEGMENT_KEYS) for row in rows]


class EarlyCorrection:
    def __init__(self, runtime):
        self.runtime = runtime
        self.pending = None
        self.source = None

    def start(self, segments, merge):
        if self.pending is not None or not self.runtime.slot.acquire(blocking=False):
            return
        try:
            rows = merge(copy.deepcopy(segments))
            self.source = segment_keys(rows)
            self.pending = self.runtime.pool.submit(self.runtime.correct, rows)
            self.pending.add_done_callback(lambda _: self.runtime.slot.release())
        except Exception:
            self.runtime.slot.release()
            logger.exception("Could not start correction alongside speaker matching")

    def result(self, annotated, correct):
        if self.pending is not None and segment_keys(annotated) == self.source:
            try:
                corrected = self.pending.result()
                if len(corrected) == len(annotated):
                    result = copy.deepcopy(annotated)
                    for after, final in zip(corrected, result):
                        if any(after[key] != final[key] for key in SEGMENT_KEYS[:-1]):
                            raise ValueError("corrected segment metadata differs")
                        if after["text"] != final["text"] and after.get("raw_text") != final["text"]:
                            raise ValueError("corrected segment source text differs")
                        for key in ("text", "raw_text", "post_processed"):
                            if key in after:
                                final[key] = after[key]
                    return result
            except Exception:
                logger.exception("Could not reuse early correction, correcting final segments")
        return correct(annotated)


class FastRuntime:
    def __init__(self, correct):
        self.correct = correct
        self.slot = BoundedSemaphore(1)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="early-asr-correction")

    def new_request(self):
        return EarlyCorrection(self)


@contextmanager
def install(main):
    import numba
    from umap.umap_ import UMAP

    from .deep_camp import install_deep_prefetch
    from .npu_correction import install as install_correction
    from .qwen_rms import install as install_norm
    from .spooled_wav import install as install_wav
    from .voice_parallel import install_voice
    from .window_batches import install as install_windows

    service, post = main.asr_service, main.post_processor
    if not service.device.startswith("npu") or not post.load():
        raise RuntimeError("fast ASR requires loaded ASR and Qwen models on Ascend")

    def fit(estimator, *args, **kwargs):
        jobs = estimator.n_jobs
        threads = numba.get_num_threads()
        estimator.n_jobs = min(8, numba.config.NUMBA_NUM_THREADS)
        numba.set_num_threads(estimator.n_jobs)
        try:
            return original_fit(estimator, *args, **kwargs)
        finally:
            estimator.n_jobs = jobs
            numba.set_num_threads(threads)

    with ExitStack() as stack:
        original_vad = service.model.inference_with_vad
        stack.callback(setattr, service.model, "inference_with_vad", original_vad)
        install_windows(service.model, 32)
        prefetch = install_deep_prefetch(service.model, 32, depth=12)
        stack.callback(prefetch.uninstall)
        original_fit = UMAP.fit
        UMAP.fit = fit
        stack.callback(setattr, UMAP, "fit", original_fit)

        annotate = service.annotate_leaders
        voice = install_voice(service, workers=8)
        stack.callback(setattr, service, "annotate_leaders", annotate)

        def identify(path, segments, store, threshold, return_scores=False):
            if not store.has_samples():
                return annotate(path, segments, store, threshold, return_scores)
            return voice.annotate(path, segments, store, threshold, return_scores)

        service.annotate_leaders = identify
        rms = install_norm(post._model)
        stack.callback(rms.uninstall)
        correction = install_correction(post)
        stack.callback(correction.uninstall)
        stack.enter_context(install_wav(main, {}))
        runtime = FastRuntime(post.correct_segments)
        stack.callback(runtime.pool.shutdown, wait=True, cancel_futures=True)
        yield runtime
