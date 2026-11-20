"""Process-local shadow candidate for the saved production FunASR build."""

import hashlib
import inspect
import textwrap
from types import MethodType

import numpy as np
import torch


PRODUCTION_VAD_METHOD = "7de5639660246492b85d1e6c15054597deb805db179613b8ce6f2802c3990110"


def batch_fixed_windows(groups, infer, limit, timings=None):
    """Return one embedding matrix per VAD region, in input order.

    Groups contain the original sv_chunk (start, end, 24,000-sample audio)
    records. Only the current minibatch's audio is retained here. The output
    rows and timing pairs are needed later for clustering the whole recording.
    """
    if not 1 <= limit <= 128:
        raise ValueError("window batch size must be between 1 and 128")
    sizes = []
    pending = []
    blocks = []

    def flush():
        block = infer(pending)
        if block.ndim != 2 or block.shape[0] != len(pending):
            raise ValueError("speaker encoder returned unexpected row count")
        blocks.append(block)
        pending.clear()

    for group in groups:
        size = 0
        for start, end, audio in group:
            if audio.shape != (24000,):
                raise ValueError("expected FunASR 1.5-second speaker window")
            if timings is not None:
                timings.append((start, end))
            pending.append(audio)
            size += 1
            if len(pending) == limit:
                flush()
        if size == 0:
            raise ValueError("empty VAD group")
        sizes.append(size)
    if pending:
        flush()
    if not blocks:
        return []
    joined = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=0)
    return joined.split(sizes, dim=0)


def _attach_speaker_embeddings(instance, speech, sorted_regions, results,
                               timings, kwargs, cfg, sv_chunk, limit):
    def groups():
        for waveform, (region, _) in zip(speech, sorted_regions):
            yield sv_chunk([[region[0] / 1000.0, region[1] / 1000.0, np.array(waveform)]])

    def infer(windows):
        response = instance.inference(windows, input_len=None, model=instance.spk_model,
                                      kwargs=kwargs, **cfg)
        # Upstream returns one matrix per inner batch in a list of dictionaries.
        matrices = [item["spk_embedding"] for item in response]
        return matrices[0] if len(matrices) == 1 else torch.cat(matrices, dim=0)

    if not (len(results) == len(speech) == len(sorted_regions)):
        raise ValueError("ASR result count differs from VAD input count")
    for result, embedding in zip(results, batch_fixed_windows(groups(), infer, limit, timings)):
        result["spk_embedding"] = embedding


def patched_method(source):
    source = textwrap.dedent(source).rstrip() + "\n"
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != PRODUCTION_VAD_METHOD:
        raise ValueError("unrecognized FunASR inference_with_vad version")
    start = source.index("                # compose vad segments:")
    end = source.index("            beg_idx = end_idx", start)
    source = source[:start] + (
        "                _attach_speaker_embeddings(self, speech_j, sorted_data[beg_idx:end_idx],\n"
        "                                           results, all_segments, kwargs, cfg, sv_chunk, _window_limit)\n"
    ) + source[end:]
    # The upstream incremental concatenation recovers chronological VAD order.
    # Keep the same order and perform a single concatenation/host transfer.
    start = source.index('                elif k == "spk_embedding":')
    end = source.index('                elif "text" in k:', start)
    source = source[:start] + (
        '                elif k == "spk_embedding":\n'
        '                    result.setdefault(k, []).append(v)\n'
    ) + source[end:]
    source = source.replace('spk_embedding = result["spk_embedding"]',
                            'spk_embedding = torch.cat(result["spk_embedding"], dim=0).cpu()')
    return source.replace("spk_embedding.cpu()", "spk_embedding")


def install(instance, limit=32, on_speaker_batch=None):
    """Patch one explicitly supplied, isolated FunASR model instance in memory."""
    if not 1 <= limit <= 128:
        raise ValueError("window batch size must be between 1 and 128")
    original = instance.inference_with_vad.__func__
    source = patched_method(inspect.getsource(original))
    def attach(*args):
        if on_speaker_batch is not None:
            on_speaker_batch()
        return _attach_speaker_embeddings(*args)

    namespace = dict(original.__globals__, _attach_speaker_embeddings=attach, _window_limit=limit)
    exec(compile(source, "<shadow fixed-window speaker batches>", "exec"), namespace)
    instance.inference_with_vad = MethodType(namespace["inference_with_vad"], instance)
