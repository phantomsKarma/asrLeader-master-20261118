"""Bounded background window cutting and exact CAMPPlus feature preparation."""

from collections import deque
from concurrent.futures import wait
from functools import partial
import threading
import time

import torch

from .prefetch_camp import PreparedEncoder, chunks, install_prefetch


class Tiles:
    """Single producer and consumer. A reserved slot bounds work in preparation too."""
    def __init__(self, depth):
        self.depth = depth
        self.changed = threading.Condition()
        self.items = deque()
        self.cancelled = self.done = False
        self.peak = self.bytes = self.peak_bytes = 0

    def reserve(self):
        with self.changed:
            self.changed.wait_for(lambda: self.cancelled or len(self.items)<self.depth)
            return not self.cancelled

    def publish(self, tile):
        # The producer prepares exactly one tile after a successful reservation.
        with self.changed:
            if not self.cancelled:
                self.items.append(tile)
                self.bytes += tile[-1]
                self.peak_bytes = max(self.peak_bytes,self.bytes)
                self.peak = max(self.peak,len(self.items))
                self.changed.notify_all()

    def pop(self):
        with self.changed:
            self.changed.wait_for(lambda: self.items or self.done)
            if not self.items:
                return None
            tile = self.items.popleft()
            self.bytes -= tile[-1]
            self.changed.notify_all()
            return tile

    def finish(self):
        with self.changed:
            self.done = True
            self.changed.notify_all()

    def cancel(self):
        with self.changed:
            self.cancelled = True
            self.items.clear()
            self.changed.notify_all()


def _run(handle, instance, speech, regions, model, kwargs, cfg, timings, sv_chunk):
    def recognize():
        return instance.inference(speech,input_len=None,model=model,kwargs=kwargs,**cfg)
    if instance.spk_model is None:
        return recognize()
    if len(speech)!=len(regions):
        raise ValueError("ASR waveform/region count mismatch")
    # The original Paraformer reads CPU views, pads/stacks inputs into a new
    # tensor and scales out of place. The producer copies each region in chunks().
    waveforms, spans = tuple(speech), tuple((tuple(region),i) for region,i in regions)
    sample_rate = kwargs.get("fs",16000)
    queue = Tiles(handle.depth)
    blocks = []
    row = dict(start_epoch=time.time(), depth=handle.depth, windows=0, tiles=0,
               feature_seconds=0., feature_asr_overlap_seconds=0., feature_wait_seconds=0.,
               speaker_host_seconds=0., asr_host_seconds=0., windows_ready_by_asr_end=0,
               tiles_ready_by_asr_end=0, completed=False, scope="host-time intervals without device synchronization")

    def produce():
        sizes, pairs = [], []
        iterator = chunks(waveforms,spans,handle.limit,sv_chunk,sizes,pairs)
        cut_time = blocked = 0.
        try:
            while True:
                start=time.perf_counter()
                reserved=queue.reserve()
                blocked += time.perf_counter()-start
                if not reserved:
                    break
                start=time.perf_counter()
                tile=next(iterator,None)
                cut_time += time.perf_counter()-start
                if tile is None:
                    break
                prepared,beg,end = handle.prepare(tile,sample_rate)
                size_bytes=prepared[0].numel()*prepared[0].element_size()
                queue.publish((tile,prepared,beg,end,size_bytes))
                del tile, prepared
        finally:
            queue.finish()
        return sizes,pairs,cut_time,blocked

    try:
        producer = handle.pool.submit(produce)
    except RuntimeError:
        recognize()
        raise
    try:
        asr_start=time.perf_counter()
        try:
            results=recognize()
        finally:
            asr_end=time.perf_counter()
            row["asr_host_seconds"]=asr_end-asr_start
        if handle.callback:
            handle.callback()
        if len(results)!=len(speech):
            raise ValueError("ASR result count differs from VAD input count")
        while True:
            start=time.perf_counter()
            message=queue.pop()
            row["feature_wait_seconds"] += time.perf_counter()-start
            if message is None:
                break
            tile,prepared,beg,end,_=message
            if end<=asr_end:
                row["windows_ready_by_asr_end"] += len(tile)
                row["tiles_ready_by_asr_end"] += 1
            row["feature_seconds"] += end-beg
            row["feature_asr_overlap_seconds"] += max(0.,min(end,asr_end)-max(beg,asr_start))
            proxy=PreparedEncoder(instance.spk_model,tile,prepared,sample_rate)
            start=time.perf_counter()
            items=instance.inference(tile,input_len=None,model=proxy,kwargs=kwargs,**cfg)
            row["speaker_host_seconds"] += time.perf_counter()-start
            matrices=[item["spk_embedding"] for item in items]
            block=matrices[0] if len(matrices)==1 else torch.cat(matrices)
            if proxy.cursor[0]!=len(tile) or block.ndim!=2 or block.shape[0]!=len(tile):
                raise RuntimeError("speaker encoder did not consume every window in order")
            blocks.append(block)
            row["tiles"]+=1
            row["windows"]+=len(tile)
            del message,tile,prepared,proxy
        sizes,pairs,cut_time,blocked=producer.result()
        if len(sizes)!=len(results) or sum(sizes)!=len(pairs) or len(pairs)!=row["windows"]:
            raise RuntimeError("speaker group counts or timing rows changed")
        timings.extend(pairs)
        if blocks:
            joined=blocks[0] if len(blocks)==1 else torch.cat(blocks)
            for result,block in zip(results,joined.split(sizes)):
                result["spk_embedding"]=block
        row.update(completed=True, background_window_cut_seconds=cut_time,producer_capacity_wait_seconds=blocked)
        return results
    finally:
        # A failed ASR/encoder has priority over simultaneous feature failure.
        # Wake a full-queue producer and wait for the in-flight tile to release its input.
        queue.cancel()
        wait((producer,))
        row.update(end_epoch=time.time(),max_queued_tiles=queue.peak,
                   peak_queued_fbank_bytes=queue.peak_bytes)
        with handle.lock:
            handle.runs.append(row)


def install_deep_prefetch(instance, limit=32, depth=12):
    """Install on a model with window_batches installed, without installing shallow prefetch first."""
    if not 1<=depth<=16:
        raise ValueError("depth must be between 1 and 16 tiles")
    handle=install_prefetch(instance,limit,depth=2)
    handle.depth=depth
    # install_prefetch made a fresh method namespace solely for this instance.
    handle.installed.__func__.__globals__["_run_prefetch"]=partial(_run,handle)
    return handle
