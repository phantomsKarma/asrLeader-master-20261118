"""Normalize an already-spooled, conventional PCM16 WAV without a second full copy."""
import asyncio
from contextlib import contextmanager
import os
import stat
import struct
import subprocess
import tempfile
from pathlib import Path


def single_pcm_data(source, total):
    source.seek(0)
    header = source.read(12)
    if (len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE"
            or int.from_bytes(header[4:8], "little") + 8 != total):
        return False
    pcm = False
    # Restrict the streaming route to the common single-data-chunk layout.
    # Other WAV layouts can require a seek during demuxing.
    for _ in range(16):
        chunk = source.read(8)
        if len(chunk) != 8:
            return False
        tag, length = struct.unpack("<4sI", chunk)
        end = source.tell() + length + length % 2
        if end > total:
            return False
        if tag == b"fmt " and length == 16 and not pcm:
            codec, channels, rate, byte_rate, align, bits = struct.unpack("<HHIIHH", source.read(16))
            pcm = (codec == 1 and channels in (1, 2) and bits == 16 and rate > 0
                   and align == 2 * channels and byte_rate == align * rate)
            if not pcm:
                return False
        elif tag == b"data":
            return pcm and length > 0 and end == total
        elif tag not in (b"LIST", b"JUNK", b"PAD "):
            return False
        source.seek(end)
    return False


def normalize(source, target, binary, timeout):
    try:
        with source:
            source.seek(0)
            subprocess.run([binary, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", "pipe:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a",
                            "pcm_s16le", str(target)], stdin=source, capture_output=True,
                           check=True, timeout=timeout)
        return str(target)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


@contextmanager
def install(main, stats):
    standard = main.save_upload

    async def save(file, prefix):
        if (prefix != "asr" or Path(file.filename or "").suffix.lower() != ".wav"
                or ".wav" not in getattr(main.config, "ALLOWED_AUDIO_EXTENSIONS", main.DEFAULT_ALLOWED_AUDIO_EXTENSIONS)
                or not getattr(file.file, "_rolled", False) or file.file.tell() != 0):
            return await standard(file, prefix)
        info = os.fstat(file.file.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= main.config.MAX_FILE_SIZE:
            return await standard(file, prefix)
        # ffmpeg inherits the OS file position. Avoid a buffered header read
        # leaving it past the RIFF header after a Python-level seek.
        source = os.fdopen(os.dup(file.file.fileno()), "rb", buffering=0)
        try:
            usable = single_pcm_data(source, info.st_size)
            source.seek(0)
        except BaseException:
            source.close()
            raise
        if not usable:
            source.close()
            file.file.seek(0)
            return await standard(file, prefix)
        target = None
        try:
            with tempfile.NamedTemporaryFile(prefix="asr_pcm_", suffix="_16k.wav", delete=False) as output:
                target = Path(output.name)
            future = asyncio.get_running_loop().run_in_executor(main.audio_io_executor, normalize, source,
                    target, "ffmpeg", float(os.getenv("ASR_FFMPEG_TIMEOUT_SECONDS", "900")))
        except BaseException:
            source.close()
            if target is not None:
                target.unlink(missing_ok=True)
            raise
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            def remove_after_exit(done):
                try:
                    done.exception()
                finally:
                    target.unlink(missing_ok=True)
            future.add_done_callback(remove_after_exit)
            raise
        except subprocess.TimeoutExpired as exc:
            raise main.HTTPException(504, "audio normalization timed out") from exc
        except FileNotFoundError as exc:
            raise main.HTTPException(500, "服务器未安装 ffmpeg，无法处理音频") from exc
        except subprocess.CalledProcessError as exc:
            raise main.HTTPException(400, "音频文件无效、已损坏或内容与扩展名不一致") from exc
        stats.update(used=True, original_bytes=info.st_size, additional_source_copy_bytes=0)
        return result

    main.save_upload = save
    try:
        yield
    finally:
        main.save_upload = standard
