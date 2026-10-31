"""Produce the audio format a caller (e.g. a voice satellite) prefers.

Assist pipelines pass preferred_format / sample_rate / channels / bytes.
When an engine doesn't support those options, Home Assistant converts the
audio itself with ffmpeg, reading it as WAV, and ffmpeg's WAV reader holds
back roughly the first 1.4 s of a live stream before emitting anything. That
delay lands on every streamed voice reply.

Supporting the options here avoids it: the engine's WAV is stripped to raw
PCM and converted by an ffmpeg that is told the exact input format, so it
skips probing and starts emitting within a fraction of a second.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import logging
import struct

from homeassistant.components.tts import (
    ATTR_PREFERRED_FORMAT,
    ATTR_PREFERRED_SAMPLE_BYTES,
    ATTR_PREFERRED_SAMPLE_CHANNELS,
    ATTR_PREFERRED_SAMPLE_RATE,
)

_LOGGER = logging.getLogger(__name__)

PREFERRED_OPTIONS = [
    ATTR_PREFERRED_FORMAT,
    ATTR_PREFERRED_SAMPLE_RATE,
    ATTR_PREFERRED_SAMPLE_CHANNELS,
    ATTR_PREFERRED_SAMPLE_BYTES,
]


@dataclass(frozen=True)
class Target:
    """The output a caller asked for (None: leave as is)."""

    format: str
    sample_rate: int | None
    channels: int | None
    sample_bytes: int | None


def target_from_options(options: dict, default_format: str) -> Target | None:
    """Return the requested output, or None when no preference was given."""
    if not any(options.get(o) is not None for o in PREFERRED_OPTIONS):
        return None

    def as_int(key: str) -> int | None:
        value = options.get(key)
        return int(value) if value is not None else None

    return Target(
        format=options.get(ATTR_PREFERRED_FORMAT) or default_format,
        sample_rate=as_int(ATTR_PREFERRED_SAMPLE_RATE),
        channels=as_int(ATTR_PREFERRED_SAMPLE_CHANNELS),
        sample_bytes=as_int(ATTR_PREFERRED_SAMPLE_BYTES),
    )


class WavPcm:
    """Strips a streamed WAV down to its 16-bit sample data.

    Feed it the response bytes in any chunking; `feed` returns whatever
    sample data is available so far. `sample_rate` and `channels` are set
    once the fmt chunk has been read.
    """

    def __init__(self) -> None:
        self._header = b""
        self._in_data = False
        self._odd = b""
        self.sample_rate: int | None = None
        self.channels: int | None = None

    def feed(self, data: bytes) -> bytes:
        if not self._in_data:
            self._header += data
            data = b""
            pos = 12  # "RIFF" <size> "WAVE"
            while len(self._header) >= pos + 8:
                chunk_id = self._header[pos : pos + 4]
                size = struct.unpack("<I", self._header[pos + 4 : pos + 8])[0]
                if chunk_id == b"fmt " and len(self._header) >= pos + 16:
                    self.channels, self.sample_rate = struct.unpack(
                        "<HI", self._header[pos + 10 : pos + 16]
                    )
                if chunk_id == b"data":
                    data = self._header[pos + 8 :]
                    self._header = b""
                    self._in_data = True
                    break
                pos += 8 + size + (size & 1)
            if not self._in_data:
                return b""
        data = self._odd + data
        even = len(data) - (len(data) % 2)
        self._odd = data[even:]
        return data[:even]


def _ffmpeg_command(input_args: list[str], target: Target) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", *input_args, "-i", "pipe:0"]
    if target.sample_rate:
        command += ["-ar", str(target.sample_rate)]
    if target.channels:
        command += ["-ac", str(target.channels)]
    if target.sample_bytes == 2:
        command += ["-sample_fmt", "s16"]
    return command + ["-f", target.format, "-flush_packets", "1", "-fflags", "+bitexact", "pipe:1"]


async def convert_pcm_stream(
    pcm: AsyncIterator[tuple[int, int, bytes]], target: Target
) -> AsyncIterator[bytes]:
    """Convert a stream of (sample_rate, channels, s16le PCM) to the target.

    ffmpeg starts on the first chunk, once the input rate is known.
    """
    first = await anext(pcm, None)
    if first is None:
        return
    rate, channels, data = first
    process = await asyncio.create_subprocess_exec(
        *_ffmpeg_command(
            # Raw input of a known format: nothing to probe, so nothing held back.
            ["-f", "s16le", "-ar", str(rate), "-ac", str(channels),
             "-probesize", "32", "-analyzeduration", "0", "-fflags", "nobuffer"],
            target,
        ),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin and process.stdout

    async def write_input() -> None:
        try:
            process.stdin.write(data)
            await process.stdin.drain()
            async for _, _, chunk in pcm:
                process.stdin.write(chunk)
                await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    writer = asyncio.create_task(write_input())
    try:
        while chunk := await process.stdout.read(4096):
            yield chunk
        await writer
    finally:
        if not writer.done():
            writer.cancel()
        if process.returncode is None:
            process.kill()
        await process.wait()
        if process.returncode not in (0, -9) and process.stderr:
            err = (await process.stderr.read()).decode(errors="replace")[:300]
            _LOGGER.warning("ffmpeg conversion exited %s: %s", process.returncode, err)


async def convert_audio(data: bytes, extension: str, target: Target) -> bytes:
    """Convert a complete clip to the target (non-streaming path)."""
    process = await asyncio.create_subprocess_exec(
        *_ffmpeg_command(["-f", extension], target),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate(data)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {process.returncode}: {err.decode(errors='replace')[:300]}")
    return out
