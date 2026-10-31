"""Produce the audio format a caller (e.g. a voice satellite) prefers.

Assist pipelines pass preferred_format / sample_rate / channels / bytes.
When an engine doesn't support those options, Home Assistant converts the
audio itself with ffmpeg, reading it as WAV, and ffmpeg's WAV reader holds
back roughly the first 1.4 s of a live stream before emitting anything. That
delay lands on every streamed voice reply.

Supporting the options here avoids it. The engine's WAV is stripped to raw
PCM; for WAV output (what voice satellites ask for) it is resampled here, so
every sample goes out as soon as it arrives. That matters most for a short
sentence spoken before a tool call: any ffmpeg stage holds a fraction of a
second of audio until more arrives, which would keep "let me check on that"
waiting until the tool has finished. Other formats go through an ffmpeg told
the exact input format.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import logging
import math
import struct

import numpy as np

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


class Resampler:
    """Streaming s16 mono resampler: low-pass FIR, then linear interpolation.

    Stateful across chunks and emits every output sample it can compute from
    the input so far; nothing is held back waiting for more audio.
    """

    def __init__(self, in_rate: int, out_rate: int) -> None:
        self.step = in_rate / out_rate
        if out_rate < in_rate:
            # Windowed-sinc low-pass just under the new Nyquist frequency.
            cutoff = 0.45 * out_rate / in_rate
            n = np.arange(31) - 15
            taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(31)
            self.taps = (taps / taps.sum()).astype(np.float32)
        else:
            self.taps = None
        self.history = np.zeros(0 if self.taps is None else len(self.taps) - 1, dtype=np.float32)
        self.pos = 0.0  # next output position, in input samples, relative to `self.prev`
        self.prev = np.zeros(0, dtype=np.float32)  # filtered input not yet fully consumed

    def process(self, pcm: bytes) -> bytes:
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        if self.taps is not None:
            padded = np.concatenate([self.history, x])
            self.history = padded[len(padded) - len(self.history):]
            x = np.convolve(padded, self.taps, mode="valid").astype(np.float32)
        buf = np.concatenate([self.prev, x])
        if len(buf) < 2:
            self.prev = buf
            return b""
        count = max(0, math.floor((len(buf) - 1 - self.pos) / self.step) + 1)
        t = self.pos + self.step * np.arange(count)
        i = np.floor(t).astype(np.int64)
        frac = (t - i).astype(np.float32)
        y = buf[i] * (1 - frac) + buf[np.minimum(i + 1, len(buf) - 1)] * frac
        next_pos = self.pos + self.step * count
        keep = int(math.floor(next_pos))
        self.prev = buf[keep:]
        self.pos = next_pos - keep
        return np.clip(np.round(y), -32768, 32767).astype("<i2").tobytes()


class EdgeFade:
    """Fades a sentence's audio in and out (s16, interleaved).

    The engine's audio for a sentence starts and stops wherever the waveform
    happens to be, and the jump from there to the silence around it is an
    audible click. The last FADE_OUT_S is held back until the sentence ends
    so it can be faded.
    """

    FADE_IN_S = 0.005
    FADE_OUT_S = 0.015

    def __init__(self, rate: int, channels: int) -> None:
        self.frame = 2 * channels
        self.fade_in = int(rate * self.FADE_IN_S) * self.frame
        self.hold = int(rate * self.FADE_OUT_S) * self.frame
        self.done_in = 0  # bytes of the fade-in already applied
        self.held = b""

    def _ramp(self, data: bytes, start: int, total: int, up: bool) -> bytes:
        x = np.frombuffer(data, dtype="<i2").astype(np.float32)
        ch = self.frame // 2
        n = len(x) // ch
        pos = (start // self.frame + np.arange(n)) / max(1, total // self.frame)
        gain = np.clip(pos if up else 1 - pos, 0, 1)
        return (x.reshape(n, ch) * gain[:, None]).astype("<i2").tobytes()

    def feed(self, data: bytes) -> bytes:
        buf = self.held + data
        if self.done_in < self.fade_in and buf:
            n = min(len(buf), self.fade_in - self.done_in)
            buf = self._ramp(buf[:n], self.done_in, self.fade_in, True) + buf[n:]
            self.done_in += n
        cut = max(0, len(buf) - self.hold)
        cut -= cut % self.frame
        self.held = buf[cut:]
        return buf[:cut]

    def end(self) -> bytes:
        tail, self.held = self.held, b""
        return self._ramp(tail, 0, len(tail), False) if tail else b""


def _wav_header(rate: int, channels: int) -> bytes:
    """Streaming WAV header (unknown length), 16-bit PCM."""
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * channels * 2, channels * 2, 16)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


async def _wav_pcm_stream(
    pcm: AsyncIterator[tuple[int, int, bytes]], target: Target
) -> AsyncIterator[bytes]:
    """16-bit WAV output, resampled here so audio leaves as soon as it arrives."""
    resampler: Resampler | None = None
    started = False
    out_channels = target.channels or 1
    async for rate, channels, data in pcm:
        if channels != 1:  # down-mix to mono first
            data = np.frombuffer(data, dtype="<i2").reshape(-1, channels).mean(axis=1).astype("<i2").tobytes()
        if not started:
            started = True
            out_rate = target.sample_rate or rate
            if out_rate != rate:
                resampler = Resampler(rate, out_rate)
            yield _wav_header(out_rate, out_channels)
        out = resampler.process(data) if resampler else data
        if out and out_channels == 2:
            out = np.repeat(np.frombuffer(out, dtype="<i2"), 2).tobytes()
        if out:
            yield out


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

    16-bit WAV is produced here directly; other formats use ffmpeg, started on
    the first chunk once the input rate is known.
    """
    if target.format == "wav" and target.sample_bytes in (None, 2):
        async for chunk in _wav_pcm_stream(pcm, target):
            yield chunk
        return
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
