"""Bridge-side acoustic echo canceller (PBFDAF).

Partitioned-block frequency-domain adaptive filter with per-bin
power-normalised NLMS updates. Adaptation is guarded by a double-talk
gate (see EchoCanceller._process_block): it tracks the echo-path power
gain over the whole filter window and freezes adaptation whenever the
microphone carries energy the echo path cannot explain (near-end
speech), so the user's voice never teaches the filter.

Intended use in open-xiaoai-bridge (see design.md):
  - mic:  uplink microphone PCM, 16 kHz mono (int16 or float32)
  - ref:  the exact TTS PCM the bridge sends to the speaker,
          resampled to 16 kHz and bulk-delay aligned by the caller
          (see estimate_delay).

Streaming API:
    aec = EchoCanceller(filter_len=2048, block_len=256)
    clean = aec.process(mic_frame, ref_frame)   # same length out as in

The filter adapts block by block (block_len samples per step); arbitrary
input chunk sizes are buffered internally.
"""

from __future__ import annotations

import numpy as np


def to_float(x: np.ndarray) -> np.ndarray:
    """Convert int16 PCM (or float input) to float32 in [-1, 1]."""
    x = np.asarray(x)
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0).copy()
    return x.astype(np.float32, copy=True)


def to_int16(x: np.ndarray) -> np.ndarray:
    """Convert float32 in [-1, 1] back to int16 PCM (clipped)."""
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def estimate_delay(mic: np.ndarray, ref: np.ndarray, max_delay: int) -> int:
    """Estimate the bulk delay of `ref` inside `mic` via cross-correlation.

    Returns the lag L (samples) such that mic[n] ~ ref[n - L].
    Searches lags in [0, max_delay]. Uses float arrays (any scale).
    """
    mic = np.asarray(mic, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = len(mic) + len(ref)
    nfft = 1 << int(np.ceil(np.log2(n)))
    m = np.fft.rfft(mic, nfft)
    r = np.fft.rfft(ref, nfft)
    corr = np.fft.irfft(m * np.conj(r), nfft)
    # corr[L] corresponds to sum_n mic[n] * ref[n - L]
    hi = min(max_delay, len(corr) - 1)
    lag = int(np.argmax(corr[: hi + 1]))
    return lag


class EchoCanceller:
    """Partitioned-block FDAF echo canceller (overlap-save)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        filter_len: int = 2048,
        block_len: int = 256,
        mu: float = 0.4,
        dtd_ratio: float = 1.0,
    ):
        if filter_len % block_len != 0:
            raise ValueError("filter_len must be a multiple of block_len")
        self.sample_rate = sample_rate
        self.block_len = block_len
        self.n_part = filter_len // block_len
        self.fft_len = 2 * block_len
        self.n_bins = self.fft_len // 2 + 1
        self.mu = mu
        self.dtd_ratio = dtd_ratio
        self.reset()

    def reset(self):
        b, p, n = self.block_len, self.n_part, self.fft_len
        # Frequency-domain weights, one spectrum per partition.
        nb = self.n_bins
        self._w = np.zeros((p, nb), dtype=np.complex128)
        # Reference spectra history (newest first).
        self._x = np.zeros((p, nb), dtype=np.complex128)
        # Per-bin reference power estimate (summed over partitions).
        self._pxx = np.full(nb, 1e-6, dtype=np.float64)
        # Raw sample buffers so callers may feed arbitrary chunk sizes.
        self._mic_buf = np.zeros(0, dtype=np.float32)
        self._ref_buf = np.zeros(0, dtype=np.float32)
        # Previous reference block (for overlap-save FFT input).
        self._ref_prev = np.zeros(b, dtype=np.float64)
        # Double-talk gate state (see _process_block).
        self._ref_hist = np.zeros(0, dtype=np.float64)
        self._p_ref_s = 0.0
        self._p_mic_s = 0.0
        self._gain2 = None  # tracked echo-path power gain (mic/ref)
        self._total_adapted = 0
        self._hangover = 0
        self._freeze_stretch = 0
        self._stretch_ratios: list[float] = []
        self._stretch_peak_fired = False

    # -- internals ------------------------------------------------------

    def _constrain(self, w_spec: np.ndarray) -> np.ndarray:
        """Gradient constraint: keep only the first block_len taps."""
        t = np.fft.irfft(w_spec, self.fft_len)
        t[self.block_len :] = 0.0
        return np.fft.rfft(t)

    def _process_block(self, mic_blk: np.ndarray, ref_blk: np.ndarray) -> np.ndarray:
        b = self.block_len
        mic64 = mic_blk.astype(np.float64)
        ref64 = ref_blk.astype(np.float64)

        # Shift in the new reference spectrum.
        seg = np.concatenate([self._ref_prev, ref64])
        self._ref_prev = ref64.copy()
        self._x[1:] = self._x[:-1]
        self._x[0] = np.fft.rfft(seg)

        # Echo estimate: sum over partitions, take the last block.
        y_full = np.fft.irfft(np.sum(self._w * self._x, axis=0), self.fft_len)
        y = y_full[b:]
        err = mic64 - y

        # Double-talk / robustness gate (decides whether this block may
        # adapt the filter). Uses smoothed powers (~100 ms), not per-block
        # peaks, so syllabic fluctuations do not fool it:
        #  - The gate tracks the echo-path power gain g2 = P_mic / P_ref
        #    over echo-dominant blocks. A block is echo-dominant while
        #    the mic power stays within 2.5x of the expected echo power
        #    g2 * P_ref; unexplained extra energy means near-end speech
        #    -> freeze, with a 10-block hangover bridging syllable gaps.
        #  - Before the gain is known (session start, in practice always
        #    echo-only), adapt freely for the first blocks, guarded only
        #    by the classic Geigel peak freeze (factor 2).
        #  - If freezing persists >3 s with a *stable* mic/ref ratio and
        #    no peak freeze at all, the echo path itself changed (e.g.
        #    volume): adopt the new gain and resume adapting. Double-talk
        #    ratios fluctuate far too much to pass the stability check.
        self._ref_hist = np.concatenate([self._ref_hist, np.abs(ref64)])
        win = self.n_part * b
        if len(self._ref_hist) > win:
            self._ref_hist = self._ref_hist[-win:]
        ref_max = self._ref_hist.max() if len(self._ref_hist) else 0.0
        mic_max = np.abs(mic64).max() if len(mic64) else 0.0
        p_mic = float(np.mean(mic64**2))
        # Reference activity must be measured over the whole filter
        # window, not the current block: while the current reference
        # block is silent (syllable gap), the echo tail of earlier loud
        # blocks is still sounding, and those blocks carry exactly the
        # information the tail partitions learn from.
        p_ref_win = float(np.mean(self._ref_hist**2)) if len(self._ref_hist) else 0.0
        # Slow smoothing (~0.5 s) for the dominance comparison.
        self._p_ref_s = 0.97 * self._p_ref_s + 0.03 * p_ref_win
        self._p_mic_s = 0.97 * self._p_mic_s + 0.03 * p_mic
        peak_freeze = mic_max > 2.0 * max(ref_max, 1e-9)

        adapt = False
        if p_ref_win <= 1e-12:
            self._freeze_stretch = 0
            self._stretch_ratios = []
            self._stretch_peak_fired = False
        elif self._gain2 is None:
            adapt = not peak_freeze
            if adapt and self._total_adapted >= 60 and self._p_ref_s > 1e-9:
                self._gain2 = float(
                    np.clip(self._p_mic_s / self._p_ref_s, 1e-4, 1.0)
                )
        else:
            expected = self._gain2 * self._p_ref_s
            if self._hangover > 0:
                self._hangover -= 1
                dominant = False
            else:
                dominant = self._p_mic_s < 2.5 * max(expected, 1e-12)
            if dominant and not peak_freeze:
                adapt = True
                self._freeze_stretch = 0
                self._stretch_ratios = []
                self._stretch_peak_fired = False
                if self._p_ref_s > 1e-9:
                    r = float(np.clip(self._p_mic_s / self._p_ref_s, 1e-4, 1.0))
                    self._gain2 = 0.95 * self._gain2 + 0.05 * r
            else:
                if dominant is False and self._p_mic_s >= 2.5 * max(expected, 1e-12):
                    self._hangover = 10
                self._freeze_stretch += 1
                if peak_freeze:
                    self._stretch_peak_fired = True
                if self._p_ref_s > 1e-12:
                    self._stretch_ratios.append(self._p_mic_s / self._p_ref_s)
                if self._freeze_stretch >= 190:
                    ratios = np.array(self._stretch_ratios[-190:])
                    if (
                        not self._stretch_peak_fired
                        and len(ratios) == 190
                        and ratios.max() / max(ratios.min(), 1e-12) < 2.0
                    ):
                        self._gain2 = float(np.median(ratios))
                    self._freeze_stretch = 0
                    self._stretch_ratios = []
                    self._stretch_peak_fired = False

        if adapt:
            self._total_adapted += 1
            # Per-bin power normalisation across partitions.
            p_inst = np.sum(np.abs(self._x) ** 2, axis=0)
            self._pxx = 0.9 * self._pxx + 0.1 * p_inst
            e_spec = np.fft.rfft(np.concatenate([np.zeros(b), err]))
            grad = np.conj(self._x) * (e_spec / (self._pxx + 1e-6))[None, :]
            self._w += self.mu * grad
            for p in range(self.n_part):
                self._w[p] = self._constrain(self._w[p])

        return err.astype(np.float32)

    # -- public API -----------------------------------------------------

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Cancel echo from `mic` using reference `ref`.

        Inputs: equal-length 1-D arrays, int16 PCM or float in [-1, 1].
        Output: float32 echo-cancelled samples. Streaming semantics: the
        output is the processed stream in order; because processing runs
        block by block, a call may return up to block_len-1 fewer samples
        than it consumed (they arrive with the next call). Cumulative
        output therefore lags cumulative input by < block_len samples.
        """
        mic_f = to_float(mic)
        ref_f = to_float(ref)
        if len(mic_f) != len(ref_f):
            raise ValueError("mic and ref must have equal length")
        self._mic_buf = np.concatenate([self._mic_buf, mic_f])
        self._ref_buf = np.concatenate([self._ref_buf, ref_f])

        out = np.zeros(0, dtype=np.float32)
        b = self.block_len
        while len(self._mic_buf) >= b:
            m_blk, self._mic_buf = self._mic_buf[:b], self._mic_buf[b:]
            r_blk, self._ref_buf = self._ref_buf[:b], self._ref_buf[b:]
            out = np.concatenate([out, self._process_block(m_blk, r_blk)])
        return out
