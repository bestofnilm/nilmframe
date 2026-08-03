"""Line-by-line Python transcription of the original C++ FITPS.

Source: ``tests/reference/fitps.h``, kept alongside this file. It was a pybind11
extension in the pre-refactor tree (``data/highfreq/core/fitps.h``, now deleted
per plan.html's delete list -- git history retains it at the initial-import
commit). The C++ is preserved *here* rather than in the package because it is no
longer a dependency, only the oracle: it lets the torch rewrite be checked against
the algorithm it replaces rather than only against an analytic signal.

Two variants are provided:

``allocate_original``
    The weight exactly as the C++ wrote it -- ``zero_crossing_shifts[0]``, a
    constant for the whole cycle. This is the bug; it is kept so a test can
    demonstrate the error it produces.

``allocate_fixed``
    The same routine with the interpolation weight corrected to the fractional
    part ``k1 - k2``. This is the reference the torch implementation must match.

Everything else (rising-crossing detection, sub-sample shift, tolerance rejection,
buffer reset on an out-of-tolerance period) is transcribed unchanged, including
the streaming structure, so that a mismatch points at a real semantic difference
rather than at a difference in framing.
"""

from __future__ import annotations

import math


def _allocate(buffer, zero_crossings, shifts, cycle_size, *, fixed: bool):
    """Transcription of ``FITPS::allocate``.

    ``fixed=False`` reproduces the shipped C++ exactly; ``fixed=True`` corrects
    only the interpolation weight.
    """
    real_start = zero_crossings[0] + shifts[0]
    real_end = zero_crossings[1] + shifts[1]
    real_len = real_end - real_start
    dist = real_len / cycle_size

    out = [0.0] * cycle_size
    for k in range(cycle_size):
        k1 = real_start + dist * k
        k2 = int(math.floor(k1))
        k3 = int(math.ceil(k1))
        k3 = min(k3, len(buffer) - 1)  # the C++ leaves this unguarded
        weight = (k1 - k2) if fixed else shifts[0]
        out[k] = buffer[k2] + (buffer[k3] - buffer[k2]) * weight
    return out


def allocate_original(buffer, zero_crossings, shifts, cycle_size):
    return _allocate(buffer, zero_crossings, shifts, cycle_size, fixed=False)


def allocate_fixed(buffer, zero_crossings, shifts, cycle_size):
    return _allocate(buffer, zero_crossings, shifts, cycle_size, fixed=True)


class FITPSReference:
    """Streaming FITPS, transcribed from ``fitps.h``.

    Args:
        cycle_size: samples per emitted cycle.
        buffer_size: ring buffer length; the C++ asserts ``> cycle_size``.
        thresh: tolerance in *samples* on the observed period.
        fixed: use the corrected interpolation weight.
    """

    def __init__(self, cycle_size: int, buffer_size: int, thresh: int, fixed: bool = True):
        assert buffer_size > cycle_size
        self.cycle_size = cycle_size
        self.buffer_size = buffer_size
        self.thresh = thresh
        self.fixed = fixed
        self.clear()
        self.zero_crossings: list[int] = []
        self.shifts: list[float] = []

    def clear(self) -> None:
        self.volt_buffer: list[float] = []
        self.amp_buffer: list[float] = []

    def add_samples(self, volt_sample: float, amp_sample: float):
        if len(self.volt_buffer) == self.buffer_size:
            # The C++ decrements both entries unconditionally; on a deque<size_t>
            # with fewer than two entries that is out-of-range access and unsigned
            # underflow. Guarded here so the reference is well-defined.
            for n in range(len(self.zero_crossings)):
                self.zero_crossings[n] -= 1
            self.volt_buffer.pop(0)
            self.amp_buffer.pop(0)

        self.volt_buffer.append(volt_sample)
        self.amp_buffer.append(amp_sample)

        if len(self.volt_buffer) <= 1:
            return None

        prev_index = len(self.volt_buffer) - 2
        v_prev = self.volt_buffer[prev_index]
        v_last = self.volt_buffer[-1]

        if v_prev < 0 and v_last >= 0:
            if len(self.zero_crossings) == 2:
                self.zero_crossings.pop(0)
            self.zero_crossings.append(prev_index)

            shift = -v_prev / (v_last - v_prev + 1e-9)
            if len(self.shifts) == 2:
                self.shifts.pop(0)
            self.shifts.append(shift)

        if len(self.zero_crossings) != 2:
            return None

        actual = int(self.zero_crossings[1] - self.zero_crossings[0])
        out = None
        if abs(actual - self.cycle_size) <= self.thresh:
            out = (
                _allocate(
                    self.volt_buffer,
                    self.zero_crossings,
                    self.shifts,
                    self.cycle_size,
                    fixed=self.fixed,
                ),
                _allocate(
                    self.amp_buffer,
                    self.zero_crossings,
                    self.shifts,
                    self.cycle_size,
                    fixed=self.fixed,
                ),
            )

        # Reset, retaining the two samples straddling the latest crossing.
        i_1 = self.amp_buffer[prev_index]
        i_2 = self.amp_buffer[-1]
        self.zero_crossings = [0]
        self.shifts = [self.shifts[1]]
        self.clear()
        self.volt_buffer.extend([v_prev, v_last])
        self.amp_buffer.extend([i_1, i_2])
        return out

    def transform(self, volts, amps):
        v_cycles, i_cycles = [], []
        for vs, amp in zip(volts, amps, strict=True):
            result = self.add_samples(float(vs), float(amp))
            if result is not None:
                v_cycles.append(result[0])
                i_cycles.append(result[1])
        return v_cycles, i_cycles
