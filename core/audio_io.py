"""
Audio I/O via sounddevice with WASAPI Exclusive mode on Windows.
Pipeline: Input callback → queue → processing thread → queue → Output callback
"""

import time
import queue
import threading
import numpy as np
import sounddevice as sd
from typing import Callable, List, Tuple, Optional


SAMPLE_RATE = 16000
CHUNK = 3200  # 200ms


def list_devices() -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Returns (input_devices, output_devices) as list of (index, name)."""
    devices = sd.query_devices()
    inputs = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    outputs = [(i, d['name']) for i, d in enumerate(devices) if d['max_output_channels'] > 0]
    return inputs, outputs


def get_default_devices() -> Tuple[int, int]:
    """Returns (default_input_index, default_output_index)."""
    return sd.default.device[0], sd.default.device[1]


class AudioStream:
    """
    Real-time audio stream with background processing thread.

    The processing function runs in a dedicated thread to avoid blocking
    the audio callback. Output queue absorbs the processing latency.
    """

    def __init__(
        self,
        input_device: Optional[int],
        output_device: Optional[int],
        process_fn: Callable[[np.ndarray], Optional[np.ndarray]],
        use_wasapi_exclusive: bool = True,
    ):
        self.process_fn = process_fn
        self.input_q: queue.Queue = queue.Queue(maxsize=6)
        self.output_q: queue.Queue = queue.Queue(maxsize=10)
        self.running = False
        self.processing_time_ms = 0.0
        self.output_underruns = 0

        extra_in = extra_out = None
        if use_wasapi_exclusive:
            try:
                extra_in = sd.WasapiSettings(exclusive=True)
                extra_out = sd.WasapiSettings(exclusive=True)
                print("WASAPI Exclusive mode enabled.")
            except Exception:
                print("WASAPI Exclusive not available, using shared mode.")

        self.in_stream = sd.InputStream(
            device=input_device,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK,
            channels=1,
            dtype='int16',
            callback=self._input_cb,
            latency='low',
            extra_settings=extra_in,
        )
        self.out_stream = sd.OutputStream(
            device=output_device,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK,
            channels=1,
            dtype='float32',
            callback=self._output_cb,
            latency='low',
            extra_settings=extra_out,
        )

        self._proc_thread = threading.Thread(target=self._processing_loop, daemon=True)

    def _input_cb(self, indata, frames, time_info, status):
        samples = indata[:, 0].astype(np.float32) / 32768.0
        try:
            self.input_q.put_nowait(samples.copy())
        except queue.Full:
            pass  # drop oldest if queue is full

    def _output_cb(self, outdata, frames, time_info, status):
        try:
            data = self.output_q.get_nowait()
            n = min(len(data), frames)
            outdata[:n, 0] = data[:n]
            if n < frames:
                outdata[n:, 0] = 0.0
        except queue.Empty:
            outdata[:, 0] = 0.0
            self.output_underruns += 1

    def _processing_loop(self):
        while self.running:
            try:
                samples = self.input_q.get(timeout=0.3)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                output = self.process_fn(samples)
            except Exception as e:
                print(f"Processing error: {e}")
                output = None

            self.processing_time_ms = (time.perf_counter() - t0) * 1000

            if output is not None:
                try:
                    self.output_q.put_nowait(output.astype(np.float32))
                except queue.Full:
                    pass

    def start(self):
        self.running = True
        self._proc_thread.start()
        self.in_stream.start()
        self.out_stream.start()

    def stop(self):
        self.running = False
        self.in_stream.stop()
        self.out_stream.stop()
        self.in_stream.close()
        self.out_stream.close()

    @property
    def total_latency_ms(self) -> float:
        """Estimated end-to-end latency."""
        chunk_ms = CHUNK / SAMPLE_RATE * 1000  # 200ms
        io_ms = 15.0  # WASAPI Exclusive typical overhead
        return chunk_ms + self.processing_time_ms + io_ms
