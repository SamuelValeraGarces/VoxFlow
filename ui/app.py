"""
VoxFlow — Gradio WebUI
Real-time zero-shot voice conversion with GPU acceleration.
"""

import sys
import time
import threading
import gradio as gr
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.voice_converter import VoxFlowConverter, SAMPLE_RATE, CHUNK
from core.audio_io import AudioStream, list_devices, get_default_devices


# ── Global state ─────────────────────────────────────────────────────────────

converter: VoxFlowConverter = None
stream: AudioStream = None
stream_lock = threading.Lock()


def get_converter() -> VoxFlowConverter:
    global converter
    if converter is None:
        converter = VoxFlowConverter(model_dir="models", device="auto", steps=2)
    return converter


# ── Device helpers ────────────────────────────────────────────────────────────

def get_device_choices():
    try:
        inputs, outputs = list_devices()
        in_choices = [f"{i}: {name}" for i, name in inputs]
        out_choices = [f"{i}: {name}" for i, name in outputs]
        return in_choices, out_choices
    except Exception as e:
        return [f"Error: {e}"], [f"Error: {e}"]


def parse_device_id(choice: str) -> int:
    return int(choice.split(":")[0])


# ── Real-time tab ─────────────────────────────────────────────────────────────

def load_reference(audio_path: str, steps: int) -> str:
    try:
        conv = get_converter()
        conv.steps = steps
        if steps == 1:
            conv.timesteps = [1.0, 0.0]
        elif steps == 2:
            conv.timesteps = [1.0, 0.8, 0.0]
        conv.set_target_speaker(audio_path)
        return f"Reference loaded. GPU: {str(conv.device).upper()}"
    except Exception as e:
        return f"Error: {e}"


def start_stream(in_device_str: str, out_device_str: str, wasapi_exclusive: bool) -> str:
    global stream
    with stream_lock:
        if stream is not None:
            return "Already running. Stop first."

        conv = get_converter()
        if not conv.is_ready:
            return "Load a reference speaker audio first."

        conv.warmup(n_iters=5)

        in_id = parse_device_id(in_device_str)
        out_id = parse_device_id(out_device_str)

        try:
            stream = AudioStream(
                input_device=in_id,
                output_device=out_id,
                process_fn=conv.process_chunk,
                use_wasapi_exclusive=wasapi_exclusive,
            )
            stream.start()
            return f"Running — Input: {in_device_str.split(':', 1)[1].strip()}"
        except Exception as e:
            stream = None
            return f"Failed to start: {e}"


def stop_stream() -> str:
    global stream
    with stream_lock:
        if stream is None:
            return "Not running."
        try:
            stream.stop()
        except Exception:
            pass
        stream = None
        return "Stopped."


def get_status() -> str:
    global stream
    conv = get_converter() if converter is not None else None
    if stream is None or not stream.running:
        return "Stopped"
    proc = stream.processing_time_ms
    total = stream.total_latency_ms
    underruns = stream.output_underruns
    return (
        f"Running\n"
        f"Processing: {proc:.1f}ms\n"
        f"Est. total latency: {total:.0f}ms\n"
        f"Output underruns: {underruns}"
    )


# ── File conversion tab ───────────────────────────────────────────────────────

def convert_file(source_audio, reference_audio, steps: int):
    """Convert a file offline (no streaming)."""
    import tempfile
    import torchaudio.functional as taF
    import torch

    if source_audio is None or reference_audio is None:
        return None, "Upload both source and reference audio."

    try:
        conv = get_converter()
        conv.steps = steps
        conv.set_target_speaker(reference_audio)

        wav_np, sr = sf.read(source_audio, dtype='float32', always_2d=False)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)
        if sr != SAMPLE_RATE:
            waveform_t = torch.from_numpy(wav_np).unsqueeze(0)
            wav_np = taF.resample(waveform_t, sr, SAMPLE_RATE).squeeze().numpy()
        waveform = wav_np

        # Process in chunks
        n_chunks = max(1, len(waveform) // CHUNK)
        output_chunks = []

        for i in range(n_chunks):
            chunk = waveform[i * CHUNK: (i + 1) * CHUNK]
            if len(chunk) < CHUNK:
                chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
            result = conv.process_chunk(chunk)
            if result is not None:
                output_chunks.append(result)

        if not output_chunks:
            return None, "Conversion produced no output."

        output = np.concatenate(output_chunks).astype(np.float32)

        # Save to temp file (use soundfile to avoid torchcodec/FFmpeg dependency)
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        sf.write(tmp_path, output, SAMPLE_RATE)

        info = f"Converted {len(waveform)/SAMPLE_RATE:.1f}s → {len(output)/SAMPLE_RATE:.1f}s"
        return tmp_path, info

    except Exception as e:
        return None, f"Error: {e}"


# ── Build UI ──────────────────────────────────────────────────────────────────

def build_ui():
    in_choices, out_choices = get_device_choices()
    default_in = in_choices[0] if in_choices else ""
    default_out = out_choices[0] if out_choices else ""

    with gr.Blocks(title="VoxFlow", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# VoxFlow\n**Real-time zero-shot voice conversion** · MeanVC + Vocos + WASAPI")

        with gr.Tabs():

            # ── Tab 1: Real-time ──────────────────────────────────────────────
            with gr.Tab("Real-time"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 1 · Reference Speaker")
                        ref_audio = gr.Audio(
                            label="Reference audio (5–15s recommended)",
                            type="filepath",
                        )
                        steps_rt = gr.Slider(1, 2, value=2, step=1, label="Inference steps (1=faster, 2=better)")
                        load_btn = gr.Button("Load reference", variant="primary")
                        load_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("### 2 · Audio Devices")
                        in_device = gr.Dropdown(in_choices, value=default_in, label="Input device (microphone)")
                        out_device = gr.Dropdown(out_choices, value=default_out, label="Output device (speakers)")
                        wasapi_cb = gr.Checkbox(value=True, label="WASAPI Exclusive mode (lower latency, Windows only)")

                        with gr.Row():
                            start_btn = gr.Button("▶ Start", variant="primary")
                            stop_btn = gr.Button("■ Stop", variant="stop")

                    with gr.Column(scale=1):
                        gr.Markdown("### Monitor")
                        status_box = gr.Textbox(label="Stream status", lines=6, interactive=False)
                        refresh_btn = gr.Button("Refresh status")

                load_btn.click(load_reference, inputs=[ref_audio, steps_rt], outputs=load_status)
                start_btn.click(start_stream, inputs=[in_device, out_device, wasapi_cb], outputs=load_status)
                stop_btn.click(stop_stream, outputs=load_status)
                refresh_btn.click(get_status, outputs=status_box)

            # ── Tab 2: File conversion ────────────────────────────────────────
            with gr.Tab("File conversion"):
                with gr.Row():
                    with gr.Column():
                        source_audio = gr.Audio(label="Source audio", type="filepath")
                        ref_audio_file = gr.Audio(label="Reference speaker", type="filepath")
                        steps_file = gr.Slider(1, 2, value=2, step=1, label="Inference steps")
                        convert_btn = gr.Button("Convert", variant="primary")
                    with gr.Column():
                        output_audio = gr.Audio(label="Output")
                        convert_info = gr.Textbox(label="Info", interactive=False)

                convert_btn.click(
                    convert_file,
                    inputs=[source_audio, ref_audio_file, steps_file],
                    outputs=[output_audio, convert_info],
                )

            # ── Tab 3: System info ────────────────────────────────────────────
            with gr.Tab("System"):
                import torch
                gpu_info = (
                    f"GPU: {torch.cuda.get_device_name(0)} "
                    f"({torch.cuda.get_device_properties(0).total_memory // 1024**3}GB VRAM)"
                    if torch.cuda.is_available() else "GPU: Not available (CPU mode)"
                )
                gr.Markdown(f"""
### System information
- **{gpu_info}**
- Sample rate: 16kHz
- Chunk size: 200ms (3200 samples)
- Vocoder: Vocos (ISTFT-based, no GAN artifacts)
- Pipeline: ASR encoder → Mean flow matching DiT → Vocos
- Audio I/O: sounddevice WASAPI Exclusive

### Expected latency (RTX 4070)
- Processing: ~30–60ms (GPU FP16)
- Audio buffer: ~200ms (1 chunk)
- WASAPI overhead: ~10–15ms
- **Total: ~240–275ms**

### Tips
- Use **WASAPI Exclusive** for lowest latency (requires no other app using the device)
- **Steps=1** is ~30ms faster but slightly lower quality
- Reference audio: clean speech, no music, 5–15 seconds optimal
""")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
