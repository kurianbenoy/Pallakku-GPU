from typing import Optional, BinaryIO, Union

import ffmpeg
import numpy as np
from fastapi import FastAPI, Header, File, UploadFile
from faster_whisper import WhisperModel

# from fastapi.responses import StreamingResponse

import modal

web_app = FastAPI()
stub = modal.Stub("faster-whisper-app")
image = modal.Image.from_dockerhub(
    "onerahmet/openai-whisper-asr-webservice"
).pip_install("ffmpeg-python")

web_app = FastAPI()
SAMPLE_RATE = 16000


@web_app.get("/checkpoint")
async def handle_root(user_agent: Optional[str] = Header(None)) -> str:
    print(f"GET /checkpoint    - received user_agent={user_agent}")
    return "Hello World"


def load_audio(file: BinaryIO, encode=True, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    if encode:
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd="ffmpeg",
                    capture_stdout=True,
                    capture_stderr=True,
                    input=file.read(),
                )
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


@stub.function(gpu="T4")
@web_app.post("/")
async def upload_file(file: UploadFile = File(...)):
    # Process the file contents here
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    segments, info = model.transcribe(load_audio(file.file), beam_size=5, language="en")
    chunks = [(segment.start, segment.end, segment.text) for segment in segments]
    return {"filename": file.filename, "language": info.language, "chunks": chunks}


@stub.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app


if __name__ == "__main__":
    stub.deploy("webapp")
