from typing import Optional

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from modal import Image, Stub, asgi_app

GPU_TYPE = "T4"

def download_models(): 
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from faster_whisper import WhisperModel

    WhisperProcessor.from_pretrained(f"smcproject/Malwhisper-v1-small")
    WhisperForConditionalGeneration.from_pretrained(f"smcproject/Malwhisper-v1-small")

    WhisperProcessor.from_pretrained("smcproject/Malwhisper-v1-medium")
    WhisperForConditionalGeneration.from_pretrained("smcproject/Malwhisper-v1-medium")

    model_path = "vegam-whisper-medium-ml"
    model = WhisperModel(model_path, device="cuda", compute_type="float16")


web_app = FastAPI()
stub = Stub("pallakku")

image = (
    Image.from_registry("nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.1.1",
        "transformers==4.39.3"
    )
    .pip_install(
        "faster-whisper @ https://github.com/guillaumekln/faster-whisper/archive/refs/heads/master.tar.gz"
    )
    .run_function(download_models, gpu=GPU_TYPE)
)

# Initialize the processing stub with the defined Docker image
stub = Stub(name="seamless_m4t_speech", image=image)


origins = ["*"]

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def handle_root(user_agent: Optional[str] = Header(None)):
    print(f"GET /     - received user_agent={user_agent}")
    return "API is all good and passed the health check."


@web_app.get("/vegam-whiser")
async def vegam_whisper():
    from faster_whisper import WhisperModel

    model_path = "vegam-whisper-medium-ml"
    model = WhisperModel(model_path, device="cuda", compute_type="float16")

    segments, info = model.transcribe(
                    "resampled.wav",
                    beam_size=5,
                    language="ml",
                )
    result = []
    for segment in segments:
        result.append(segment.text)

    return {"text": " ".join(result)}




@stub.function(
    image=image,
)
@asgi_app()
def fastapi_app():
    return web_app

