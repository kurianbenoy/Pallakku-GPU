from typing import Optional

from fastapi import FastAPI, Header

# from fastapi.responses import StreamingResponse

import modal

web_app = FastAPI()
stub = modal.Stub("faster-whisper-app")
image = modal.Image.from_dockerhub(
    "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel"
).pip_install("faster-whisper")

web_app = FastAPI()


@web_app.get("/checkpoint")
async def handle_root(user_agent: Optional[str] = Header(None)) -> str:
    print(f"GET /checkpoint    - received user_agent={user_agent}")
    return "Hello World"


@stub.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app


if __name__ == "__main__":
    stub.deploy("webapp")
