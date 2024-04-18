from typing import Optional

# import modal
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from modal import Image, Stub, asgi_app

web_app = FastAPI()
stub = Stub("pallakku")
image = Image.debian_slim().pip_install_from_requirements("requirements.txt")

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


@stub.function(
    image=image,
)
@asgi_app()
def fastapi_app():
    return web_app

