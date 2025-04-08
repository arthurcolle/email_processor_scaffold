import httpx
import redis
from fastapi import FastAPI, Request

redis_client = redis.Redis(host="redis", port=6379, db=0)

app = FastAPI()


@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("received webhook:", data)
    return {"status": "ok"}
