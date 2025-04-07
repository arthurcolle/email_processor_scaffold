from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("received webhook:", data)
    return {"status": "ok"}
