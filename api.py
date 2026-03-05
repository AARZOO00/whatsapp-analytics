from fastapi import FastAPI, UploadFile, File

app = FastAPI()


@app.get("/")
def home():
    return {"message": "WhatsApp Analyzer API running"}


@app.post("/analyze")
async def analyze_chat(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    lines = text.split("\n")

    return {
        "total_messages": len(lines),
        "sample_messages": lines[:5]
    }