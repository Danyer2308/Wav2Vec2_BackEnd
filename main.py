from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from models.wav2vec2_model import transcribe_any
import tempfile

# ============================================================
# 🎧 APP CONFIGURATION
# ============================================================

app = FastAPI(title="Wav2Vec2 Speech-to-Text API")

# 🧩 CORS Middleware (para permitir conexiones externas)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes restringir a ["https://tudominio.com"] si lo deseas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🗣️ ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {"message": "✅ API de transcripción con Wav2Vec2 funcionando correctamente."}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Recibe un archivo de audio y devuelve la transcripción en texto.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    text = transcribe_any(tmp_path)
    return {"transcription": text}
