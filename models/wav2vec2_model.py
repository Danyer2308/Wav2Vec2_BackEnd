from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
from pydub import AudioSegment, effects
import tempfile
import os

# ============================================================
# 🧠 CONFIGURACIÓN DEL MODELO
# ============================================================

# 👉 Si en el futuro haces fine-tuning, cambia esta línea por:
# MODEL_NAME = "./wav2vec2-colombia"
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

# ============================================================
# 🎧 UTILIDADES DE AUDIO
# ============================================================

def preprocess_audio(input_path: str) -> str:
    """
    Normaliza el volumen, elimina silencios y convierte a mono 16kHz.
    Devuelve una ruta temporal WAV procesada.
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    normalized = effects.normalize(audio)

    # Quitar silencios al inicio y final
    start_trim = detect_leading_silence(normalized)
    end_trim = detect_leading_silence(normalized.reverse())
    duration = len(normalized)
    trimmed = normalized[start_trim:duration - end_trim]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    trimmed.export(tmp.name, format="wav")
    return tmp.name


def detect_leading_silence(sound, silence_threshold=-40.0, chunk_size=10):
    """
    Detecta y devuelve la duración del silencio inicial en milisegundos.
    """
    trim_ms = 0
    while trim_ms < len(sound) and sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size
    return trim_ms


def split_audio(file_path: str, chunk_length_ms=30000):
    """
    Divide el audio en fragmentos de 30s (por defecto) y devuelve las rutas.
    """
    audio = AudioSegment.from_file(file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.wav")
        chunk.export(tmp.name, format="wav")
        chunk_paths.append(tmp.name)
    return chunk_paths

# ============================================================
# 🗣️ TRANSCRIPCIÓN
# ============================================================

def transcribe_chunk(wav_path: str) -> str:
    """
    Transcribe un fragmento de audio WAV ya preprocesado.
    """
    speech_array, sampling_rate = torchaudio.load(wav_path)
    speech = speech_array.squeeze()
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()


def transcribe_any(input_path: str) -> str:
    """
    Convierte, limpia, divide y transcribe cualquier tipo de audio.
    Soporta audios largos y modelos fine-tuned.
    """
    # 🔹 1. Limpieza y normalización
    cleaned_wav = preprocess_audio(input_path)

    # 🔹 2. Dividir audio si es largo
    chunks = split_audio(cleaned_wav)
    full_transcription = []

    # 🔹 3. Transcribir cada parte
    for chunk in chunks:
        text = transcribe_chunk(chunk)
        full_transcription.append(text)
        os.remove(chunk)

    # 🔹 4. Unir todo el texto y limpiar formato
    os.remove(cleaned_wav)
    final_text = " ".join(full_transcription).strip()
    final_text = postprocess_text(final_text)
    return final_text


def postprocess_text(text: str) -> str:
    """
    Limpia el texto final: espacios, mayúsculas, punto final.
    """
    text = text.strip()
    text = " ".join(text.split())
    if text:
        text = text[0].upper() + text[1:]
        if not text.endswith("."):
            text += "."
    return text
