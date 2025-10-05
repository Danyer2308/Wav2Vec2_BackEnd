from pydub import AudioSegment, effects
import tempfile

def preprocess_audio(input_path: str) -> str:
    """
    Normaliza el volumen, elimina silencios largos y deja el audio en mono 16kHz.
    Retorna la ruta a un archivo WAV temporal procesado.
    """
    audio = AudioSegment.from_file(input_path)

    # Convertir a mono y 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Normalizar volumen
    normalized = effects.normalize(audio)

    # (Opcional) cortar silencios largos al inicio y final
    start_trim = detect_leading_silence(normalized)
    end_trim = detect_leading_silence(normalized.reverse())
    duration = len(normalized)
    trimmed = normalized[start_trim:duration - end_trim]

    # Guardar temporalmente
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    trimmed.export(tmp.name, format="wav")
    return tmp.name


def detect_leading_silence(sound, silence_threshold=-40.0, chunk_size=10):
    """
    Devuelve la cantidad de milisegundos de silencio al inicio del audio.
    """
    trim_ms = 0
    while trim_ms < len(sound) and sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size
    return trim_ms
