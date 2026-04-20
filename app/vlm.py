import requests
import json
import base64

from ocr import extract_blocks

MODEL = "qwen3-vl"

PROMPT_STRUCT = """
Convertí el siguiente contenido en JSON válido EXACTO con este formato:

{
  "noticias": [
    {
      "titulo": "",
      "texto": "",
      "seccion": "",
      "bounding_box": {"x":0,"y":0,"w":0,"h":0},
      "volanta": null,
      "bajada": null,
      "epigrafes": [],
      "destacados": [],
      "complementarios": [],
      "autor": null
    }
  ]
}

Reglas:
- Solo JSON válido
- No markdown
- No texto extra
- Si algo no está, dejar vacío o null
"""

def call_ollama(messages):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": MODEL,
            "stream": False,
            "messages": messages,
        },
    )

    data = response.json()

    if "message" not in data:
        print("RESPUESTA CRUDA:")
        print(data)
        return data.get("response", "")

    return data["message"]["content"]


def run_vlm(image_path: str):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    blocks = extract_blocks(image_path)

    if not blocks:
        print("OCR VACÍO")
        return None

    ocr_text = "\n".join([
        f"[{i}] ({b['x']:.2f},{b['y']:.2f},{b['w']:.2f},{b['h']:.2f}) {b['text']}"
        for i, b in enumerate(blocks[:80])
    ])

    prompt_extract = f"""
Esta es una página de diario.

Tenés:
1) La imagen completa
2) Bloques de texto detectados con OCR (con coordenadas)

OCR:
{ocr_text}

Tareas:
- Identificá todas las noticias reales (ignorá publicidad)
- Uní correctamente bloques aunque estén en múltiples columnas
- Detectá títulos y cuerpos

Para cada noticia describí:
- título
- contenido
- sección
- bounding box aproximado
"""

    raw = call_ollama([
        {
            "role": "user",
            "content": prompt_extract,
            "images": [img_b64],
        }
    ])

    if not raw:
        print("RESPUESTA VACÍA STEP 1")
        return None

    print("RAW STEP 1:")
    print(raw[:500])

    structured = call_ollama([
        {
            "role": "user",
            "content": PROMPT_STRUCT + "\n\n" + raw,
        }
    ])

    if not structured:
        print("RESPUESTA VACÍA STEP 2")
        return None

    structured = structured.replace("```json", "").replace("```", "").strip()

    try:
        start = structured.find("{")
        end = structured.rfind("}") + 1
        return json.loads(structured[start:end])
    except (json.JSONDecodeError, ValueError):
        print("ERROR JSON:")
        print(structured[:500])
        return None