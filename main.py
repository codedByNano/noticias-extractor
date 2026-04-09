import os
import sys
import ctypes

cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"

print(f"--- Verificando entorno ---")
if os.path.exists(cuda_bin):
    print(f"✅ Carpeta CUDA encontrada.")
    os.add_dll_directory(cuda_bin)
else:
    print(f"❌ ERROR: No se encuentra la carpeta en {cuda_bin}. Revisa la versión.")

# Intentar cargar una DLL de CUDA manualmente para forzar el error real si falta algo
try:
    ctypes.CDLL(os.path.join(cuda_bin, "cudart64_13.dll"))
    print("✅ DLLs de CUDA cargadas correctamente.")
except Exception as e:
    print(f"❌ Error cargando dependencias de NVIDIA CUDA: {e}")

print(f"---------------------------\n")

import json
import base64
import fitz
from io import BytesIO
from PIL import Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llama32VisionChatHandler

MODEL_PATH = "model/Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf"
MMPROJ_PATH = "model/Llama-3.2-11B-Vision-Instruct-mmproj.f16.gguf"
INPUT_PDF = "data/test.pdf"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def pdf_to_base64(path, page_num=0):
    """Convierte una página de PDF a imagen base64 para la IA."""
    doc = fitz.open(path)
    page = doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

chat_handler = Llama32VisionChatHandler(clip_model_path=MMPROJ_PATH)

print("Cargando modelo en la GPU")
llm = Llama(
    model_path=MODEL_PATH,
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=4096,
    temp=0.1,
    verbose=False
)

SYSTEM_PROMPT = """
Quiero que identifiques cada noticia y devuelvas SOLO un JSON con una lista de objetos que contengan los siguientes campos:
        Omitir avisos o piezas publicitarias, sólo nos interesa el contenido periodístico. 
         Algunas pautas para localizar artículos/noticias completas:
            - El titular de un artículo suele tener cuerpo de tipografía más grande que el resto y puede abarcar una o varias columnas.
            - La volanta de un artículo suele tener cuerpo de tipografía más pequeño que el título y aparecer encima del titular.
            - La bajada de un artículo suele tener cuerpo de tipografía más pequeño que el título, mas grande que el texto del cuerpo y aparece debajo del titular.
            - El epigrafe de un artículo suele tener cuerpo de tipografía más pequeño que el texto del cuerpo y aparece debajo de las fotos o imágenes incrustadas dentro del bloque de la noticia.
            - Los destacados son pequeños párrafos con una tipografía o tomaño distinto al cuerpo del texto, generalemente insertados en el cuerpo de la noticia, puede haber varios.
            - Los complementarios son pequeñas cajas de texto que aparecen en el cuerpo de la noticia, generalemente insertados en el cuerpo de la noticia, pueden tener un título y pueden haber varios.
            - En algunos casos puede aparecer el autor de la noticia si podes identificarlo.
            - Los medios gráficos suelen organizar su diseño en columnas. 
            - No necesariamente el titular abarca todas las columnas del artículo.
            - las fotos generalmente ocupan un número entero de columnas: una, dos, tres, etc. 
            - la noticia puede incluir elementos visuales adicionales tales como una foto con su correspondiente epígrafe, alguna infografía o tabla, etc. 
        Entonces, cada noticia debe tener:
            El título.
            Un cuerpo de texto (puede estar en varias columnas).
            Una categorización del contenido entre las siguientes posibilidades: economia, politica, policiales, finanzas, espectaculos, deportes, etc. Si no hay match evidente con ninguna de las categorías enumeradas anteriormente, simplemente categorizar como "otra".
            Una caja delimitadora (bounding box) que encierre toda la noticia.
            Cada objeto debe incluir:
            - titulo (string)
            - texto (string)
            - seccion (string)
            - bounding_box (objeto con x, y, ancho, alto)
            Adicionalmente puede incluir:
            - volanta (string)
            - bajada (string)
            - epigrafes (array de strings)
            - destacados (array de strings)
            - complementarios (array de objetos con titulo y texto)
            - autor (string)
"""


def procesar():
    if not os.path.exists(INPUT_PDF):
        print(f"Error: No se encuentra {INPUT_PDF}")
        return

    print(f"Procesando {INPUT_PDF}...")
    img_b64 = pdf_to_base64(INPUT_PDF)

    print("Iniciando análisis")
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extrae las noticias de esta página según el formato JSON solicitado."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        response_format={"type": "json_object"},
        stop=["<|im_start|>", "<|im_end|>", "<|end_of_text|>"]
    )

    # 3. Guardar el resultado
    json_output = response["choices"][0]["message"]["content"]
    
    file_name = os.path.basename(INPUT_PDF).replace(".pdf", ".json")
    output_path = os.path.join(OUTPUT_FOLDER, f"resultado_{file_name}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_output)
    
    print(f"Trabajo Completado. JSON generado: {output_path}")

if __name__ == "__main__":
    procesar()