import ollama
import fitz
import os
import json
import base64
import threading
import time
import sys
from tqdm import tqdm

INPUT_PDF = "data/test.pdf"
OUTPUT_FOLDER = "output"
MODEL_NAME = "llama3.2-vision"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def animacion_proceso(stop_event):
    """Muestra un spinner y contador de tiempo mientras la IA trabaja."""
    spinner = ["|", "/", "-", "\\"]
    idx = 0
    inicio = time.time()
    while not stop_event.is_set():
        segundos = time.time() - inicio
        sys.stdout.write(f"\r{spinner[idx % 4]} Analizando noticias con Llama 3.2 Vision... [{segundos:.1f}s]")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)

    sys.stdout.write("\rAnálisis de IA finalizado.                    \n")

def procesar_pdf_a_json():
    if not os.path.exists(INPUT_PDF):
        print(f"Error: No se encuentra el archivo {INPUT_PDF}")
        return

    print(f"Abriendo PDF: {INPUT_PDF}...")
    doc = fitz.open(INPUT_PDF)
    
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("jpg")
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    prompt = """
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
    
    Estructura:
    {
      "noticias": [
        {
          "titulo": "string",
          "texto": "string completo",
          "seccion": "economia|politica|deportes|etc",
          "bounding_box": {"x": 0, "y": 0, "w": 0, "h": 0}
        }
      ]
    }
    No escribas nada más que el JSON.
    """

    stop_event = threading.Event()
    thread_animacion = threading.Thread(target=animacion_proceso, args=(stop_event,))
    
    print(f"Iniciando motor en GPU...")
    thread_animacion.start()

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            format='json',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64]
                }
            ]
        )

        stop_event.set()
        thread_animacion.join()

        resultado = response['message']['content']
        
        output_path = os.path.join(OUTPUT_FOLDER, "noticias_extraidas.json")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(resultado)
            
        print(f"Tarea completada. Archivo guardado en: {output_path}")

    except Exception as e:
        stop_event.set()
        print(f"\nError durante la inferencia: {e}")

if __name__ == "__main__":
    for _ in tqdm(range(100), desc="Cargando dependencias", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        time.sleep(0.005)
        
    procesar_pdf_a_json()