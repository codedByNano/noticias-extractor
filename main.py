import ollama
import json
import io
import os
import glob
import re
from pdf2image import convert_from_path

MODEL_NAME = "extractor-diarios"
POPPLER_PATH = r"F:\Poppler\Library\bin"
INPUT_DIR = "data"
OUTPUT_DIR = "output"

PROMPT_SISTEMA = """
ERES UN ANALISTA DE MEDIOS GRÁFICOS. TRABAJAS SOBRE UN DIARIO COMPLETO.
IDENTIFICA Y EXTRAE CADA ARTÍCULO PERIODÍSTICO DE LA IMAGEN.

REGLAS:
1. TRANSCRIPCIÓN LITERAL Y COMPLETA. NO RESUMAS.
2. FORMATO: LISTA DE OBJETOS JSON.
3. CAMPOS: "titulo", "texto", "seccion", "estado".
4. CONTINUIDAD: 
   - Si la noticia sigue en otra página: "estado": "continua".
   - Si es el final de una noticia previa: "estado": "es_continuacion".

JSON: [{"titulo": "...", "texto": "...", "seccion": "...", "estado": "..."}]
"""

def slugify(text):
    if not text or not isinstance(text, str): return "sin-titulo"
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    return re.sub(r'[-\s]+', '-', text).strip('-')[:40]

def extraer_json_limpio(texto):
    if not texto: return []
    try:
        match = re.search(r'(\[.*\])', texto, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r'(\{.*\})', texto, re.DOTALL)
        if match: return [json.loads(match.group(1))]
    except:
        pass
    return []

def procesar_ejemplar(pdf_path):
    nombre = os.path.basename(pdf_path)
    print(f"\n--- {nombre} ---")
    
    try:
        paginas = convert_from_path(
            pdf_path, 
            dpi=200, 
            poppler_path=POPPLER_PATH if os.path.exists(POPPLER_PATH) else None
        )
    except Exception as e:
        print(f"Error Poppler: {e}")
        return

    mapa_noticias = {} 

    for i, pagina in enumerate(paginas):
        nro = i + 1
        print(f"Pág {nro}/{len(paginas)}...", end="\r")
        
        buf = io.BytesIO()
        pagina.save(buf, format='JPEG', quality=85)
        img_bytes = buf.getvalue()

        try:
            respuesta = ollama.generate(
                model=MODEL_NAME,
                prompt=f"{PROMPT_SISTEMA}\nPÁGINA {nro}.",
                images=[img_bytes],
                format="json",
                options={"temperature": 0, "num_ctx": 16384, "num_predict": -1}
            )

            noticias_extraidas = extraer_json_limpio(respuesta.get('response', ''))
            
            for n in noticias_extraidas:
                if not isinstance(n, dict): continue

                titulo = n.get("titulo", "Sin Título")
                cuerpo = n.get("texto", "")
                estado = n.get("estado", "completa")
                noticia_id = slugify(titulo)

                if noticia_id in mapa_noticias and (estado == "es_continuacion" or "continuacion" in titulo.lower()):
                    mapa_noticias[noticia_id]["texto"] += "\n\n" + str(cuerpo)
                    if nro not in mapa_noticias[noticia_id]["paginas"]:
                        mapa_noticias[noticia_id]["paginas"].append(nro)
                else:
                    n["paginas"] = [nro]
                    n["id_interno"] = noticia_id
                    mapa_noticias[noticia_id] = n

        except Exception as e:
            print(f"\nError Pág {nro}: {e}")

    final = list(mapa_noticias.values())
    out_file = os.path.join(OUTPUT_DIR, f"resultado_{os.path.splitext(nombre)[0]}.json")
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=4)
    
    print(f"\nFinalizado. Noticias: {len(final)}")

def main():
    pdfs = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    if not pdfs: return
    for p in pdfs:
        procesar_ejemplar(p)

if __name__ == "__main__":
    main()