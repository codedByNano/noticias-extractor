import ollama
import json
import io
import os
import glob
import re
import gc
import time
import subprocess
from pdf2image import convert_from_path

# Configuración de constantes y rutas del sistema
MODEL_NAME = "extractor-diarios"
POPPLER_PATH = r"F:\Poppler\Library\bin"
INPUT_DIR = "data"
OUTPUT_DIR = "output"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp_pages")

# Definición del Prompt de Sistema para el modelo de visión
PROMPT_SISTEMA = """
ACTÚA COMO UN EXPERTO EN DIAGRAMACIÓN DE MEDIOS IMPRESOS.
OBJETIVO: ANALIZAR LA IMAGEN DE UNA PÁGINA DE DIARIO Y EXTRAER LA INFORMACIÓN PERIODÍSTICA.

REGLAS DE EXTRACCIÓN:
1. EXCLUIR: ANUNCIOS, BANNERS, AVISOS CLASIFICADOS Y FÚNEBRES.
2. CAMPOS OBLIGATORIOS: TITULO, CUERPO (TEXTO), SECCIÓN.
3. CAMPOS ADICIONALES: VOLANTA, BAJADA, EPÍGRAFES, DESTACADOS Y AUTOR (SI ESTÁN PRESENTES).
4. REQUISITO: TRANSCRIPCIÓN LITERAL Y COMPLETA. PROHIBIDO RESUMIR.
5. FORMATO DE SALIDA: LISTA DE OBJETOS JSON.

GESTIÓN DE CONTINUIDAD:
- "estado": "continua" (si la noticia prosigue en otra página).
- "estado": "es_continuacion" (si la noticia es remanente de una página previa).
- "estado": "completa" (caso por defecto).

JSON SCHEMA: [{"titulo": "string", "volanta": "string", "bajada": "string", "texto": "string", "autor": "string", "seccion": "string", "estado": "string"}]
"""

def slugify(text):
    """Genera un identificador único normalizado para cada noticia."""
    if not text or not isinstance(text, str):
        return "sin-titulo"
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    return re.sub(r'[-\s]+', '-', text).strip('-')[:40]

def extraer_json_limpio(texto):
    """Parsea la respuesta del modelo para extraer estructuras JSON válidas."""
    if not texto:
        return []
    try:
        match = re.search(r'(\[.*\])', texto, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r'(\{.*\})', texto, re.DOTALL)
        if match:
            return [json.loads(match.group(1))]
    except (json.JSONDecodeError, AttributeError):
        pass
    return []

def liberar_vram_total():
    """Finaliza el proceso del modelo en Ollama para liberar memoria de video."""
    try:
        subprocess.run(["ollama", "stop", MODEL_NAME], capture_output=True, check=False)
        time.sleep(3)
        gc.collect()
    except Exception:
        pass

def unificar_resultados(nombre_base):
    """Consolida los archivos temporales de páginas individuales en un único reporte."""
    mapa_noticias = {}
    archivos_temp = glob.glob(os.path.join(TEMP_DIR, f"{nombre_base}_p*.json"))
    archivos_temp.sort(key=lambda x: int(re.search(r'_p(\d+)\.json', x).group(1)))

    for archivo in archivos_temp:
        with open(archivo, "r", encoding="utf-8") as f:
            try:
                data_json = json.load(f)
                nro_pag = int(re.search(r'_p(\d+)\.json', archivo).group(1))
                for n in data_json:
                    if not isinstance(n, dict):
                        continue
                    
                    titulo = n.get("titulo", "Sin Titulo")
                    noticia_id = slugify(titulo)
                    
                    if noticia_id in mapa_noticias and (n.get("estado") == "es_continuacion" or "continuacion" in titulo.lower()):
                        mapa_noticias[noticia_id]["texto"] += "\n\n" + str(n.get("texto", ""))
                        if nro_pag not in mapa_noticias[noticia_id].get("paginas", []):
                            mapa_noticias[noticia_id].setdefault("paginas", []).append(nro_pag)
                    else:
                        n["paginas"] = [nro_pag]
                        mapa_noticias[noticia_id] = n
            except (json.JSONDecodeError, IOError):
                continue
                
    return list(mapa_noticias.values())

def procesar_ejemplar(pdf_path):
    """Ejecuta el pipeline de procesamiento para un archivo PDF completo."""
    nombre_base = os.path.splitext(os.path.basename(pdf_path))[0]
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    print(f"Iniciando procesamiento: {nombre_base}")
    liberar_vram_total()

    try:
        # Configuración de resolución optimizada para OCR
        paginas = convert_from_path(pdf_path, dpi=150, poppler_path=POPPLER_PATH)
    except Exception as e:
        print(f"Error en conversión de PDF: {e}")
        return

    for i, pagina in enumerate(paginas):
        nro = i + 1
        temp_file = os.path.join(TEMP_DIR, f"{nombre_base}_p{nro}.json")
        
        if os.path.exists(temp_file):
            continue

        if nro > 1 and nro % 2 == 0:
            liberar_vram_total()

        print(f"Procesando página {nro}/{len(paginas)}...", end="\r")
        
        buf = io.BytesIO()
        pagina.save(buf, format='JPEG', quality=85)
        img_bytes = buf.getvalue()
        buf.close()

        try:
            res = ollama.generate(
                model=MODEL_NAME,
                prompt=f"{PROMPT_SISTEMA}\nPAGINA {nro}.",
                images=[img_bytes],
                format="json",
                options={
                    "temperature": 0, 
                    "num_ctx": 8192,
                    "num_predict": -1,
                    "low_vram": True
                }
            )

            data = extraer_json_limpio(res.get('response', ''))
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

            del img_bytes
            gc.collect()

        except Exception as e:
            print(f"\nError en ejecución de modelo (Página {nro}): {e}")
            liberar_vram_total()
            break

    resultado_final = unificar_resultados(nombre_base)
    out_file = os.path.join(OUTPUT_DIR, f"resultado_{nombre_base}.json")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(resultado_final, f, ensure_ascii=False, indent=4)
    
    print(f"\nProceso finalizado. Salida en: {out_file}")

def main():
    """Punto de entrada principal del script."""
    if not os.path.exists(INPUT_DIR):
        print(f"Error: No se encuentra el directorio de entrada {INPUT_DIR}")
        return
        
    pdfs = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    if not pdfs:
        print("No se detectaron archivos PDF para procesar.")
        return
        
    for p in pdfs:
        procesar_ejemplar(p)

if __name__ == "__main__":
    main()