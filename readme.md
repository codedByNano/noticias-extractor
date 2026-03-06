El extractor no es un simple "lector de texto", es un **Pipeline de Visión Artificial** que funciona en 4 capas:

1. **Capa de Ingesta (PDF Engine):** - Transforma el PDF (vectorial/texto) en una imagen (ráster) a 300 DPI.

   - _Razón:_ Los diarios usan layouts complejos que el texto plano de un PDF no puede representar. La IA necesita "ver" el diseño para entender qué es una noticia y qué es un aviso.

2. **Capa de Visión (Ollama):**

   - El modelo analiza la jerarquía visual: tamaño de fuente (Titulares), peso (Negritas) y proximidad (Cuerpo).
   - _Capacidad Multimodal:_ Detecta las coordenadas (x, y, w, h) de cada bloque.

3. **Capa de Estructuración (JSON Schema):**

   - Se fuerza al modelo a responder en formato JSON puro.
   - Filtra automáticamente el contenido no periodístico (publicidad, avisos fúnebres, etc.) basado en el contexto.

4. **Capa de Salida:**
   - Exportación a archivo `.json` para consumo por bases de datos o front-ends.

## Requisitos mínimos del Sistema

- **CPU:** 8 nucleos 16 hilos - Soporte AVX-512 (Intel Core i7(12va+) / AMD Ryzen 7 5800+)
- **RAM:** 24GB DDR5 5600mhz 36CL
- **GPU:** 16GB VRAM (RTX 4070 Ti Super - AMD RADEON **NO** COMPATIBLE)

## Guía de Instalación

1. Dependencias de Sistema

- Para que Python pueda "dibujar" el PDF necesita **Poppler**. [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/).
  Extrae y Agrega `\poppler\Library\bin` a tus Variables de Entorno (PATH).

- Versión de Python: 3.11.9 (recomendado)

2. Dependencias de Sistema

- Instala [Ollama](https://ollama.com)

- Abrir una terminal en la raíz del proyecto y ejecutar:

```bash
cd model
ollama create extractor-diarios -f models/Modelfile
```

para cargar el modelo localmente

- Cargar el entorno de phyton **volviendo a la carpeta raiz**:

```bash
py -3.11 -m venv venv
.\venv\Scripts\activate
pip install ollama pdf2image pillow
```

## Uso

1. **Cargar el PDF:** Colocar el archivo dentro de la carpeta `data/`.
2. **Inicializar el entorno**

```bash
   .\venv\Scripts\activate
```

3. **Lanzar el script:**

```bash
   python scripts/main.py
```

## BLoque de salida esperado (ejemplo con test.pdf)

El sistema devolverá una estructura similar a esta por cada noticia detectada:

```json
{
  "titulo": "Con fuerte tono político, Llaryora abrió las sesiones",
  "seccion": "politica",
  "texto": "El mandatario apuntó, sin nombrarlo, a la injerencia de Juez...",
  "bounding_box": {
    "x": 150,
    "y": 200,
    "ancho": 450,
    "alto": 300
  },
  "volanta": "148° Período Legislativo",
  "autor": "Redacción Hoy Día"
}
```
