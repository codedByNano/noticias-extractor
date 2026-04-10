python -m venv venv

.\venv\Scripts\Activate

pip install ollama pymupdf pillow tqdm

ollama pull llama3.2-vision

python main.py