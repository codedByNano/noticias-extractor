from preprocess import pdf_to_images
from vlm import run_vlm


if __name__ == "__main__":
    paths = pdf_to_images("data/raw/test.pdf")

    all_news = []

    for path in paths:
        print(f"Procesando: {path}")

        result = run_vlm(path)

        if not result:
            continue

        noticias = result.get("noticias", [])
        all_news.extend(noticias)

    print("\n====================")
    print(f"TOTAL NOTICIAS: {len(all_news)}")

    for n in all_news[:5]:
        print("\n====================")
        print("TITULO:", n.get("titulo"))
        print("TEXTO:", (n.get("texto") or "")[:200])