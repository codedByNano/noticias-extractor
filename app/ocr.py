from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)

def extract_blocks(image_path):
    doc = DocumentFile.from_images(image_path)
    result = model(doc)

    blocks_out = []

    for page in result.export()["pages"]:
        for block in page["blocks"]:
            text = []
            x1s, y1s, x2s, y2s = [], [], [], []

            for line in block["lines"]:
                words = [w["value"] for w in line["words"]]
                line_text = " ".join(words).strip()

                if line_text:
                    text.append(line_text)

                (x1, y1), (x2, y2) = line["geometry"]

                x1s.append(x1)
                y1s.append(y1)
                x2s.append(x2)
                y2s.append(y2)

            if not text:
                continue

            blocks_out.append({
                "text": " ".join(text),
                "x": min(x1s),
                "y": min(y1s),
                "w": max(x2s) - min(x1s),
                "h": max(y2s) - min(y1s),
            })

    return blocks_out