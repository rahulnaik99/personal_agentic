"""
PDF loader — partitions a PDF into text/table/image elements using
`unstructured`. Tables are kept as whole HTML/markdown blocks (never
split element counts drop meaning); images are saved to disk and
captioned via a vision LLM call so they become searchable text.
"""
import os

from rag_agent_service.app.loaders.base import DocElement
from rag_agent_service.app.loaders.vision import caption_image
from shared.core.config import get_settings
from shared.core.logging import log_step


def load_pdf(file_path: str, source: str) -> list[DocElement]:
    from unstructured.partition.pdf import partition_pdf

    settings = get_settings()
    os.makedirs(settings.INGESTION_IMAGE_DIR, exist_ok=True)

    with log_step("load_pdf", agent="rag_agent", file_path=file_path) as ctx:
        raw_elements = partition_pdf(
            filename=file_path,
            strategy=settings.UNSTRUCTURED_PDF_STRATEGY,
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_output_dir=settings.INGESTION_IMAGE_DIR,
        )

        elements: list[DocElement] = []
        for el in raw_elements:
            category = getattr(el, "category", "")

            if category == "Table":
                # el.metadata.text_as_html gives a structured table representation
                # when available; fall back to plain text otherwise.
                table_text = getattr(el.metadata, "text_as_html", None) or str(el)
                elements.append(DocElement(type="table", text=table_text))

            elif category == "Image":
                image_path = getattr(el.metadata, "image_path", None)
                if image_path and os.path.exists(image_path):
                    caption = caption_image(image_path)
                else:
                    image_path = None
                    caption = str(el) or "[image: no extractable content]"
                elements.append(DocElement(type="image", text=caption, image_path=image_path))

            else:
                text = str(el).strip()
                if text:
                    elements.append(DocElement(type="text", text=text))

        ctx["output"] = {
            "num_elements": len(elements),
            "num_tables": sum(1 for e in elements if e.type == "table"),
            "num_images": sum(1 for e in elements if e.type == "image"),
        }

    return elements
