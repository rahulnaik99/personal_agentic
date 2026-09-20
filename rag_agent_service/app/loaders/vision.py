"""
Captions an image using a vision-capable chat model (Claude/GPT-4o both
support multimodal input via LangChain's content-block message format).
The caption becomes the searchable text for that image's child chunk.
"""
import base64

from langchain_core.messages import HumanMessage

from shared.services.chat_llm import get_chat_llm

CAPTION_PROMPT = (
    "Describe this image in 2-4 sentences, focused on any factual content "
    "(data, labels, diagrams, text visible in the image) that would help "
    "someone find this image via a text search. Be concise and literal."
)


def caption_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    llm = get_chat_llm(temperature=0.0)  # uses configured provider/model; must be vision-capable
    message = HumanMessage(content=[
        {"type": "text", "text": CAPTION_PROMPT},
        {
            "type": "image",
            "source_type": "base64",
            "data": image_b64,
            "mime_type": "image/png",
        },
    ])
    try:
        response = llm.invoke([message])
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        # Vision call failing (e.g. non-vision model configured) shouldn't
        # break ingestion — fall back to a placeholder so the image chunk
        # still exists (just not semantically searchable).
        return f"[image: caption unavailable — {exc}]"
