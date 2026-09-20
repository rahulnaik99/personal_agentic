FROM python:3.11-slim

WORKDIR /srv

# unstructured[pdf] needs poppler/tesseract for hi_res PDF partitioning
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-common.txt ./
COPY rag_agent_service/requirements.txt ./rag_agent_service/requirements.txt
RUN pip install --no-cache-dir -r rag_agent_service/requirements.txt

COPY proto ./proto
COPY shared ./shared
COPY rag_agent_service ./rag_agent_service

ENV PYTHONPATH=/srv

# gRPC (agent tasks) on 50061, HTTP (ingestion) on 8001 — both started via
# scripts/entrypoint.sh so this one image serves both without two containers.
EXPOSE 50061 8001
COPY rag_agent_service/entrypoint.sh /srv/entrypoint.sh
RUN chmod +x /srv/entrypoint.sh
CMD ["/srv/entrypoint.sh"]
