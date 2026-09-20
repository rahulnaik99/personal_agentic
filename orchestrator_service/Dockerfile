FROM python:3.11-slim

WORKDIR /srv

COPY requirements-common.txt ./
COPY orchestrator_service/requirements.txt ./orchestrator_service/requirements.txt
RUN pip install --no-cache-dir -r orchestrator_service/requirements.txt

COPY proto ./proto
COPY shared ./shared
COPY orchestrator_service ./orchestrator_service

ENV PYTHONPATH=/srv
EXPOSE 8000

CMD ["uvicorn", "orchestrator_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
