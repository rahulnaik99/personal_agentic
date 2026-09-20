FROM python:3.11-slim

WORKDIR /srv

COPY requirements-common.txt ./
COPY tool_agent_service/requirements.txt ./tool_agent_service/requirements.txt
RUN pip install --no-cache-dir -r tool_agent_service/requirements.txt

COPY proto ./proto
COPY shared ./shared
COPY tool_agent_service ./tool_agent_service

ENV PYTHONPATH=/srv
EXPOSE 50062

CMD ["python", "-m", "tool_agent_service.app.server"]
