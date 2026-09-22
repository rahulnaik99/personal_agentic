"""
gRPC server for the RAG agent — implements AgentService (GetAgentCard,
SendTask, StreamTask) per proto/agent.proto. This is the network
boundary; all actual logic lives in agent.py.
"""
import json
import logging
import time
from concurrent import futures
from pathlib import Path

import grpc

from rag_agent_service.app.agent import run_rag_agent
from shared.core.config import get_settings
from shared.core.logging import (
    configure_logging,
    set_trace_id,
    start_trace_collector,
)
from shared.generated import agent_pb2, agent_pb2_grpc

AGENT_CARD_PATH = Path(__file__).parent.parent / "agent_card.json"


class RagAgentServicer(agent_pb2_grpc.AgentServiceServicer):
    def __init__(self):
        with open(AGENT_CARD_PATH) as f:
            self._card_dict = json.load(f)

    def GetAgentCard(self, request, context):
        return agent_pb2.AgentCard(
            name=self._card_dict["name"],
            version=self._card_dict["version"],
            description=self._card_dict["description"],
            skills=[
                agent_pb2.Skill(id=s["id"], name=s["name"], description=s["description"])
                for s in self._card_dict["skills"]
            ],
            input_modes=self._card_dict["input_modes"],
            output_modes=self._card_dict["output_modes"],
        )

    def SendTask(self, request, context):
        set_trace_id(request.trace_id or request.task_id)
        collector = start_trace_collector()

        forced_category = request.context.get("category") or None
        model_provider = request.context.get("model_provider") or None
        model_override = {"provider": model_provider, "model": request.context.get("model_name") or None} \
            if model_provider else None
        try:
            conversation_history = json.loads(request.context.get("conversation_history", "[]"))
        except json.JSONDecodeError:
            conversation_history = []
        try:
            result = run_rag_agent(
                query=request.query, trace_id=request.trace_id,
                forced_category=forced_category, model_override=model_override,
                conversation_history=conversation_history,
            )
        except Exception as exc:
            logging.getLogger("rag_agent.server").exception("SendTask failed")
            return agent_pb2.SendTaskResponse(
                task_id=request.task_id, state=agent_pb2.TASK_STATE_FAILED, error=str(exc),
            )

        if result["blocked"]:
            return agent_pb2.SendTaskResponse(
                task_id=request.task_id, state=agent_pb2.TASK_STATE_FAILED,
                error=result["block_reason"] or "blocked by guardrails",
            )

        usage = [
            agent_pb2.UsageMetadata(input_tokens=s.input_tokens, output_tokens=s.output_tokens, model=s.model or "")
            for s in collector.steps if s.model
        ]

        return agent_pb2.SendTaskResponse(
            task_id=request.task_id,
            state=agent_pb2.TASK_STATE_COMPLETED,
            result_text=result["answer"],
            sufficient=result["sufficient"],
            metadata={
                "ragas_scores": json.dumps(result["ragas_scores"]),
                "guardrail_flags": json.dumps(result["guardrail_flags"]),
                "contexts": json.dumps(result["contexts"]),
                "category_used": result["category_used"] or "",
            },
            usage=usage,
        )

    def StreamTask(self, request, context):
        set_trace_id(request.trace_id or request.task_id)
        yield agent_pb2.TaskEvent(
            task_id=request.task_id, state=agent_pb2.TASK_STATE_WORKING,
            message="Retrieving context and generating answer...",
            timestamp_ms=int(time.time() * 1000),
        )
        response = self.SendTask(request, context)
        yield agent_pb2.TaskEvent(
            task_id=request.task_id,
            state=agent_pb2.TASK_STATE_COMPLETED if response.state == agent_pb2.TASK_STATE_COMPLETED else agent_pb2.TASK_STATE_FAILED,
            message=response.result_text or response.error,
            timestamp_ms=int(time.time() * 1000),
        )


def serve():
    configure_logging()
    settings = get_settings()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_pb2_grpc.add_AgentServiceServicer_to_server(RagAgentServicer(), server)
    address = f"{settings.RAG_AGENT_GRPC_HOST}:{settings.RAG_AGENT_GRPC_PORT}"
    server.add_insecure_port(address)
    server.start()
    logging.getLogger("rag_agent.server").info(f"RAG agent gRPC server listening on {address}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
