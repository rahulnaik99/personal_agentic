#!/usr/bin/env bash
# Regenerates Python gRPC stubs from proto/agent.proto into shared/generated/.
# Run this after any change to the .proto file.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m grpc_tools.protoc \
  -I proto \
  --python_out=shared/generated \
  --grpc_python_out=shared/generated \
  --pyi_out=shared/generated \
  proto/agent.proto

# protoc emits a flat `import agent_pb2` which breaks once these files live
# inside the shared.generated package — patch it to a relative import.
sed -i.bak 's/^import agent_pb2 as agent__pb2$/from . import agent_pb2 as agent__pb2/' \
  shared/generated/agent_pb2_grpc.py
rm -f shared/generated/agent_pb2_grpc.py.bak

echo "Regenerated shared/generated/agent_pb2*.py"
