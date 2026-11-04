# syntax=docker/dockerfile:1
#
# One image for x86-64 and ARM64 (Apple silicon, AWS Graviton, ...): pip picks the TensorFlow wheel
# for the machine Docker builds on, so there is no separate Dockerfile per CPU any more.
FROM python:3.12-slim

# Optional parts of the package (see src/detector/pyproject.toml):
#   export  the ONNX export for the web demo (default)
#   gpu     TensorFlow with the CUDA libraries, for NVIDIA GPUs on x86-64: --build-arg EXTRAS=export,gpu
#   test    pytest, to run the test suite in the container
ARG EXTRAS=export

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app
COPY src/detector src/detector
# Editable install: docker-compose.yml mounts src/detector over the copy, so code changes need no rebuild.
# The pip cache survives rebuilds, so TensorFlow is downloaded once.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e "./src/detector[${EXTRAS}]"

# Relative output paths (like the comparison figure ./results.png) land here.
WORKDIR /outputs
ENTRYPOINT ["blur-detector"]
CMD ["run"]
