# Run worker image: one evaluation run per container (MOD-135). See docker/worker.py.
#
# Batch instances are x86, so build for linux/amd64 (needed on Apple Silicon):
#   docker build --platform linux/amd64 -t mme-run-worker .

FROM public.ecr.aws/docker/library/python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# The package and its dependencies, then boto3 for the entrypoint's S3 calls.
COPY pyproject.toml ./
COPY src ./src
RUN pip install . boto3

# The CLI reads catalog/catalog.yaml relative to the working directory.
COPY catalog ./catalog
COPY docker/worker.py ./worker.py

# Run as a non-root user; /work holds the run's records and report until the worker deletes them.
RUN useradd --create-home --uid 1000 worker && mkdir /work && chown worker /work
USER worker

CMD ["python", "/app/worker.py"]
