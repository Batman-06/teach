# Build Stage
FROM python:3.12-slim as build

WORKDIR /us-oa-custom-sources-model
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime Stage
FROM python:3.12-slim as runtime

WORKDIR /us-oa-custom-sources-model

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /us-oa-custom-sources-model /us-oa-custom-sources-model

# The consumer's own status endpoint. Deliberately the SAME port the HTTP server
# used, so the K8s Service and containerPort do not change — only the probe PATH
# has to move to /health in CLEM-AI-DevOps-K8s-Manifest/jpn-oa-dc-optimised.
EXPOSE 8008

# No web server any more. The process polls its SQS lane queue, runs the
# optimisation, publishes the result, and deletes the message only after that.
# Concurrency is CLEM_WORKERS (processes), not gunicorn workers.
CMD ["python3", "consumer_adapter.py"]
