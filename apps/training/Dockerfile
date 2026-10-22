# Optional: container image for the PyTorch sampler in serve_mock.py.
# The talk also showed a Rust inference service. It is not part of this
# repository, and this image does not need it.
#
# Build context must be the REPO ROOT (needs src/pcrf + pyproject.toml):
#   docker build -f apps/training/Dockerfile -t pcrf-serve-mock .
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

COPY apps/training/serve_mock.py ./serve_mock.py

EXPOSE 50080
CMD ["uvicorn", "serve_mock:app", "--host", "0.0.0.0", "--port", "50080"]
