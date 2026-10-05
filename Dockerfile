FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir tokenizers numpy fastapi "uvicorn[standard]" pydantic
COPY loopthink/ loopthink/
COPY serve/ serve/
# put a trained run folder (model.pt + tokenizer.json) at ./model before building
COPY model/ model/
ENV MODEL_DIR=/app/model TAU=0.5 MAX_LOOPS=16
EXPOSE 8000
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
