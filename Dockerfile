FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN pip install uv --quiet

# CUDA-enabled PyTorch — wheels include cuDNN, host just needs drivers + container toolkit
RUN uv pip install --system torch \
    --index-url https://download.pytorch.org/whl/cu128

RUN uv pip install --system \
    "sentence-transformers>=5.1.2" \
    mtcb \
    "chonkie[st,semantic,neural]" \
    datasets \
    pandas pyarrow boto3 python-dotenv numpy openai

WORKDIR /app
RUN mkdir -p results cache dataset

COPY benchmark.py .

ENTRYPOINT ["python", "benchmark.py"]
