FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake make && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY cpp ./cpp
RUN cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build/cpp -j2

FROM python:3.12-slim
WORKDIR /app
RUN useradd -m atlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=builder /app/build/cpp/atlassearch_bm25 /app/build/cpp/atlassearch_bm25
COPY atlassearch ./atlassearch
COPY api ./api
COPY data ./data
COPY scripts ./scripts
RUN PYTHONPATH=. python scripts/build_sample_corpus.py && chown -R atlas:atlas /app
USER atlas
ENV PYTHONPATH=/app PORT=8080
EXPOSE 8080
CMD ["python", "api/app.py"]
