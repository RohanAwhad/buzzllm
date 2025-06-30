FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir ipython ipykernel jupyter_client numpy pandas matplotlib && \
    rm -rf /var/lib/apt/lists/*

COPY kernel_server.py /kernel_server.py

EXPOSE 8787

ENTRYPOINT ["python", "/kernel_server.py"]
