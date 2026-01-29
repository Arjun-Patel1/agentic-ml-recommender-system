FROM python:3.10-slim

# ---------- System deps ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Prevent BLAS thread explosion ----------
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# ---------- Working dir ----------
WORKDIR /app

# ---------- Install Python deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy project ----------
COPY . .

# ---------- Default command ----------
CMD ["python", "experiments/run_agentic_graph.py"]
