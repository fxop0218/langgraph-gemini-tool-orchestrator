# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

# Seguridad/buenas prácticas
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependencias de sistema mínimas (xlsxwriter usa puro python; openpyxl opcional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requisitos primero para cacheo eficiente
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copia código
COPY . .

# Crear carpeta de salida de ficheros (también se montará volumen)
RUN mkdir -p /app/files /app/public

# Puertos y comando
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
