FROM python:3.9-slim

WORKDIR /app

# Instala dependências do sistema para TensorFlow
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Primeiro copia apenas o requirements.txt
COPY ./app/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Depois copia o resto da aplicação
COPY ./app /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]