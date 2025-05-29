# Previsão de Preços de Ações com LSTM

Projeto desenvolvido para o Tech Challenge de Pós-Graduação em Machine Learning, utilizando redes neurais LSTM para previsão do preço de fechamento de ações.
Inclui treinamento do modelo, exportação dos artefatos e disponibilização via API RESTful (FastAPI).

## Sumário

* [Pré-requisitos](#pré-requisitos)
* [Instalação](#instalação)
* [Treinamento do Modelo](#treinamento-do-modelo)
* [API para Previsão](#api-para-previsão)
* [Testando a API](#testando-a-api)
* [Monitoramento e Métricas](#monitoramento-e-métricas)
* [Docker e Docker-Compose](#docker-e-docker-compose)
* [Estrutura do Projeto](#estrutura-do-projeto)
* [Autores](#autores)

---

## Pré-requisitos

* Python 3.8 ou superior instalado
* Pip atualizado (`python -m pip install --upgrade pip`)

## Instalação

Clone o repositório e entre na pasta do projeto:

```bash
git clone https://github.com/vagnerfabiog/fiaptech4.git
cd fiaptech4
```

Crie e ative um ambiente virtual (opcional, recomendado):

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

Instale todas as dependências necessárias:

```bash
pip install -r requirements.txt
```

---

## Treinamento do Modelo

O script `modelo.py` baixa os dados históricos, treina o modelo, avalia e salva os artefatos.

Para treinar ou re-treinar o modelo:

```bash
python modelo.py
```

Após executar, serão criados na pasta `model_artifacts`:

* `lstm_model.h5` — modelo treinado (rede neural)
* `scaler.pkl` — normalizador dos dados
* `metadata.json` — informações de configuração e métricas

---

## API para Previsão

O arquivo `api.py` disponibiliza o modelo treinado através de uma API FastAPI.

Para executar a API localmente:

```bash
uvicorn api:app --reload
```

Acesse a documentação interativa em:

`http://127.0.0.1:8000/docs`

---

## Testando a API

Você pode usar o Swagger UI em `/docs`, Postman ou curl. Exemplo de requisição curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
-d '{
  "historical_prices": [/* lista de 60 valores */],
  "future_steps": 3
}'
```

Ou por script Python:

```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "historical_prices": [/* 60 valores */],
    "future_steps": 2
}
resp = requests.post(url, json=payload)
print(resp.json())
```

---

## Monitoramento e Métricas

* Métricas Prometheus expostas em `/metrics` para integração com Grafana.
* Rota de health check em `/health`.
* Cada resposta de predição retorna `processing_time`.
* Logs configurados para auditoria e depuração.

---

## Docker e Docker-Compose

Você pode executar toda a stack com Docker:

1. Construir a imagem:

   ```bash
   docker build -t fiaptech4-api .
   ```
2. Subir via Docker Compose:

   ```bash
   docker-compose up -d
   ```

Isso vai iniciar:

* **api** (FastAPI) em `localhost:8000`
* **Prometheus** em `localhost:9090`

---

## Estrutura do Projeto

```
fiaptech4/
├── api.py               # API FastAPI
├── modelo.py           # Script de treinamento e salvamento do modelo
├── requirements.txt     # Dependências Python
├── Dockerfile           # Imagem da API
├── docker-compose.yml   # Orquestração (API + Prometheus)
├── prometheus.yml       # Configurações do Prometheus
├── readme.md            # Este arquivo de documentação
└── model_artifacts/     # Artefatos gerados pelo treinamento
    ├── lstm_model.h5
    ├── scaler.pkl
    └── metadata.json
```

---

## Autores

* Vagner Fabio e Diego Varela (FIAP Tech Challenge Fase 4)

---
