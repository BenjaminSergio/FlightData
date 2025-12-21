# Flask ML Wrapper - Integração FlightOnTime

API Flask que atua como camada de integração entre a API Java (Spring Boot) e o serviço ML externo.

## Instalação

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (Windows)
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

## Configuração

Crie um arquivo `.env` baseado em `.env.example`:

```bash
cp .env.example .env
```

Ajuste a variável `ML_SERVICE_URL` com a URL fornecida pela equipe de Data Science.

## Execução

```bash
# Desenvolvimento
python run.py

# Produção
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

## Endpoints

- `POST /predict` - Predição de atraso de voo
- `GET /health` - Health check

## Testes

```bash
pytest
```
