# FlightOnTime - Arquitetura e Fluxo de Dados

> **📚 Documentação Completa:**
> - **[README.md](README.md)** - Arquitetura e Fluxo de Dados (você está aqui)
> - **[Docker.md](Docker.md)** - Guia de Docker e Containers

## 📋 Visão Geral

O **FlightOnTime** é um sistema de previsão de atrasos de voos baseado em Machine Learning, composto por três camadas:

1. **API Java (Spring Boot)** - Backend principal que expõe endpoints REST
2. **Flask ML Wrapper (Python)** - Camada de integração/adapter entre Java e ML
3. **ML Service** - Serviço de Machine Learning (real ou mock)

## 🏗️ Arquitetura


```
Cliente (Postman, Frontend, etc)
  |
  | HTTP POST /api/v1/predict
  v
API Java (Spring Boot)
  Container: fot-api | Porta: 8080
    ├─ Controller Layer
    ├─ Service Layer
    └─ ML Client (RestTemplate)
      |
      | HTTP POST http://ml-wrapper:5000/predict
      v
  Flask ML Wrapper (Python)
    Container: ml-wrapper | Porta: 5000
      ├─ Flask Routes
      └─ ML Service Client
            |
            | HTTP POST http://ml-service:8000/predict
            v
        ML Service
          Container: ml-service-mock (dev) | Porta: 8000
            └─ Modelo de Machine Learning (RandomForest, XGBoost, etc)
```

## 🔄 Fluxo de Dados Detalhado

### 1. Requisição do Cliente

**Endpoint:** `POST http://localhost:8080/api/v1/predict`

**Request Body:**
```json
{
  "flightNumber": "AA1234",
  "companyName": "AA",
  "flightOrigin": "JFK",
  "flightDestination": "LAX",
  "flightDepartureDate": "2025-12-25T14:30:00",
  "flightDistance": 3974
}
```

### 2. Processamento na API Java

#### 2.1 Controller Layer
- **Classe:** `PredictionController`
- **Ação:** Recebe requisição HTTP
- **Validação:** Bean Validation (@Valid)
- **Log:** Registra recebimento da requisição

```java
@PostMapping("/predict")
public ResponseEntity<FlightPredictionResponseDTO> predict(
    @Valid @RequestBody FlightPredictionRequestDTO request)
```

#### 2.2 Service Layer
- **Classe:** `PredictionServiceImpl`
- **Ação:** Orquestra a lógica de negócio
- **Processo:** 
  1. Valida dados de entrada
  2. Chama MLServiceClient
  3. Processa resposta do ML
  4. Calcula nível de confiança

```java
public FlightPredictionResponseDTO predictDelay(
    FlightPredictionRequestDTO request)
```

#### 2.3 Client Layer
- **Classe:** `MLServiceClient`
- **Ação:** Comunicação HTTP com Flask Wrapper
- **Tecnologia:** RestTemplate
- **Configuração:** Timeout de 5 segundos

```java
ResponseEntity<MLServiceResponseDTO> response = 
    restTemplate.postForEntity(mlServiceUrl, entity, 
                               MLServiceResponseDTO.class);
```

**Dados Enviados para Flask:**
```json
{
  "flightNumber": "AA1234",
  "companyName": "AA",
  "flightOrigin": "JFK",
  "flightDestination": "LAX",
  "flightDepartureDate": "2025-12-25T14:30:00",
  "flightDistance": 3974
}
```

### 3. Processamento no Flask Wrapper

#### 3.1 Flask Route
- **Arquivo:** `app/routes/prediction_routes.py`
- **Endpoint:** `/predict`
- **Ação:** Recebe dados do Java

#### 3.2 Validação
- **Tecnologia:** Pydantic
- **Ação:** Valida formato e tipos de dados
- **Conversão:** Normaliza códigos (uppercase)

```python
class FlightPredictionRequest(BaseModel):
    flightNumber: str
    companyName: str
    flightOrigin: str
    flightDestination: str
    flightDepartureDate: str
    flightDistance: int
```

#### 3.3 ML Service Client
- **Arquivo:** `app/services/ml_client.py`
- **Ação:** Envia requisição para ML Service
- **Tecnologia:** requests library
- **Timeout:** 30 segundos

```python
response = requests.post(
    self.ml_service_url,
    json=flight_data,
    timeout=self.timeout
)
```

#### 3.4 Mapeamento de Resposta
- **Conversão:** `probability` → `confidence`
- **Motivo:** Compatibilidade com contrato da API Java

```python
response = {
    "prediction": ml_result.get("prediction"),
    "confidence": ml_result.get("probability")  # Mapeamento
}
```

### 4. Processamento no ML Service

#### 4.1 Recepção de Dados
- **Arquivo:** `mock_ml_service.py` (desenvolvimento)
- **Ação:** Recebe dados de voo

#### 4.2 Inferência
- **Mock:** Gera predição aleatória para testes
- **Real:** Aplica modelo treinado (RandomForest, etc.)

#### 4.3 Resposta
```json
{
  "prediction": 1,        // 0 = ON_TIME, 1 = DELAYED
  "probability": 0.91     // Confiança 0.0 - 1.0
}
```

### 5. Retorno para Flask Wrapper

**Flask recebe do ML Service:**
```json
{
  "prediction": 1,
  "probability": 0.91
}
```

**Flask transforma para Java:**
```json
{
  "prediction": 1,
  "confidence": 0.91
}
```

### 6. Retorno para API Java

**Java recebe do Flask:**
```json
{
  "prediction": 1,
  "confidence": 0.91
}
```

**Java processa:**
1. Converte `prediction` (int) → `FlightPrediction` (enum)
2. Calcula nível de confiança:
   - ≥ 0.90: VERY_HIGH
   - ≥ 0.75: HIGH
   - ≥ 0.60: MEDIUM
   - ≥ 0.45: LOW
   - < 0.45: VERY_LOW

### 7. Resposta Final ao Cliente

```json
{
  "prediction": "DELAYED",
  "probability": 0.91,
  "confidence": "VERY_HIGH",
  "probabilityPercentage": 91,
  "formattedProbability": "91.00%",
  "confidenceLevel": "VERY_HIGH",
  "highConfidence": true,
  "lowConfidence": false,
  "summary": "Flight is predicted to be DELAYED with VERY_HIGH confidence (91.00%)"
}
```

## 🐳 Containers e Comunicação

### Rede Docker

Todos os containers estão na mesma rede: `fot-network`

```yaml
networks:
  fot-network:
    driver: bridge
```

### Resolução de Nomes

- **fot-api** → **ml-wrapper**: `http://ml-wrapper:5000/predict`
- **ml-wrapper** → **ml-service**: `http://ml-service:8000/predict`

### Portas Expostas

| Serviço | Porta Interna | Porta Externa | Acesso |
|---------|---------------|---------------|--------|
| fot-api | 8080 | 8080 | http://localhost:8080 |
| ml-wrapper | 5000 | 5000 | http://localhost:5000 |
| ml-service | 8000 | 8000 | http://localhost:8000 |

## 📊 Monitoramento e Logs

### Ver Logs em Tempo Real

```powershell
# API Java
docker logs -f fot-api

# Flask Wrapper
docker logs -f ml-wrapper

# ML Service
docker logs -f ml-service-mock
```

### Logs de Uma Requisição Completa

**1. API Java:**
```
INFO - Received prediction request for flight AA1234 from JFK to LAX
INFO - Processing prediction for flight AA1234
INFO - Sending prediction request to ML service for flight: AA1234
INFO - Received prediction from ML service: prediction=1, probability=0.91
INFO - Prediction result from ML service: Delayed with probability 0.91
```

**2. Flask Wrapper:**
```
INFO - Request received from Java API: AA1234
INFO - Forwarding to external ML service...
INFO - Sending request to ML service: AA1234
INFO - Prediction received from ML service: prediction=1, probability=0.91
INFO - Returning result to Java API: {'prediction': 1, 'confidence': 0.91}
```

**3. ML Service:**
```
Mock ML Service - Received: AA1234
Mock ML Service - Returning: {'prediction': 1, 'probability': 0.91}
172.18.0.2 - - [21/Dec/2025 19:02:22] "POST /predict HTTP/1.1" 200 -
```

## 🔧 Configurações

### Variáveis de Ambiente

#### API Java (fot-api)
```yaml
environment:
  - SPRING_PROFILES_ACTIVE=prod
  - ML_SERVICE_URL=http://ml-wrapper:5000/predict
  - ML_SERVICE_TIMEOUT=5000
```

#### Flask Wrapper (ml-wrapper)
```yaml
environment:
  - FLASK_ENV=production
  - FLASK_DEBUG=False
  - ML_SERVICE_URL=http://ml-service:8000/predict
  - ML_SERVICE_TIMEOUT=30
  - LOG_LEVEL=INFO
```

### Arquivos de Configuração

- **Java:** `fot/src/main/resources/application.properties`
- **Python:** `mlwrapper/app/config.py`

## 🧪 Modos de Operação

### 1. Modo Desenvolvimento (Mock ML)

```powershell
# Usar mock ML service
docker compose --profile mock up -d
```

**Características:**
- ML Service retorna predições aleatórias
- Útil para testes de integração
- Não requer modelo treinado

### 2. Modo Produção (ML Real)

```powershell
# Usar ML service real
docker compose up -d
```

**Características:**
- Conecta a serviço ML real
- Requer modelo treinado
- Configurar `ML_SERVICE_URL` adequadamente

## 🔒 Segurança

### API Java

- **Autenticação:** Spring Security (Basic Auth)
- **Validação:** Bean Validation em todos os DTOs
- **Tratamento de Erros:** GlobalExceptionHandler

### Flask Wrapper

- **Validação:** Pydantic models
- **Timeout:** Previne requisições longas
- **Error Handling:** Try-catch com logs detalhados

## 🚀 Inicialização Rápida

```powershell
# 1. Subir todos os serviços (com mock ML)
cd d:\FlightOnTime
docker compose --profile mock up -d

# 2. Aguardar serviços ficarem healthy
docker ps

# 3. Testar
$body = @{
    flightNumber = "AA1234"
    companyName = "AA"
    flightOrigin = "JFK"
    flightDestination = "LAX"
    flightDepartureDate = "2025-12-25T14:30:00"
    flightDistance = 3974
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8080/api/v1/predict" `
                  -Method POST `
                  -Body $body `
                  -ContentType "application/json"
```

## 📚 Documentação Adicional

- **[Docker.md](Docker.md)** - Guia completo de Docker e containers
- **API Docs:** http://localhost:8080/swagger-ui.html (quando ativo)

## 🤝 Responsabilidades por Camada

### API Java
- ✅ Validação de entrada robusta
- ✅ Autenticação e autorização
- ✅ Cache de predições (futuro)
- ✅ Rate limiting (futuro)
- ✅ Métricas e monitoramento

### Flask Wrapper
- ✅ Adapter/Bridge entre Java e ML
- ✅ Transformação de formatos
- ✅ Validação adicional
- ✅ Retry logic (futuro)
- ✅ Circuit breaker (futuro)

### ML Service
- ✅ Inferência do modelo
- ✅ Feature engineering
- ✅ Otimização de performance
- ✅ Versionamento de modelos
- ✅ A/B testing (futuro)

## 🔍 Troubleshooting

### ML Service não responde

```powershell
# Verificar se container está rodando
docker ps | Select-String ml-service

# Ver logs
docker logs ml-service-mock

# Reiniciar
docker restart ml-service-mock
```

### Timeout na comunicação

```powershell
# Verificar conectividade entre containers
docker exec fot-api ping ml-wrapper
docker exec ml-wrapper ping ml-service
```

### Erro de mapeamento de campos

- Verificar se Flask está retornando `confidence` (não `probability`)
- Verificar logs do Flask Wrapper
- Validar DTO do Java (`MLServiceResponseDTO`)

---

**Última atualização:** 21 de dezembro de 2025
