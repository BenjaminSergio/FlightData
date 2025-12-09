# FlightOnTime - Backend API

## Sobre o Projeto

Este é o backend do FlightOnTime, um sistema que ajuda a prever se um voo vai atrasar ou não.

A ideia é simples: você manda as informações de um voo (companhia aérea, origem, destino, horário, etc.) e a API retorna se o voo provavelmente será **Pontual** ou **Atrasado**, junto com a probabilidade dessa previsão.

## Objetivo

Desenvolver uma API REST em Java com Spring Boot que atue como intermediário entre aplicações clientes e o modelo de Machine Learning de previsão de atrasos de voos.

### Funcionalidades Principais

- Receber dados de um voo via requisição HTTP POST
- Validar se todas as informações necessárias foram enviadas corretamente
- Comunicar-se com o microserviço Python que hospeda o modelo de ML
- Processar a resposta do modelo e formatá-la adequadamente
- Retornar previsão estruturada em formato JSON
- Tratar erros de forma adequada e informativa

## Arquitetura da Solução

### Visão Geral

O sistema segue uma arquitetura de microserviços onde:

1. **API Gateway (Este Projeto)**: Aplicação Spring Boot que expõe endpoints REST e gerencia requisições
2. **Serviço de ML**: Aplicação Python (FastAPI/Flask) que carrega e executa o modelo treinado
3. **Comunicação**: HTTP/REST entre os dois serviços

### Fluxo de Requisição

```Java
Cliente (Postman/App) 
    ↓ POST /predict
API Spring Boot (validação + transformação)
    ↓ HTTP Request
Serviço Python (modelo ML)
    ↓ Resposta JSON
API Spring Boot (formatação)
    ↓ Response JSON
Cliente recebe previsão
```

### Estrutura do Projeto

```Java
src/
├── main/
│   ├── java/com/backend/
│   │   ├── controller/          # Camada de apresentação (REST endpoints)
│   │   │   └── PredictionController.java
│   │   ├── service/             # Camada de negócio
│   │   │   ├── PredictionService.java
│   │   │   └── MLServiceClient.java
│   │   ├── dto/                 # Data Transfer Objects
│   │   │   ├── FlightPredictionRequest.java
│   │   │   ├── FlightPredictionResponse.java
│   │   │   └── MLServiceResponse.java
│   │   ├── config/              # Configurações
│   │   │   └── RestTemplateConfig.java
│   │   ├── exception/           # Tratamento de exceções
│   │   │   ├── GlobalExceptionHandler.java
│   │   │   ├── InvalidRequestException.java
│   │   │   └── MLServiceException.java
│   │   └── BackendApplication.java
│   └── resources/
│       ├── application.properties
│       └── application-dev.properties
└── test/
    └── java/com/flightontime/
        ├── controller/
        └── service/
```

## Planejamento de Desenvolvimento

### Fase 1: Configuração Inicial

**Objetivo**: Preparar o ambiente e estrutura base do projeto

**Atividades**:

- Criar projeto Spring Boot via Spring Initializr ou Maven
- Configurar dependências no `pom.xml`:
  - Spring Web (para criar REST APIs)
  - Spring Boot Validation (para validação de dados)
  - Lombok (para reduzir código boilerplate)
  - Spring Boot DevTools (para hot reload durante desenvolvimento)
- Configurar `application.properties` com porta e URL do serviço ML
- Validar que a aplicação inicia sem erros

**Entregável**: Projeto compila e aplicação Spring Boot inicia com sucesso

---

### Fase 2: Implementação dos DTOs

**Objetivo**: Criar os objetos de transferência de dados que definem contratos da API

**FlightPredictionRequest** (Entrada da API):

```java
{
  "companhia": String (obrigatório, 2 caracteres)
  "origem": String (obrigatório, código IATA 3 letras)
  "destino": String (obrigatório, código IATA 3 letras)
  "data_partida": LocalDateTime (obrigatório, formato ISO-8601)
  "distancia_km": Integer (obrigatório, > 0)
}
```

**Validações**:

- `@NotNull` e `@NotBlank` para campos obrigatórios
- `@Size` para limitar tamanho de strings
- `@Positive` para distância
- `@Future` para data de partida (opcional, dependendo do caso de uso)

**FlightPredictionResponse** (Saída da API):

```java
{
  "previsao": String ("Pontual" ou "Atrasado")
  "probabilidade": Double (0.0 a 1.0)
}
```

**MLServiceResponse** (Resposta do serviço Python):

```java
{
  "prediction": Integer (0 ou 1)
  "probability": Double (0.0 a 1.0)
}
```

**Entregável**: Classes DTO criadas com anotações de validação

---

### Fase 3: Desenvolvimento do Controller

**Objetivo**: Implementar o endpoint REST que recebe requisições

**PredictionController**:

- Endpoint: `POST /api/v1/predict`
- Recebe `FlightPredictionRequest` validado com `@Valid`
- Retorna `ResponseEntity<FlightPredictionResponse>`
- Status HTTP 200 para sucesso
- Delega processamento para `PredictionService`

**Responsabilidades**:

- Receber e validar entrada
- Invocar camada de serviço
- Retornar resposta HTTP apropriada
- Não contém lógica de negócio

**Entregável**: Controller funcional que responde requisições HTTP

---

### Fase 4: Implementação do Service Layer

**Objetivo**: Criar lógica de negócio e integração com serviço ML

**PredictionService**:

- Recebe `FlightPredictionRequest`
- Transforma dados para formato esperado pelo modelo
- Invoca `MLServiceClient`
- Converte resposta do modelo para `FlightPredictionResponse`
- Mapeia valores (0 → "Pontual", 1 → "Atrasado")

**MLServiceClient**:

- Configura `RestTemplate` ou `WebClient`
- Faz chamada HTTP POST para serviço Python
- URL configurável via `application.properties`
- Trata timeouts e erros de conexão
- Deserializa resposta JSON

**Configuração**:

```properties
ml.service.url=http://localhost:5000/predict
ml.service.timeout=5000
```

**Entregável**: Integração funcional com serviço Python

---

### Fase 5: Tratamento de Erros

**Objetivo**: Implementar tratamento robusto de exceções

**GlobalExceptionHandler** (`@ControllerAdvice`):

Trata múltiplos cenários:

1. **Validação de Entrada** (`MethodArgumentNotValidException`):
   - Status: 400 Bad Request
   - Retorna lista de erros de validação

2. **Serviço ML Indisponível** (`ResourceAccessException`):
   - Status: 503 Service Unavailable
   - Mensagem: "Serviço de previsão temporariamente indisponível"

3. **Resposta Inválida do ML** (`MLServiceException`):
   - Status: 502 Bad Gateway
   - Mensagem: "Erro ao processar previsão"

4. **Erros Genéricos** (`Exception`):
   - Status: 500 Internal Server Error
   - Mensagem genérica (sem expor detalhes internos)

**Formato de Erro Padronizado**:

```json
{
  "timestamp": "2025-12-08T14:30:00",
  "status": 400,
  "error": "Bad Request",
  "message": "Validação falhou",
  "errors": [
    {
      "field": "origem",
      "message": "Código IATA deve ter 3 caracteres"
    }
  ]
}
```

**Entregável**: API retorna erros estruturados e informativos

---

### Fase 6: Testes e Documentação

**Objetivo**: Garantir qualidade e facilitar uso da API

**Testes Unitários**:

- Testar validações dos DTOs
- Testar lógica de transformação no Service
- Mockar chamadas ao serviço ML
- Testar tratamento de exceções

**Testes de Integração** (opcional):

- Testar endpoint completo com WireMock
- Simular respostas do serviço Python

**Documentação**:

- Atualizar README com instruções de execução
- Criar collection do Postman com exemplos
- Documentar variáveis de ambiente necessárias
- Adicionar exemplos de cURL

**Entregável**: Testes passando e documentação completa

---

## Exemplos de Uso

**O que você envia:**

```json
{
  "companhia": "AZ",
  "origem": "GIG",
  "destino": "GRU",
  "data_partida": "2025-11-10T14:30:00",
  "distancia_km": 350
}
```

**O que você recebe:**

```json
{
  "previsao": "Atrasado",
  "probabilidade": 0.78
}
```

Isso significa: "Este voo tem 78% de chance de atrasar"

## Próximos Passos

1. Configurar o ambiente de desenvolvimento (JDK, IDE, Maven)
2. Criar o projeto Spring Boot
3. Alinhar com o time de Data Science sobre o formato dos dados
4. Começar a implementação seguindo as etapas acima

## Funcionalidades Extras (Se Der Tempo)

### 1. Front-end com Thymeleaf

Adicionar dependência `spring-boot-starter-thymeleaf` e criar páginas HTML em `templates/` com formulário de entrada e exibição de resultados.

### 2. PostgreSQL + Looker Studio

Adicionar `spring-boot-starter-data-jpa` e `postgresql` no pom.xml, configurar conexão no `application.properties` e criar entidade JPA para salvar histórico de previsões.
Conectar o Looker Studio ao PostgreSQL para criar dashboards de análise.

### 3. Outras Funcionalidades

- Criar endpoint para estatísticas (ex: % de voos atrasados no dia)
- Fazer testes automatizados
- Containerizar com Docker
- Implementar cache com Redis para previsões frequentes

## Integração com Data Science

**Importante:** Precisamos combinar com o time de DS:

- Em que porta/URL o serviço Python vai rodar
- Se o formato de entrada/saída está alinhado
- Como vamos testar a integração

---

**Lembre-se:** Este é um MVP (Produto Mínimo Viável). O objetivo é fazer funcionar de forma simples primeiro, e depois melhorar se houver tempo!
