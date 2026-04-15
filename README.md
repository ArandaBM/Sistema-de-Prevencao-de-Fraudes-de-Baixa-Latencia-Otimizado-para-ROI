# 🛡️ Sistema de Prevenção de Fraudes de Baixa Latência Otimizado para ROI

[![Status](https://img.shields.io/badge/Status-Em_Desenvolvimento-blue.svg)]()
[![Focus](https://img.shields.io/badge/Focus-Low_Latency_|_ROI-green.svg)]()
[![Model](https://img.shields.io/badge/Model-CatBoost-orange.svg)]()

Este projeto é um sistema end-to-end de detecção de fraudes em transações de cartão de crédito, projetado para equilibrar **precisão estatística** com **viabilidade econômica (ROI)** e **performance de tempo real (baixa latência)**.

## 📈 O Diferencial: Foco no ROI (Return on Investment)

Diferente de modelos puramente acadêmicos que buscam apenas maximizar o F1-Score ou AUC, este sistema utiliza um **Reporter Financeiro** que traduz métricas de ML em valores monetários. Otimizamos o *threshold* de decisão para:
- **Maximizar o Dinheiro Salvo**: Interceptar fraudes reais.
- **Minimizar o Custo de Fricção**: Evitar o bloqueio indevido de bons clientes (custo de suporte e perda de churn).

## 🚀 Arquitetura de Experimentos (Evolução do Modelo)

O projeto é dividido em experimentos incrementais para medir o ganho de cada nova camada de inteligência:

1.  **[Exp 01: Baseline Stateless](./src/models/exp01_baseline_stateless.py)**: Modelo base sem histórico, apenas dados imediatos da transação.
2.  **[Exp 02: Stateful Velocity](./src/models/exp02_stateful_velocity.py)**: Introdução de janelas deslizantes (24h/7d) para capturar a frequência de uso do cliente.
3.  **[Exp 03: Spatial Velocity](./src/models/exp03_spatial_velocity.py)**: Geoprocessamento com fórmula de Haversine para detectar anomalias de distância.
4.  **[Exp 04: Full State & Micro-Latency](./src/models/exp04_full_state_and_micro_latency.py)**: Otimização total incluindo comportamento do lojista e latência em segundos entre compras.

---

## 🗺️ Roadmap e Próximos Passos (O Futuro do Protótipo)

### 1. 🛡️ A API de Inferência (O Escudo)
Não utilizaremos o Pandas em produção. Implementaremos uma API com **FastAPI** capaz de responder em **< 100ms**.
- **Memória Célere**: Utilização de Redis ou dicionários em memória para acessar o estado do cliente e do lojista instantaneamente.
- **Lightweight Preprocessing**: Migração da lógica do `processing.py` para formatos otimizados para tempo real.

### 2. 🔄 O Loop de Feedback (A Chegada dos Chargebacks)
Sistemas de fraude aprendem com a dor. Criaremos um script de automação para processar **Chargebacks**:
- Quando um cliente reporta uma fraude não detectada (Falso Negativo), o dado é rotulado e injetado de volta na base original para retreinamento.

### 3. 🌑 Shadow Mode (Campeão vs. Desafiante)
Nunca substituímos um modelo de produção sem evidências. Implementaremos um workflow de **Shadow Mode**:
- O modelo novo (**Desafiante**) roda silenciosamente avaliando transações reais junto com o antigo (**Campeão**).
- O Desafiante só assume o controle se provar um ROI superior através dos relatórios do `reporter.py`.

---

## 📂 Estrutura do Repositório

```bash
├── data/               # Camadas de dados (Raw, Processed)
├── models/             # Modelos serializados (.cbm)
├── reports/            # Relatórios Executivos de ROI
├── src/
│   ├── data/           # ETL e Ingestão
│   ├── features/       # Feature Engineering e Utils Geoespaciais
│   ├── models/         # Scripts de Experimentos e Reporter
│   └── visualization/  # Análise Exploratória
└── requirements.txt    # Dependências (CatBoost, Pandas, Sklearn)
```

## 🛠️ Como Executar

1. Instale as dependências: `pip install -r requirements.txt`
2. Baixe os dados: `python src/data/download_data.py`
3. Rode o experimento mais avançado: `python src/models/exp04_full_state_and_micro_latency.py`

