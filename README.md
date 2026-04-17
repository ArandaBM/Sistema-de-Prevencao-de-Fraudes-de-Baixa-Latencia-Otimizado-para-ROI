<div align="center">
  <h1>🚀 Motor de Prevenção a Fraudes de Baixa Latência (Otimizado para ROI)</h1>
  <p><i>Projeto pessoal End-to-End focado no impacto financeiro, State Management in-memory e arquitetura de inferência crítica.</i></p>

  ![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?style=for-the-badge&logo=fastapi&logoColor=white)
  ![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
  ![SQLite](https://img.shields.io/badge/SQLite-Logs-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
</div>

---
```mermaid
graph LR
    classDef input fill:#2d3436,stroke:#dfe6e9,stroke-width:2px,color:#fff;
    classDef api fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef feat fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff;
    classDef model fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    classDef db fill:#e17055,stroke:#ffeaa7,stroke-width:2px,color:#fff;
    classDef block fill:#d63031,stroke:#ff7675,stroke-width:2px,color:#fff;

    A((Payload<br>JSON)):::input --> B[FastAPI + Pydantic<br>Validação]:::api

    subgraph "Docker Container: API de Baixa Latência"
        B --> C[Stateless Features<br>Haversine Distance]:::feat
        B --> D[(Stateful RAM<br>Micro-Latência / Velocity)]:::feat
        
        C --> E{CatBoost<br>Motor de Inferência}:::model
        D --> E
        
        E -->|Score 0.0 a 1.0| F[Decision Logic<br>Threshold: 0.88]:::api
    end

    F -->|Risco < 0.88| G([APPROVE<br>~11ms SLA]):::feat
    F -->|Risco >= 0.88| H([BLOCK<br>~11ms SLA]):::block
    
    F -.->|Assíncrono| I[(SQLite DB<br>production_logs)]:::db

## 💼 1. O Desafio de Negócio (DRE)

Diferente de competições acadêmicas voltadas puramente para F1-Score ou ROC-AUC, decidi focar este projeto em como as fraudes transacionais (adquirência/emissão) são sentidas no mundo real: através do **Demonstrativo de Resultados do Exercício (DRE)**. Um modelo que visa zerar Falsos Negativos (fraude não detectada) muitas vezes eleva o Custo de Fricção no usuário legítimo (Falsos Positivos).

Desenvolvi a arquitetura de decisão deste motor utilizando uma calibração extrema do *Threshold* visando a **rentabilidade da operação**, não apenas a precisão matemática. Construí a matriz de custo penalizando financeiramente o modelo em **R$ 50,00 por cada Falso Positivo** (o custo estimado de fricção/insatisfação ao bloquear um cliente bom indevidamente).

- **Limiar de Operação Otimizado (Threshold):** `0.88`
- **Lucro Líquido Gerado:** **R$ 1.074.159,60** (Resultado estrito das fraudes interceptadas subtraindo os R$ 50,00 de cada fricção e os valores perdidos nos falsos negativos).
- **Latência Crítica alcançada (SLA):** **~11ms**, totalmente compatível com esteiras autorizadoras assíncronas que exigem respostas sub-100ms.

---

## 🏗️ 2. Arquitetura e Engenharia de Atributos

Como operações reais de Machine Learning de baixa latência excluem pré-processamentos pesados envolvendo pipelines do Pandas, decidi elevar o desafio de engenharia traduzindo heurísticas temporais complexas para memória bruta matemática.

### O "State Manager" em Memória RAM
Implementei um gerenciador de estado mutável nativo que simula um Feature Store paralelo durante as rotas de inferência. Nele, o sistema calcula séries de dados temporais de forma otimizada com complexidade assintótica de **O(1)** e mapeia agrupamentos no espaço amostral de **O(N)**, removendo inteiramente o peso de funções clássicas como `.rolling()`.

### Features Comportamentais (Stateful & Spatial)
Para treinar o algoritmo (CatBoost), criei manualmente as seguintes features:
- **Micro-Latência (Force-Brute Prevention):** Mede em milissegundos o delta entre requisições de um mesmo cartão (`cc_num`). Foi a minha solução para mitigar ataques de botnets instantaneamente.
- **Velocity Tracking:** Contadores construídos no State Manager que medem as explosões na volumetria das útimas 24 horas e 7 dias.
- **Spend Ratio:** Proporção gerada entre o ticket médio `amt` atual contrajetado à média histórica de 7 dias do próprio cliente.
- **Geofísica Mapeada (Haversine):** Expurguei variáveis categóricas ruidosas e altamente sujeitas a Data Drift (CEP, Estado, Cidade). Em seu lugar, moldei operações ortodrômicas (*Haversine*) sendo executadas no pré-processamento. Elas convertem a lat/long entre o Cliente e o Lojista em uma matriz nítida de quilometragem.

---

## 📂 3. Estrutura do Projeto

Para demonstrar maturidade arquitetural semelhante à de grandes operações de MLOps, separei meu repositório isolando completamente o ambiente de pesquisa (`experiments/`) das esteiras de produção (`api/`).

```text
├── docs/                        # Relatórios gerados do treinamento e tracking de DRE original
├── models/                      
│   └── catboost_sota.cbm        # O meu melhor Artefato serializado (Warm-up load na APi)
├── src/
│   ├── api/                     # Código fechado e seguro para a containerização de Produção
│   │   ├── inference_api.py     # O motor Uvicorn/FastAPI desenvolvido
│   │   └── attack_simulation.py # Meu script gerador de botnet/stress test
│   └── experiments/             # Meu ambiente de pesquisa (onde ocorreu a feitiçaria!)
│       ├── core/                # Lib central que estruturei (processing.py, reporter.py)
│       ├── notebooks/           # EDA isolado (Evitando arquivos binários na raiz)
│       ├── exp01_baseline_stateless.py
│       └── exp04_full_state_and_micro_latency.py
├── production_logs.db           # Banco SQLite transacional (Explicado abaixo)
├── Dockerfile                   # A Receita Lean do Contêiner
├── requirements.txt             # Dependências da Imagem Docker (Enxutas)
└── requirements-dev.txt         # Pacotes de pesquisa que chamam recursos produtivos internamente
```

*(Obs: Utilizei a base de dados pública [Kaggle: Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data). Os datasets brutos são baixados e extraídos automaticamente na pasta oculta `data/` pelo meu script core no ambiente local).*

---

## 🐋 4. MLOps e Deploy

Aloquei 100% da inteligência da API em uma Imagem Docker estritamente focada em leveza. O segredo dessa estrutura foi criar um `.dockerignore` robusto que cega o Docker durante o *build* para os scripts soltos do meu laboratório. 

Para alcançar a separação completa dos escopos, o `requirements.txt` serve de alimento exclusivo para o Dockerfile contendo apenas (API, Pydantic, CatBoost), enquanto criei o sufixo `-dev` para eu mesmo desenvolver em máquina local com as comodidades analíticas (Jupyter, Seaborn, Sklearn).

**Run Local via Shell/Bash:**
```bash
# 1. Empacotar a imagem (Gerará uma casca extremamente leve só com FastAPI + Modelo)
docker build -t fraud-engine-api:latest .

# 2. Subir a API conectando na porta host
docker run -p 8000:8000 fraud-engine-api:latest
```

---

## 🔁 5. Visão de Futuro (Latência de Rótulo)

Problemas de detecção lidam contra o que conceituamos em engenharia de fraudes de **Latência de Ground Truth** (ou *Chargebacks*). Em cenários reais, descobrimos que nossa predição de ontem falhou apenas quando a fatura do cliente é contestada 45 dias depois.

Em termos de arquitetura e evolução do portfólio, estruturei e anexei um **Banco SQLite** (`production_logs.db`). Sua função é atuar como uma *Shadow Database* na API:
1. Ele loga toda decisão síncrona junto à probabilidade crua.
2. Permitiria arquitetar no futuro uma rotina assíncrona (como Airflow) para conciliar o status "D-0" da minha API contra o "D-45" da MasterCard/Visa.
3. Este banco é o coração que possibilita governanças avandadas de retreinamento contínuo (CT), acionando o pipeline de deploy estilo **Campeão vs. Desafiante**.