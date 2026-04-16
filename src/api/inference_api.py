"""
API de Inferência - Motor de Prevenção a Fraudes SOTA (Baixa Latência)
----------------------------------------------------------------------
Este módulo implementa o serviço de inferência em tempo real utilizando o framework FastAPI. 
Projetado para ambientes de produção de alta volumetria, ele incorpora padrões arquiteturais 
críticos de MLOps:

1. **In-Memory State Management**: Mantém o estado transacional recente do usuário e do 
   lojista em memória (RAM) para calcular features de velocidade em tempo constante O(1).
2. **Carregamento Antecipado (Warm Start)**: O modelo CatBoost é injetado na memória na 
   inicialização do processo, eliminando a latência de I/O em tempo de execução de predição.
3. **Engenharia de Atributos Real-Time**: Executa cálculos geoespaciais (Haversine) e 
   agrupamentos temporais (micro-latência) no mesmo delta de tempo da transação.
4. **Alinhamento Estrito de Tensors**: Assegura que o pipeline alimente o modelo EXATAMENTE 
   com o mesmo schema de features treinado, mitigando data drift ou desalinhamento esquemático.

Autor: Bruno (Desenvolvido para compor Arquitetura de Portfólio de Engenharia de ML)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
from catboost import CatBoostClassifier

# Inicialização da API contendo metadados autogerados no Swagger (/docs)
app = FastAPI(
    title="Motor de Prevenção a Fraudes SOTA", 
    version="2.0",
    description="API de inferência estritamente otimizada para respostas sub-milissegundo baseada em estados dinâmicos temporais."
)

# =====================================================================
# 1. Carregamento do Modelo na Inicialização (Globally Scoped)
# =====================================================================
model = CatBoostClassifier()
try:
    # Em produção, esse artefato estaria hospedado em um Bucket S3 ou GCS com versionamento (ex: MLflow)
    model.load_model('models/catboost_sota.cbm')
    print("[API] Cérebro SOTA (CatBoost) carregado com sucesso na memória!")
    
    # Salva a ordem exata das features que o modelo espera, extraídas diretamente da topologia do CBM 
    EXPECTED_FEATURES = model.feature_names_ 
except Exception as e:
    print(f"[API] ERRO FATAL: Modelo não encontrado. Verifique os volumes montados. {e}")

# O Limiar cravado pela calibração de DRE (Demonstrativo de Resultado), visando o maior ROI (88%)
BUSINESS_THRESHOLD = 0.88

# =====================================================================
# 2. STATE MANAGER (A Memória RAM de Baixa Latência)
# =====================================================================
# Estruturas de dados in-memory mutáveis que emulam um Feature Store em tempo real (como Redis/Memcached)
# AVISO (Portfólio): Para suporte a múltiplos workers (Gunicorn/Uvicorn), isso seria refatorado para um In-Memory DB.
customer_history = {}      # Mapeia { 'cc_num': [(datetime, amt), ...] } para contagem/soma da frequência.
customer_last_time = {}    # Mapeia { 'cc_num': datetime } para detectar a métrica de Micro-Latência.
merchant_history = {}      # Mapeia { 'merchant': [datetime, ...] } para detectar volume atômico do lado do adquirente.

# =====================================================================
# 3. SCHEMA DE ENTRADA (O Payload JSON Enxuto via Pydantic)
# =====================================================================
class TransactionRequest(BaseModel):
    """
    Contrato de dados da API. 
    A Pydantic garante validação de tipos de Rust em nano-segundos.
    """
    cc_num: str = Field(...,scription="Hash/Número do cartão transacionado.")
    merchant: str = Field(..., description="ID ou Razão Social do Lojista.")
    category: str = Field(..., description="Merchant Category Code (MCC).")
    job: str = Field(..., description="Agrupamento demográfico da ocupação do titular.")
    gender: str = Field(..., description="Agrupamento demográfico do gênero do titular.")
    amt: float = Field(..., description="Valor transacionado (USD/BRL).", gt=0)
    trans_date_trans_time: datetime = Field(..., description="Timestamp ISO-8601 exato da transação.")
    dob: datetime = Field(..., description="Tempo de vida (data de nascimento) do usuário logado.")
    lat: float = Field(..., description="Latitude do dispositivo pagador.")
    long: float = Field(..., description="Longitude do dispositivo pagador.")
    merch_lat: float = Field(..., description="Latitude do adquirente fiscal.")
    merch_long: float = Field(..., description="Longitude do adquirente fiscal.")

# =====================================================================
# FUNÇÕES DE AUXÍLIO
# =====================================================================
def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância ortodrômica entre as coordenadas do merchant e o billing client.
    Essencial para captar anomalias espaciais (Ex: "Viagem impossível em 5 minutos").
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2.0)**2
    return 6371 * (2 * np.arcsin(np.sqrt(a)))

# =====================================================================
# 4. MOTOR DE ENGENHARIA REAL-TIME (Complexidade O(1) e O(N))
# =====================================================================
def extract_realtime_features(txn: TransactionRequest):
    """
    Corrente Sanguínea Pipeline MLOps. Esta função consome o estado online global (memória RAM) 
    para calcular vetores agregados e séries temporais na borda.
    
    Args:
        txn (TransactionRequest): O payload em tempo real desserializado da requisição POST.
        
    Returns:
        List: Um vetor serializado de atributos, cirurgicamente ordenados para match com o artefato de Treino.
    """
    now = txn.trans_date_trans_time
    cc = txn.cc_num
    merch = txn.merchant
    amt = txn.amt

    # --- A. MICRO-LATÊNCIA (Interrupção de Força Bruta de Scripts) ---
    last_time = customer_last_time.get(cc)
    time_since_last_trans = (now - last_time).total_seconds() if last_time else -1.0
    customer_last_time[cc] = now 

    # --- B. VELOCIDADE DO CLIENTE (Análise Temporal Recente) ---
    c_hist = customer_history.get(cc, [])
    # Janela deslizante severa expurgando histórico além de 7 dias O(N otimizado)
    c_hist = [(t, a) for t, a in c_hist if (now - t).total_seconds() <= 7 * 24 * 3600]
    
    count_24h = sum(1 for t, a in c_hist if (now - t).total_seconds() <= 24 * 3600)
    count_7d = len(c_hist)
    past_sum_7d = sum(a for t, a in c_hist)
    
    amt_mean_7d = (past_sum_7d / count_7d) if count_7d > 0 else amt
    amt_to_mean_7d_ratio = amt / (amt_mean_7d + 0.001)

    c_hist.append((now, amt))
    customer_history[cc] = c_hist

    # --- C. VELOCIDADE DO LOJISTA ---
    m_hist = merchant_history.get(merch, [])
    m_hist = [t for t in m_hist if (now - t).total_seconds() <= 24 * 3600]
    merchant_trans_count_24h = len(m_hist)
    
    m_hist.append(now)
    merchant_history[merch] = m_hist

    # --- D. ESPAÇO E TEMPO (Contextos Geodemográficos) ---
    distance_km = haversine(txn.lat, txn.long, txn.merch_lat, txn.merch_long)
    customer_age = (now.date() - txn.dob.date()).days // 365
    trans_hour = now.hour
    trans_day_of_week = now.weekday()

    # --- E. ALINHAMENTO ESTRITO DE FEATURES ---
    # Cria um dicionário dinâmico com tudo o que o feature engine construiu
    feature_dict = {
        'merchant': txn.merchant,
        'category': txn.category,
        'job': txn.job, # Usando a coluna categórica extraída na raiz de treinamento caso emulada
        'gender': txn.gender, # Usando a coluna categórica extraída na raiz de treinamento caso emulada
        'amt': txn.amt,
        'time_since_last_trans': time_since_last_trans,
        'trans_count_24h': count_24h,
        'trans_count_7d': count_7d,
        'amt_mean_7d': amt_mean_7d,
        'amt_to_mean_7d_ratio': amt_to_mean_7d_ratio,
        'merchant_trans_count_24h': merchant_trans_count_24h,
        'trans_hour': trans_hour,
        'trans_day_of_week': trans_day_of_week,
        'customer_age': customer_age,
        'distance_km': distance_km
    }
    
    # IMPORTANTE: Garante que o vetor final enviado ao CatBoost tenha 
    # a exata mesma ordem de colunas do DataFrame de treino original (Feature Alignment Pipeline).
    try:
        return [feature_dict[col] for col in EXPECTED_FEATURES]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Falta uma coluna computada esperada pelo pipeline CatBoost: {e}")

# =====================================================================
# 5. ENDPOINT DE DECISÃO (A Porta da Frente)
# =====================================================================
@app.post("/predict", summary="In-Memory Scoring Engine", tags=["Inference"])
async def predict_fraud(txn: TransactionRequest):
    """
    Endpoint principal para aprovação ou declínio instantâneo de transações.
    Recebe um JSON serializado do gateway de pagamentos/adquirente.
    
    Workflows:
    1. Instancia Atributos Dinâmicos usando o RAM Cache.
    2. Realiza o Score Probabilístico via Árvores por Gradiente (CatBoost).
    3. Corta (Block) a transação se probability > Threshold Otimizado (88%).
    """
    try:
        # Extração de features sintéticas temporais
        feature_vector = extract_realtime_features(txn)
        
        # Inferência (predição crua de probabilidade variando entre [0, 1])
        prob_fraud = model.predict_proba([feature_vector])[0][1]
        
        # Regra de Negócio (Limiar estabelecido baseado na curva Precision-Recall do ROI DRE)
        decision = "BLOCK" if prob_fraud >= BUSINESS_THRESHOLD else "APPROVE"
        
        return {
            "status": decision,
            "fraud_probability": round(prob_fraud, 4),
            "business_threshold": BUSINESS_THRESHOLD,
            "metrics": {
                # Logs vitais contidos na resposta para a camada de auditoria e data observability
                "distance_km": round(feature_vector[EXPECTED_FEATURES.index('distance_km')], 2),
                "micro_latency_sec": feature_vector[EXPECTED_FEATURES.index('time_since_last_trans')]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))