import pandas as pd
import requests
import time
from datetime import datetime

# =====================================================================
# SIMULADOR DE ATAQUE COM DATASET REAL (Stream-like Testing)
# =====================================================================
# Desenvolvi este script para mimetizar comportamentos de um streaming real (ex: Kafka).
# Ele varre o arquivo bruto CSV de fraudes e empurra contra a nossa Fast API
# permitindo auditar o desempenho do score dinâmico e o cálculo da micro-latência 
# de uma forma naturalística, como se fossem usuários verdadeiros logados.

API_URL = "http://127.0.0.1:8000/predict"
CSV_PATH = "data/raw/fraudTest.csv"

# Configurações Táticas da Simulação
DELAY_BETWEEN_REQUESTS = 0.5 # Intervalo para evitar flood do roteador local e simular espaçamento de usuários reais
TRANSACTIONS_TO_SIMULATE = 555718 # Volume da carga de teste a ser puxada do dataset

print("🔥 Inicializando o Engine de Teste de Dados Reais...")
print(f" -> Lendo lote tático a partir de: {CSV_PATH}")

# Extração estrita apenas das dependências mapeadas pelo Pydantic para poupar alocação de memória RAM
cols_to_use = [
    'cc_num', 'merchant', 'category', 'job', 'gender', 'amt', 
    'trans_date_trans_time', 'dob', 'lat', 'long', 'merch_lat', 'merch_long', 'is_fraud'
]

try:
    df = pd.read_csv(CSV_PATH, usecols=cols_to_use)
    # Ordenar cronologicamente pelo timestamp da transação para ativar 
    # de forma fidedigna as features temporais em Memória na API (estado da arte).
    df = df.sort_values(by='trans_date_trans_time').reset_index(drop=True)
except Exception as e:
    print(f"🛑 Erro fatal ao carregar log. Verifique se o caminho bate com a sua máquina: {e}")
    exit(1)

# Ingestão de uma fatia do dataframe para não travar o console de auditoria
batch_df = df.head(TRANSACTIONS_TO_SIMULATE)

print(f" -> Operação autorizada. Despachando {TRANSACTIONS_TO_SIMULATE} requisições contra o Motor de Baixa Latência (API)...\n")

for idx, row in batch_df.iterrows():
    # Instanciamento do contrato Pydantic que estruturei
    payload = {
        "cc_num": str(row['cc_num']),
        "merchant": str(row['merchant']),
        "category": str(row['category']),
        "job": str(row['job']),
        "gender": str(row['gender']),
        "amt": float(row['amt']),
        # Trato de trocar o espaço no datetime por T (Exigência ISO-8601 Pydantic)
        "trans_date_trans_time": str(row['trans_date_trans_time']).replace(' ', 'T'),
        "dob": str(row['dob']).replace(' ', 'T'),
        "lat": float(row['lat']),
        "long": float(row['long']),
        "merch_lat": float(row['merch_lat']),
        "merch_long": float(row['merch_long'])
    }
    
    # Ground Truth para avaliarmos localmente a saúde das predições feitas em C++ via CatBoost
    true_label = "FRAUD" if int(row['is_fraud']) == 1 else "LEGIT"

    print(f"[Disparo {idx+1}/{TRANSACTIONS_TO_SIMULATE}] Cartão final {payload['cc_num'][-4:]} | Compra no '{payload['merchant'][:10]}...' | Rótulo Real: {true_label}")
    
    # Isolando a infra para medir rigorosamente a latência que conseguimos na borda
    inicio = time.time()
    try:
        response = requests.post(API_URL, json=payload)
        latencia_api = (time.time() - inicio) * 1000
        
        if response.status_code == 200:
            res = response.json()
            status_api = res['status']
            prob_api = res['fraud_probability']
            micro_lat = res['metrics']['micro_latency_sec']
            
            # Formatação de saída limpa e gerencial
            print(f"  -> IA Motor : {status_api} (Match Probabilidade: {prob_api*100:.2f}%)")
            print(f"  -> Latência : {latencia_api:.2f}ms | Track de Velocidade: {micro_lat} sec")
        else:
            print(f"  -> ERRO PAYLOAD REJEITADO (Pydantic/Engine): {response.text}")
    except requests.exceptions.ConnectionError:
        print("🛑 ERRO DE CONEXÃO: Uvicorn não está respondendo. Garanta que a API está rodando no terminal ao lado.")
        break
    
    time.sleep(DELAY_BETWEEN_REQUESTS)
    print("-" * 65)

print("🛑 Fim do Stress Test. Todos os logs probabilísticos e inferências SHAP (background) foram salvas no production_logs.db.")