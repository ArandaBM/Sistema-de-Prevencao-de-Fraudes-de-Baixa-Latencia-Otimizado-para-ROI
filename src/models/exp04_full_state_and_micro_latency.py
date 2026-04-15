"""
Experimento 04: Full State and Micro-Latency (Estado Completo e Micro-Latência)
-------------------------------------------------------------------------
Objetivo: Otimização máxima combinando o estado do cliente e do lojista,
adicionando a dimensão de "Micro-Latência" (tempo entre transações sucessivas).

Hipóteses Adicionais:
1. Fraude via Merchant: Padrões anômalos no volume do lojista indicam comprometimento.
2. Scripts/Automation: Transações com intervalos de segundos indicam ataques de força bruta
   ou automação, o que capturamos via 'time_since_last_trans'.
"""

import sys
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Adicionando o diretório raiz ao path para importar utilitários
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.processing import haversine_vectorized
from src.models.reporter import evaluate_and_report

def load_data():
    """Carrega os datasets de treino e teste."""
    train = pd.read_csv('data/raw/fraudTrain.csv', index_col=0)
    test = pd.read_csv('data/raw/fraudTest.csv', index_col=0)
    return train, test

def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza Engenharia de Atributos avançada (Cliente + Lojista + Micro-Latência).

    Args:
        df: DataFrame original.

    Returns:
        pd.DataFrame: DataFrame com o set completo de features.
    """
    print("Iniciando Engenharia de Atributos (Full Stack)...")
    
    # --- BLOCO 1: ORDENAÇÃO BASE (CLIENTE) ---
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df = df.sort_values(by=['cc_num', 'trans_date_trans_time']).reset_index(drop=True) 
    
    # --- BLOCO 2A: VELOCIDADE E LATÊNCIA DO CLIENTE ---
    # Micro-Latência: Segundos desde a última compra
    df['time_since_last_trans'] = df.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds()
    df['time_since_last_trans'] = df['time_since_last_trans'].fillna(-1)
    
    df_time_indexed = df.set_index('trans_date_trans_time')
    grouped = df_time_indexed.groupby('cc_num')['amt']
    
    df['trans_count_24h'] = (grouped.rolling('24h').count() - 1).values
    count_7d = grouped.rolling('7D').count() - 1
    sum_7d = grouped.rolling('7D').sum()
    
    past_sum_7d = sum_7d.values - df_time_indexed['amt'].values
    mean_7d = past_sum_7d / (count_7d.values + 1e-9)
    
    df['amt_to_mean_7d_ratio'] = df['amt'] / (mean_7d + 0.001)

    # --- BLOCO 2B: VELOCIDADE DO LOJISTA ---
    # Mudamos a ordem para calcular métricas do merchant
    df = df.sort_values(by=['merchant', 'trans_date_trans_time']).reset_index(drop=True)
    df_time_indexed_merch = df.set_index('trans_date_trans_time')
    
    grouped_merchant = df_time_indexed_merch.groupby('merchant')['amt']
    df['merchant_trans_count_24h'] = (grouped_merchant.rolling('24h').count() - 1).values
    df['merchant_trans_count_24h'] = df['merchant_trans_count_24h'].fillna(0)
    
    # --- BLOCO 3: ORDENAÇÃO TEMPORAL ESTRITA ---
    df = df.sort_values(by=['trans_date_trans_time']).reset_index(drop=True) 
    
    # --- BLOCO 4: EXTRAÇÃO TEMPORAL, DEMOGRÁFICA E ESPACIAL ---
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    
    df['dob'] = pd.to_datetime(df['dob'])
    df['customer_age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    
    df['distance_km'] = haversine_vectorized(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    
    # --- BLOCO 5: EXPURGO DE TOXIDADE ---
    cols_to_drop = [
        'trans_num', 'cc_num', 'first', 'last', 'street', 
        'dob', 'trans_date_trans_time', 'unix_time',
        'lat', 'long', 'merch_lat', 'merch_long',
        'city', 'state', 'zip', 'city_pop'
    ]
    
    return df.drop(columns=cols_to_drop)

def run_experiment_04():
    print("Iniciando Experimento 04: Full State and Micro-Latency...")
    df_train_raw, df_test_raw = load_data()

    df_train = prep_data(df_train_raw)
    df_test = prep_data(df_test_raw)

    target_col = 'is_fraud' 
    features = [col for col in df_train.columns if col != target_col]
    cat_features = df_train[features].select_dtypes(include=['object', 'category', 'string', 'O']).columns.tolist()

    split_index = int(len(df_train) * 0.8)
    X_train_int, y_train_int = df_train[features].iloc[:split_index], df_train[target_col].iloc[:split_index]
    X_val_int, y_val_int = df_train[features].iloc[split_index:], df_train[target_col].iloc[split_index:]

    scale_weight = (y_train_int == 0).sum() / (y_train_int == 1).sum()

    print("Iniciando treinamento SOTA do CatBoost...")
    model = CatBoostClassifier(
        iterations=1000, 
        learning_rate=0.05,
        depth=6,
        eval_metric='PRAUC',
        scale_pos_weight=scale_weight,
        cat_features=cat_features,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )

    model.fit(X_train_int, y_train_int, eval_set=(X_val_int, y_val_int))

    y_probs = model.predict_proba(df_test[features])[:, 1]
    evaluate_and_report(df_test[target_col], y_probs, df_test['amt'].values, "exp04_full_state_and_micro_latency")

    # 5.2. Salvar o modelo treinado
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.normpath(os.path.join(current_dir, "../../models/catboost_sota.cbm"))
    model.save_model(save_path)
    print(f"\n[MLOps] Modelo exportado com sucesso para '{save_path}'")

if __name__ == "__main__":
    run_experiment_04()
