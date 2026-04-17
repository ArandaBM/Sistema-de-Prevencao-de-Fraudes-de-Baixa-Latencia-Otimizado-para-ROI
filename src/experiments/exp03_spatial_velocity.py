"""
Experimento 03: Spatial Velocity (Espaço e Velocidade)
---------------------------------------------------
Meu Objetivo: Enriquecer o modelo estatístico Stateful anterior introduzindo 
uma camada agressiva de observabilidade geoespacial.

Minha Decisão Arquitetural:
Optei por remover as variáveis categóricas (CEP/Cidade) que são vítimas fáceis 
de Data Drift, e utilizei a Matemática de Haversine. Esta decisão gerou um balanço interessante 
onde garanti estabilidade ao mesmo tempo que a vetorização Numpy blindou o SLA estrito 
de milissegundos na API contra quebras de volumetria.
"""

import sys
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

from core.processing import haversine_vectorized
from core.reporter import evaluate_and_report

def load_data():
    """Carrega os datasets de treino e teste."""
    train = pd.read_csv('data/raw/fraudTrain.csv', index_col=0)
    test = pd.read_csv('data/raw/fraudTest.csv', index_col=0)
    return train, test

def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza Engenharia de Atributos incluindo distância geoespacial.

    Args:
        df: DataFrame original.

    Returns:
        pd.DataFrame: DataFrame com as features do experimento 03.
    """
    print("Iniciando Engenharia de Atributos...")
    
    # --- BLOCO 1: ORDENAÇÃO BASE ---
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df = df.sort_values(by=['cc_num', 'trans_date_trans_time']).reset_index(drop=True)
    
    # --- BLOCO 2: VELOCITY & ANOMALIA ---
    df_time_indexed = df.set_index('trans_date_trans_time')
    grouped = df_time_indexed.groupby('cc_num')['amt']
    
    count_24h = grouped.rolling('24h').count() - 1
    count_7d = grouped.rolling('7D').count() - 1
    sum_7d = grouped.rolling('7D').sum()
    past_sum_7d = sum_7d.values - df_time_indexed['amt'].values 

    mean_7d = past_sum_7d / (count_7d.values + 1e-9)
    
    df['trans_count_24h'] = count_24h.values
    df['trans_count_7d'] = count_7d.values
    df['amt_to_mean_7d_ratio'] = df['amt'] / (mean_7d + 0.001)
    
    # --- BLOCO 3: ORDENAÇÃO TEMPORAL ESTRITA ---
    df = df.sort_values(by=['trans_date_trans_time']).reset_index(drop=True)
    
    # --- BLOCO 4: EXTRAÇÃO TEMPORAL, DEMOGRÁFICA E ESPACIAL ---
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_day_of_week'] = df['trans_date_trans_time'].dt.dayofweek

    df['dob'] = pd.to_datetime(df['dob'])
    df['customer_age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365 

    # 4.1 Feature de Distância (Importada de processing.py)
    df['distance_km'] = haversine_vectorized(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )
    
    # --- BLOCO 5: EXPURGO DE TOXIDADE ---
    cols_to_drop = [
        'trans_num', 'cc_num', 'first', 'last', 'street', 
        'dob', 'trans_date_trans_time', 'unix_time',
        'lat', 'long', 'merch_lat', 'merch_long',
        'city', 'state', 'zip', 'city_pop'
    ]
    
    return df.drop(columns=cols_to_drop)

def run_experiment_03():
    print("Iniciando Experimento 03: Spatial Velocity...")
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
    evaluate_and_report(df_test[target_col], y_probs, df_test['amt'].values, "exp03_spatial_velocity")

if __name__ == "__main__":
    run_experiment_03()
