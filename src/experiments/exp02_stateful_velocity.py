"""
Experimento 02: Stateful Velocity (Com Estado e Velocidade)
---------------------------------------------------------
Meu Objetivo: Incrementar estruturalmente o lab introduzindo features in-memory 
que capturem o comportamento histórico latente (Stateful) do usuário.

Features que Idealizei para o Motor:
1. Rastreadores de velocidade explosiva na volumetria em janelas de 24h e 7D.
2. Razão Temporal de Gastos (Isolando a anomalia do ticket atual vs a média purificada dos 7D).
3. Atributos espaciais e demográficos primitivos.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from core.reporter import evaluate_and_report

def load_data():
    """Carrega os datasets de treino e teste."""
    train = pd.read_csv('data/raw/fraudTrain.csv', index_col=0)
    test = pd.read_csv('data/raw/fraudTest.csv', index_col=0)
    return train, test

def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza Engenharia de Atributos com foco em estado e velocidade.

    Args:
        df: DataFrame original.

    Returns:
        pd.DataFrame: DataFrame com novas features e limpeza aplicada.
    """
    print("Iniciando Engenharia de Atributos...")
    
    # --- BLOCO 1: ORDENAÇÃO BASE ---
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    
    # Precisamos ordenar pelo número de cartão e tempo para construir as features de estado
    df = df.sort_values(by=['cc_num', 'trans_date_trans_time']).reset_index(drop=True)
    
    # 1.1 Conversão de Zip Code para String (Para evitar que o CatBoost trate como número)
    df['zip'] = df['zip'].astype(str)

    # --- BLOCO 2: VELOCITY & ANOMALIA ---
    df_time_indexed = df.set_index('trans_date_trans_time')
    grouped = df_time_indexed.groupby('cc_num')['amt']
    
    # 2.1 Velocity: Frequência de uso do cartão
    count_24h = grouped.rolling('24h').count() - 1
    count_7d = grouped.rolling('7D').count() - 1
    
    # 2.2 Anomalia de Valor: Média Estrita (Excluindo o valor atual da soma)
    # Evita que a transação avaliada distorça a média histórica (Data Leakage)
    sum_7d = grouped.rolling('7D').sum()
    past_sum_7d = sum_7d.values - df_time_indexed['amt'].values 

    mean_7d = past_sum_7d / (count_7d.values + 1e-9) # Pequeno epsilon para evitar divisão por zero
    
    df['trans_count_24h'] = count_24h.values
    df['trans_count_7d'] = count_7d.values
    df['amt_to_mean_7d_ratio'] = df['amt'] / (mean_7d + 0.001)
    
    # Preenchimento de Nulos
    df['trans_count_24h'] = df['trans_count_24h'].fillna(0)
    
    # --- BLOCO 3: ORDENAÇÃO TEMPORAL ESTRITA ---
    df = df.sort_values(by=['trans_date_trans_time']).reset_index(drop=True)
    
    # --- BLOCO 4: EXTRAÇÃO TEMPORAL E DEMOGRÁFICA ---
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_day_of_week'] = df['trans_date_trans_time'].dt.dayofweek

    df['dob'] = pd.to_datetime(df['dob'])
    df['customer_age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365 
    
    # --- BLOCO 5: EXPURGO DE TOXIDADE ---
    cols_to_drop = [
        'trans_num', 'cc_num', 'first', 'last', 'street', 
        'dob', 'trans_date_trans_time', 'unix_time'
    ]
    
    return df.drop(columns=cols_to_drop)

def run_experiment_02():
    print("Iniciando Experimento 02: Stateful Velocity...")
    df_train_raw, df_test_raw = load_data()

    df_train = prep_data(df_train_raw)
    df_test = prep_data(df_test_raw)

    target_col = 'is_fraud' 
    features = [col for col in df_train.columns if col != target_col]
    cat_features = df_train[features].select_dtypes(include=['object', 'category', 'string', 'O']).columns.tolist()

    # Validação Out-of-Time
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
    evaluate_and_report(df_test[target_col], y_probs, df_test['amt'].values, "exp02_stateful_velocity")

if __name__ == "__main__":
    run_experiment_02()
