"""
Experimento 01: Baseline Stateless (Sem Estado)
----------------------------------------------
Meu Objetivo: Estabeleci esta baseline de performance utilizando estritamente os 
dados estáticos da transação, cegando o modelo para o histórico do cliente ou lojista.

Construí este modelo propositalmente restrito para servir como minha âncora principal. 
A partir daqui, medirei matematicamente o ganho financeiro incremental (ROI) após projetar 
os meus buffers de State Management In-Memory nas próximas etapas de pesquisa.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from core.reporter import evaluate_and_report

# 1. Carregamento de Dados
def load_data():
    """Carrega os datasets de treino e teste."""
    train = pd.read_csv('data/raw/fraudTrain.csv', index_col=0)
    test = pd.read_csv('data/raw/fraudTest.csv', index_col=0)
    return train, test

def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza básica e garante a integridade temporal (Anti-Leakage).

    Args:
        df: DataFrame original.

    Returns:
        pd.DataFrame: DataFrame processado para o baseline.
    """
    # 1. Conversões de Tempo e garantir a ordem dos acontecimentos (Anti-Leakage)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
    
    # 1.1 Conversão de Zip Code para String (Para evitar que o CatBoost trate como número)
    df['zip'] = df['zip'].astype(str)

    # 2. EXPURGO DE TOXIDADE (Anti-Leakage e Compliance)
    # unix_time é perigoso no baseline pois é estritamente crescente (causa drift)
    cols_to_drop = [
        'trans_num', 'cc_num', 'first', 'last', 'street', 
        'dob', 'trans_date_trans_time', 'unix_time'
    ]
    
    return df.drop(columns=cols_to_drop)

def run_experiment_01():
    print("Iniciando Experimento 01: Baseline Stateless...")
    df_train_raw, df_test_raw = load_data()

    print("Aplicando limpeza e extração temporal...")
    df_train = prep_data(df_train_raw)
    df_test = prep_data(df_test_raw)

    # 3. Definição de Variáveis
    target_col = 'is_fraud' 
    features = [col for col in df_train.columns if col != target_col]

    # Identifica colunas categóricas para o CatBoost
    cat_features = df_train[features].select_dtypes(include=['object', 'category', 'string', 'O']).columns.tolist()

    # 4. Validação Out-of-Time Interna
    # Importante: No mundo real, validamos o passado no presente para prever o futuro.
    split_index = int(len(df_train) * 0.8)
    X_train_int, y_train_int = df_train[features].iloc[:split_index], df_train[target_col].iloc[:split_index]
    X_val_int, y_val_int = df_train[features].iloc[split_index:], df_train[target_col].iloc[split_index:]

    # 5. Otimização de Peso (Alternativa SOTA ao SMOTE)
    # Calculamos o desbalanceamento para que o CatBoost penalize mais o erro na classe minoritária.
    scale_weight = (y_train_int == 0).sum() / (y_train_int == 1).sum()

    # 6. Treinamento
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

    # 7. Avaliação Final no Teste Cego
    X_test = df_test[features]
    y_test = df_test[target_col]
    amt_test = df_test['amt'].values

    y_probs = model.predict_proba(X_test)[:, 1]

    evaluate_and_report(y_test, y_probs, amt_test, "exp01_baseline_stateless")

if __name__ == "__main__":
    run_experiment_01()
