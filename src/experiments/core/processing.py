import numpy as np
import pandas as pd

def haversine_vectorized(lat1: float, lon1: float, lat2: float, lon2: float) -> np.ndarray:
    """
    Minha implementação vetorizada em Numpy para triturar distâncias esféricas (Haversine).
    Projetei esta função contornando os helpers triviais para garantir nanosegundos em D-0.

    Args:
        lat1, lon1: Coordenadas do primeiro ponto (ex: cliente).
        lat2, lon2: Coordenadas do segundo ponto (ex: estabelecimento).

    Returns:
        np.ndarray: Distâncias em quilômetros.
    """
    # Converte graus para radianos
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Diferenças
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Fórmula de Haversine
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Raio da Terra em quilômetros (6371 km)
    return 6371 * c

def clean_base_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza inicial e padronização dos dados brutos.
    
    Args:
        df: DataFrame original do dataset de fraudes.
        
    Returns:
        pd.DataFrame: DataFrame limpo e ordenado temporalmente.
    """
    df = df.copy()
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
    return df
