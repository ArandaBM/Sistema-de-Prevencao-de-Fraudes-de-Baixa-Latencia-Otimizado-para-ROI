import requests
import time
from datetime import datetime, timedelta

# O endereço da API
URL = "http://127.0.0.1:8000/predict"

print("🔥 Iniciando Simulação de Ataque Botnet (Força Bruta)...")

# Pegamos a hora atual como base
base_time = datetime.now()

# Orquestrei este loop para disparar 4 transações com o mesmo cartão, isoladas por míseros 
# 2 segundos, forçando a minha API a engatilhar a defesa de micro-latência.
for i in range(1, 5):
    # Simulando o relógio avançando 2 segundos a cada tentativa
    txn_time = base_time + timedelta(seconds=i*2)
    
    # O Payload JSON que simula a maquininha
    payload = {
        "cc_num": "4111222233334444",      # O mesmo cartão roubado
        "merchant": "FRAUD_GAMES_STORE",   # Lojista de alto risco
        "category": "shopping_net",        # Compra online
        "gender": "M",                     # Gênero do cliente
        "job": "student",                  # Ocupação do cliente
        "amt": 2500.00,                    # Valor alto para testar limite
        "trans_date_trans_time": txn_time.isoformat(), 
        "dob": "1985-05-15T00:00:00",      # Data de nascimento do cliente
        
        # O cliente mora no Brasil (São Paulo)
        "lat": -23.5505, "long": -46.6333, 
        
        # Mas a maquininha (IP) está batendo em Nova York (Distância enorme!)
        "merch_lat": 40.7128, "merch_long": -74.0060 
    }

    print(f"\n[Ataque {i}] Enviando transação às {txn_time.strftime('%H:%M:%S')}...")
    
    # Disparando o stress test contra a minha API
    inicio = time.time()
    response = requests.post(URL, json=payload)
    latencia_api = (time.time() - inicio) * 1000 # Medindo o tempo de resposta em milissegundos
    
    if response.status_code == 200:
        resposta_json = response.json()
        status = resposta_json['status']
        prob = resposta_json['fraud_probability']
        alerta = resposta_json['metrics']['micro_latency_sec']
        
        print(f" -> RESULTADO: {status} (Risco: {prob*100:.2f}%)")
        print(f" -> Micro-latência calculada pela API: {alerta} segundos")
        print(f" -> Tempo de resposta do Servidor: {latencia_api:.2f} ms")
    else:
        print(f" -> ERRO DA API: {response.text}")
        
    # Pausa de 2 segundos reais que utilizo para mimetizar um script malicioso automatizado
    time.sleep(2)

print("\n🛑 Fim do Ataque.")