# 1. IMAGEM BASE
# Optei pela versão 'slim' do Python 3.11, uma vez que ela é cravada para nuvem e encolhe brutalmente a superfície de dependências em relação à imagem default completa.
FROM python:3.11-slim

# 2. DIRETÓRIO DE TRABALHO
WORKDIR /app

# 3. OTIMIZAÇÃO DE CACHE
COPY requirements.txt .

# 4. INSTALAÇÃO DE DEPENDÊNCIAS
RUN pip install --no-cache-dir -r requirements.txt

# 5. TRANSFERÊNCIA DO SISTEMA
COPY src/ /app/src/
COPY models/ /app/models/

# 6. PORTA DE SAÍDA
EXPOSE 8000

# 7. MOTOR DE IGNIÇÃO
CMD ["uvicorn", "src.api.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]