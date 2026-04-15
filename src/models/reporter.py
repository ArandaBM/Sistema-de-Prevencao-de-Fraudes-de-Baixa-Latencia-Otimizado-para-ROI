"""
Módulo de Reporte Executivo e Diagnóstico de Performance
------------------------------------------------------
Este módulo é responsável por calcular as métricas de performance estatística
(ROC-AUC, PR-AUC) e, mais importante, traduzir essas métricas em Valor de Negócio (ROI).

O foco aqui é o DRE (Demonstrativo de Resultados) do modelo, considerando:
- Money Saved: Dinheiro interceptado em fraudes reais.
- Friction Cost: Custo de oportunidade de bloquear bons clientes.
- Total Operation Cost: Soma de perdas por fraude (FN) e custo de fricção (FP).
"""

import os
import numpy as np
from datetime import datetime
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_and_report(
    y_true: np.ndarray, 
    y_probs: np.ndarray, 
    amt_array: np.ndarray, 
    experiment_name: str, 
    fp_penalty: float = 50.0
) -> None:
    """
    Avalia o modelo estatisticamente e gera um relatório financeiro executivo.

    Args:
        y_true: Rótulos reais (0 ou 1).
        y_probs: Probabilidades preditas pelo modelo.
        amt_array: Valores monetários de cada transação.
        experiment_name: Nome do experimento para o arquivo de log.
        fp_penalty: Custo fixo estimado por cada Falso Positivo (fricção).
    """
    print(f"\n[{experiment_name}] Iniciando Avaliação Estatística e Financeira...")
    
    # 1. Performance Estatística
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    
    # 2. Varredura Financeira (Otimização de Threshold por ROI)
    # Buscamos o ponto onde o custo total da operação é minimizado.
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_threshold = 0.0
    min_total_cost = np.inf 
    best_metrics: Dict[str, Any] = {}
    total_fraud_amt = np.sum(amt_array[y_true == 1])
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        
        # Fraudes não detectadas (Custo Direto)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fraud_loss = np.sum(amt_array[fn_mask])
        
        # Clientes bons bloqueados (Custo de Atendimento/Fricção)
        fp_mask = (y_true == 0) & (y_pred == 1)
        friction_cost = np.sum(fp_mask) * fp_penalty
        
        # Fraudes evitadas (Valor Salvo)
        tp_mask = (y_true == 1) & (y_pred == 1)
        money_saved = np.sum(amt_array[tp_mask])
        
        total_cost = fraud_loss + friction_cost
        
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_threshold = thresh
            best_metrics = {
                'fraud_loss': fraud_loss,
                'friction_cost': friction_cost,
                'money_saved': money_saved,
                'fn_count': np.sum(fn_mask),
                'fp_count': np.sum(fp_mask),
                'tp_count': np.sum(tp_mask)
            }
            
    # 3. Geração do Artefato Físico
    os.makedirs('reports', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/{experiment_name}_{timestamp}.txt"
    
    net_value = best_metrics['money_saved'] - best_metrics['friction_cost']
    
    with open(report_path, 'w', encoding='utf-8') as f:
        lines = [
            "=================================================",
            f"RELATÓRIO EXECUTIVO DE IA: {experiment_name.upper()}",
            f"Data da Execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=================================================",
            "",
            "--- 1. PERFORMANCE ESTATÍSTICA ---",
            f"ROC-AUC: {roc_auc:.4f}",
            f"PR-AUC:  {pr_auc:.4f} (Foco em Precision-Recall)",
            "",
            "--- 2. DEMONSTRATIVO DE RESULTADOS (DRE) ---",
            f"Limiar de Decisão Ideal: {best_threshold:.2f} ({best_threshold*100:.0f}%)",
            f"Cenário Baseline (Total de Fraudes Reais): R$ {total_fraud_amt:,.2f}",
            "-" * 50,
            "A RECEITA (EFICIÊNCIA DO MODELO):",
            f"[+] Fraudes Interceptadas (TP): {best_metrics['tp_count']} transações",
            f"[+] Dinheiro Salvo: R$ {best_metrics['money_saved']:,.2f}",
            "-" * 50,
            "AS DESPESAS (CUSTO OPERACIONAL):",
            f"[-] Fraudes não detectadas (FN): {best_metrics['fn_count']} (Perda: R$ {best_metrics['fraud_loss']:,.2f})",
            f"[-] Falsos Positivos (FP): {best_metrics['fp_count']} (Custo Fricção: R$ {best_metrics['friction_cost']:,.2f})",
            f"CUSTO MÍNIMO DA OPERAÇÃO: R$ {min_total_cost:,.2f}",
            "-" * 50,
            f"VALOR LÍQUIDO GERADO PELO MODELO (ROI): R$ {net_value:,.2f}",
            "================================================="
        ]
        f.write("\n".join(lines))
        
    print(f"[MLOps] Relatório gerado com sucesso em: {report_path}")
    print(f"[ROI] Valor Líquido Gerado: R$ {net_value:,.2f}")