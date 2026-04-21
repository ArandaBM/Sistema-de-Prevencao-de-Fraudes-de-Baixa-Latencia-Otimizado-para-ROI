"""
Módulo de Reporte Executivo e Diagnóstico de Performance
------------------------------------------------------
Desenvolvi este framework central como o "Coração Analítico" de todo o meu lab.
Ele assume a responsabilidade de analisar as métricas cruas de Data Science (PR-AUC) 
e traduzi-las ao Valor de Negócio (ROI / DRE Transacional).

O foco da minha calibração matemática engloba:
- Money Saved: Capital protegido que não cruzou o gateway.
- Friction Cost: Fator financeiro de insatisfação por bloquear falso positivo (UX Loss).
- Net Value (ROI): Lucro Líquido Real gerado como prova de conceito.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
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
    
    rois = []
    
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
        
        net_value_thresh = money_saved - friction_cost
        rois.append(net_value_thresh)
        
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
    
    # 4. Geração do Gráfico de ROI
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, rois, label='ROI (Valor Líquido)', color='#1f77b4', linewidth=2)
    
    # Destacar o melhor ponto de ROI
    max_roi = np.max(rois)
    plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Melhor Limiar: {best_threshold:.2f}')
    plt.scatter([best_threshold], [max_roi], color='red', s=100, zorder=5)
    
    plt.annotate(f'Max ROI:\nR$ {max_roi:,.2f}', 
                 xy=(best_threshold, max_roi), 
                 xytext=(-80, -50) if best_threshold > 0.5 else (20, -50),
                 textcoords='offset points',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    plt.title(f'Otimização de Limiar de Decisão por ROI - {experiment_name}')
    plt.xlabel('Limiar de Decisão (Threshold)')
    plt.ylabel('ROI (R$)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plot_path = f"reports/{experiment_name}_{timestamp}_roi.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
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
