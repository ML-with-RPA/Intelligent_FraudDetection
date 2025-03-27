import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Configura backend não-interativo
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar
import json
import warnings

# Suprimir warnings específicos
warnings.filterwarnings("ignore", category=UserWarning)

def load_predictions(pred_path):
    """Carrega predições com verificação de tamanho"""
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {pred_path}")
    return np.loadtxt(pred_path, dtype=np.int8)

def load_metrics(base, model):
    """Carrega métricas salvas do modelo"""
    metrics_path = f"results/logs/metrics_{base}_{model}.json"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Métricas não encontradas para {model.upper()}")
    with open(metrics_path) as f:
        return json.load(f)

def plot_mcnemar_results(contingency_table, pvalue, base_name, save_dir):
    """Gera e salva o gráfico do teste McNemar"""
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(['RF errado → XGB certo', 'RF certo → XGB errado'], 
                  [contingency_table[0][1], contingency_table[1][0]],
                  color=['#1f77b4', '#ff7f0e'])
    
    plt.ylabel('Número de Casos', fontsize=12)
    plt.title(f'Comparação RF vs XGBoost - {base_name.upper()}\n'
             f'Teste McNemar (valor-p: {pvalue:.4f})', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_mcnemar.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(metrics_rf, metrics_xgb, base_name, save_dir):
    """Gera e salva o gráfico de comparação de métricas"""
    labels = ['Acurácia', 'Precision', 'Recall']
    rf_values = [metrics_rf['accuracy'], metrics_rf['precision'], metrics_rf['recall']]
    xgb_values = [metrics_xgb['accuracy'], metrics_xgb['precision'], metrics_xgb['recall']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, rf_values, width, label='Random Forest')
    plt.bar(x + width/2, xgb_values, width, label='XGBoost')
    plt.xticks(x, labels)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title(f'Comparação de Métricas - {base_name.upper()}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_metrics_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    try:
        print(f"\n🔍 Iniciando análise para base {args.base.upper()}")
        
        # 1. Carregar dados e predições
        target = "Class" if args.base == "creditcard" else "isFraud"
        df_test = pd.read_csv(f"data/processed/amostra_teste_{args.base}.csv", usecols=[target])
        y_true = df_test[target].to_numpy()
        
        preds = {
            'rf': load_predictions(f"results/predictions/predicoes_rf_{args.base}.txt"),
            'xgb': load_predictions(f"results/predictions/predicoes_xgb_{args.base}.txt")
        }

        # Verificar consistência
        assert len(y_true) == len(preds['rf']) == len(preds['xgb']), "Tamanhos inconsistentes!"
        
        # 2. Teste McNemar
        b = np.sum((preds['rf'] != y_true) & (preds['xgb'] == y_true))
        c = np.sum((preds['rf'] == y_true) & (preds['xgb'] != y_true))
        contingency_table = [[0, b], [c, 0]]
        result = mcnemar(contingency_table, exact=False)
        
        print(f"\n📊 Resultado do Teste McNemar:")
        print(f"RF errado → XGB certo: {b} casos")
        print(f"RF certo → XGB errado: {c} casos")
        print(f"Valor-p: {result.pvalue:.4f}")
        print("Conclusão:", "Diferença significativa (p < 0.05)" if result.pvalue < 0.05 
              else "Sem diferença significativa (p ≥ 0.05)")
        
        # 3. Carregar métricas para comparação
        metrics_rf = load_metrics(args.base, 'rf')
        metrics_xgb = load_metrics(args.base, 'xgb')
        
        print(f"\n📈 Métricas do Random Forest:")
        print(f"Acurácia: {metrics_rf['accuracy']:.4f}")
        print(f"Precision: {metrics_rf['precision']:.4f}")
        print(f"Recall: {metrics_rf['recall']:.4f}")
        
        print(f"\n📈 Métricas do XGBoost:")
        print(f"Acurácia: {metrics_xgb['accuracy']:.4f}")
        print(f"Precision: {metrics_xgb['precision']:.4f}")
        print(f"Recall: {metrics_xgb['recall']:.4f}")
        
        # 4. Gerar visualizações
        plot_dir = "results/plots/comparison"
        os.makedirs(plot_dir, exist_ok=True)
        
        plot_mcnemar_results(contingency_table, result.pvalue, args.base, plot_dir)
        plot_metrics_comparison(metrics_rf, metrics_xgb, args.base, plot_dir)
        
        print(f"\n✅ Gráficos salvos em {plot_dir}:")
        print(f"- {args.base}_mcnemar.png")
        print(f"- {args.base}_metrics_comparison.png")

    except Exception as e:
        print(f"\n❌ Erro durante a execução: {str(e)}")
        if 'preds' in locals():
            print("Verifique se os arquivos de predições existem em:")
            print(f"- results/predictions/predicoes_rf_{args.base}.txt")
            print(f"- results/predictions/predicoes_xgb_{args.base}.txt")
        if 'metrics_rf' not in locals():
            print("Certifique-se que ambos modelos foram treinados e geraram métricas")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparação completa entre modelos RF e XGBoost (McNemar + Métricas)"
    )
    parser.add_argument('--base', required=True, choices=['creditcard', 'ieee'],
                      help="Base de dados a ser utilizada")
    args = parser.parse_args()
    
    # Configuração adicional para evitar warnings
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'
    main(args)

#execute    
#python experiments/mcnemar_test.py --base ieee
#python experiments/mcnemar_test.py --base creditcard