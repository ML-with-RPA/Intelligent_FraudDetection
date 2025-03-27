import sys
import os
# Configurar paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Adicione no início com os outros imports
import matplotlib
matplotlib.use('Agg')  # Para evitar problemas com tkinter
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import argparse
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
import psutil
from experiments.plot_results import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_learning_curve
)



# Módulos customizados
from models.train_model_RandomForest import treinar_modelo_random_forest as treinar_rf
from models.train_model_XGBoost import treinar_modelo_xgboost as treinar_xgb
from experiments.plot_results import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)
from envioEmail import AlertaMLOutlook

# Configurações
SAMPLE_SIZE_TRAIN = 5000
SAMPLE_SIZE_TEST = 2000
RANDOM_STATE = 42
NAN_THRESHOLD = 0.5  # Remover colunas com mais de 50% de valores NaN

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("results/logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def monitorar_memoria():
    """Retorna uso de memória em MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def tratar_nan(df, target_column):
    """Realiza tratamento completo de valores NaN no dataframe"""
    logger.info("Iniciando tratamento de valores NaN...")
    
    # 1. Remover linhas com NaN na target
    df = df.dropna(subset=[target_column])
    
    # 2. Remover colunas com muitos NaN
    df = df.dropna(thresh=len(df)*NAN_THRESHOLD, axis=1)
    
    # 3. Preencher valores numéricos
    numeric_cols = df.select_dtypes(include=np.number).columns.difference([target_column])
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 4. Preencher valores categóricos
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'UNKNOWN'
            df[col] = df[col].fillna(mode_val)
    
    logger.info(f"Tratamento de NaN completo. Shape final: {df.shape}")
    return df

def criar_amostras(base):
    """Cria amostras de treino e teste se não existirem"""
    base_path = f"data/processed/amostra_treino_{base}.csv"
    test_path = f"data/processed/amostra_teste_{base}.csv"
    
    if os.path.exists(base_path) and os.path.exists(test_path):
        logger.info(f"Amostras já existem para {base}")
        return
    
    logger.info(f"Criando amostras para {base}...")
    
    # Carregar dataset completo
    if base == "creditcard":
        df = pd.read_csv("data/processed/dados_treinamento_completo.csv")
        target = "Class"
    else:  # IEEE
        df = pd.read_csv("data/processed/dados_ieee_treinamento_completo.csv")
        target = "isFraud"
    
    # Tratar NaN
    df = tratar_nan(df, target)
    
    # Balancear as classes
    fraudes = df[df[target] == 1]
    nao_fraudes = df[df[target] == 0].sample(
        n=min(len(fraudes)*2, len(df[df[target] == 0])), 
        random_state=RANDOM_STATE
    )
    df_balanced = pd.concat([fraudes, nao_fraudes]).sample(frac=1, random_state=RANDOM_STATE)
    
    # Dividir em treino e teste
    train, test = train_test_split(df_balanced, test_size=0.3, random_state=RANDOM_STATE)
    
    # Amostrar conforme tamanhos especificados
    train_sample = train.sample(min(SAMPLE_SIZE_TRAIN, len(train)), random_state=RANDOM_STATE)
    test_sample = test.sample(min(SAMPLE_SIZE_TEST, len(test)), random_state=RANDOM_STATE)
    
    # Salvar amostras
    os.makedirs("data/processed", exist_ok=True)
    train_sample.to_csv(base_path, index=False)
    test_sample.to_csv(test_path, index=False)
    logger.info(f"Amostras salvas em {base_path} e {test_path}")

def carregar_dataset(base):
    """Carrega dataset a partir dos arquivos de amostra"""
    criar_amostras(base)
    
    base_path = f"data/processed/amostra_treino_{base}.csv"
    test_path = f"data/processed/amostra_teste_{base}.csv"

    logger.info(f"Carregando dataset de treino e teste para {base}...")
    return pd.read_csv(base_path), pd.read_csv(test_path)

def preparar_dados(df_train, df_test, target_column):
    """Prepara os dados para treinamento com tratamento de NaN"""
    logger.info("Processando dados...")
    
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]
    
    # Verificar e tratar NaN
    if X_train.isnull().any().any() or y_train.isnull().any():
        logger.warning("Valores NaN encontrados - Aplicando tratamento final")
        
        # Preencher numéricos com mediana do treino
        numeric_cols = X_train.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            medians = X_train[numeric_cols].median()
            X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
            X_test[numeric_cols] = X_test[numeric_cols].fillna(medians)
        
        # Preencher categóricos com moda do treino
        categorical_cols = X_train.select_dtypes(exclude=np.number).columns
        for col in categorical_cols:
            mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else 'UNKNOWN'
            X_train[col] = X_train[col].fillna(mode_val)
            X_test[col] = X_test[col].fillna(mode_val)
    
    return X_train, y_train, X_test, y_test

def aplicar_smote(X_train, y_train, random_state=RANDOM_STATE):
    """Aplica SMOTE apenas se os dados estiverem limpos"""
    if X_train.isnull().any().any() or y_train.isnull().any():
        logger.error("Dados ainda contêm NaN após tratamento - usando dados originais")
        return pd.concat([X_train, y_train], axis=1)
    
    logger.info("Aplicando SMOTE...")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    df_smote = pd.DataFrame(X_res, columns=X_train.columns)
    df_smote[y_train.name] = y_res
    return df_smote

# Adicione estas funções (pode colocar junto com as outras funções auxiliares)
def plot_feature_importance(model, feature_names, save_path):
    """Gera gráfico de importância de features"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Importância das Features")
        plt.barh(range(len(importances)), importances[indices], align='center')
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.xlabel("Importância Relativa")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def plot_learning_curve(estimator, X, y, save_path):
    """Gera curva de aprendizado"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Treino')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Validação')
    plt.title('Curva de Aprendizado')
    plt.xlabel('Tamanho do Conjunto de Treino')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main(args):
    try:
        logger.info(f"Iniciando treinamento - Base: {args.base} | Modelo: {args.model}")
        logger.info(f"Memória inicial: {monitorar_memoria():.2f} MB")

        # Carregar e preparar dados
        df_train, df_test = carregar_dataset(args.base.lower())
        target_column = "Class" if args.base.lower() == "creditcard" else "isFraud"
        
        X_train, y_train, X_test, y_test = preparar_dados(df_train, df_test, target_column)
        df_train_smote = aplicar_smote(X_train, y_train)

        # Treinar modelo
        logger.info(f"Treinando modelo {args.model.upper()}...")
        if args.model.lower() == "rf":
            modelo, preds, *_ = treinar_rf(
                df_train_smote,
                pd.concat([X_test, y_test], axis=1),
                target_column=target_column,
                salvar_modelo=True
            )
        else:
            modelo, preds, *_ = treinar_xgb(
                df_train_smote,
                pd.concat([X_test, y_test], axis=1),
                target_column=target_column,
                salvar_modelo=True
            )

        # Calcular métricas
        logger.info("Calculando métricas...")
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        conf_matrix = confusion_matrix(y_test, preds)

        # Salvar resultados
        logger.info("Salvando resultados...")
        os.makedirs("results/logs", exist_ok=True)
        with open("results/logs/model_performance.log", "a") as f:
            f.write(f"\n{'='*40}\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base: {args.base} | Modelo: {args.model}\n")
            f.write(f"Acurácia: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Matriz de Confusão:\n{conf_matrix}\n")
            f.write(f"{'='*40}\n")

         # ... [código existente até o treinamento do modelo] ...

        # Calcular métricas (código existente)
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        conf_matrix = confusion_matrix(y_test, preds)

        # Salvar resultados (código existente)
        logger.info("Salvando resultados...")
        os.makedirs("results/logs", exist_ok=True)
        with open("results/logs/model_performance.log", "a") as f:
            f.write(f"\n{'='*40}\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base: {args.base} | Modelo: {args.model}\n")
            f.write(f"Acurácia: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Matriz de Confusão:\n{conf_matrix}\n")
            f.write(f"{'='*40}\n")

        # ==============================================
        # GERAR VISUALIZAÇÕES (ATUALIZADO)
        # ==============================================
        logger.info("Gerando visualizações...")
        plot_dir = f"results/plots/{args.base}_{args.model}"
        os.makedirs(plot_dir, exist_ok=True)

        # Gráficos originais
        plot_confusion_matrix(y_test, preds, model_name=args.model.upper(),
                            save_path=os.path.join(plot_dir, "confusion_matrix.png"))
        plot_roc_curve(modelo, X_test, y_test, model_name=args.model.upper(),
                      save_path=os.path.join(plot_dir, "roc_curve.png"))
        plot_precision_recall_curve(modelo, X_test, y_test, model_name=args.model.upper(),
                                  save_path=os.path.join(plot_dir, "precision_recall_curve.png"))

        # Novos gráficos
        plot_feature_importance(modelo, X_train.columns, 
                              save_path=os.path.join(plot_dir, "feature_importance.png"))
        plot_learning_curve(modelo, X_train, y_train,
                          save_path=os.path.join(plot_dir, "learning_curve.png"))

        # Salvar métricas para comparação posterior
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'conf_matrix': conf_matrix.tolist()
        }
        with open(f"results/logs/metrics_{args.base}_{args.model}.json", "w") as f:
            json.dump(metrics, f)

        # ... [restante do código existente] ...
        # Salvar predições
        predictions_dir = "results/predictions"
        os.makedirs(predictions_dir, exist_ok=True)
        np.savetxt(os.path.join(predictions_dir, f"predicoes_{args.model}_{args.base}.txt"), preds, fmt="%d")
        logger.info(f"Predições salvas em {predictions_dir}")
        
        # Enviar notificação por e-mail
        logger.info("Enviando notificação...")
        alerta_outlook = AlertaMLOutlook()
        alerta_outlook.enviar_alerta_ml(graficos_dir=plot_dir)
        logger.info(f"Processo concluído! Memória final: {monitorar_memoria():.2f} MB")

    except Exception as e:
        logger.error(f"Erro no pipeline: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento")
    parser.add_argument("--base", required=True, choices=["creditcard", "ieee"])
    parser.add_argument("--model", required=True, choices=["rf", "xgb"])
    args = parser.parse_args()
    main(args)


# # Treinamento com Creditcard e XGBoost
#EXECUTE> python experiments/train.py --base creditcard --model xgb

# # Treinamento com Creditcard e Random Forest
#EXECUTE> python experiments/train.py --base creditcard --model rf

# # Treinamento com IEEE e XGBoost
#EXECUTE> python experiments/train.py --base ieee --model xgb

# # Treinamento com IEEE e Random Forest
#EXECUTE> python experiments/train.py --base ieee --model rf
