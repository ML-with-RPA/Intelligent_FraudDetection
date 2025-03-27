import sys, os
import argparse
# Adiciona o diretório raiz ao sys.path para que os módulos sejam encontrados
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from models.train_model_RandomForest import treinar_modelo_random_forest as treinar_rf
from models.train_model_XGBoost import treinar_modelo_xgboost as treinar_xgb
from experiments.plot_results import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import logging
import logging_config  # Se já estiver configurado

# Parâmetros para o tamanho das amostras
SAMPLE_SIZE_TRAIN = 5000
SAMPLE_SIZE_TEST = 5000

def sample_stratified_from_file(file_path, target_column, sample_size, chunksize=100000):
    """
    Lê o arquivo CSV em chunks e retorna uma amostra estratificada de 'sample_size' linhas
    baseada na coluna 'target_column'. Se o dataset tiver menos linhas, retorna o dataset completo.
    """
    sample_list = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        if target_column not in chunk.columns:
            raise ValueError(f"A coluna {target_column} não foi encontrada no dataset.")
        # Amostragem estratificada por target_column
        sample_chunk = chunk.groupby(target_column, group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 2), random_state=42)
        )
        sample_list.append(sample_chunk)
        if sum(len(s) for s in sample_list) >= sample_size:
            break
    df_sample = pd.concat(sample_list).head(sample_size)
    return df_sample

def carregar_dataset(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho, low_memory=False)

def registrar_resultados(model_name, accuracy, classification_report, confusion_matrix, train_path, test_path):
    log_dir = "results/logs"
    log_file = os.path.join(log_dir, "model_performance.log")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n==============================\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modelo: {model_name}\n")
        f.write(f"Acurácia: {accuracy:.4f}\n")
        f.write(f"📂 Base de Dados Utilizada:\n")
        f.write(f"   - Treinamento: {os.path.abspath(train_path)}\n")
        f.write(f"   - Teste: {os.path.abspath(test_path)}\n")
        f.write(f"📊 Relatório de Classificação:\n{classification_report}\n")
        f.write(f"📌 Matriz de Confusão:\n{confusion_matrix}\n")
        f.write(f"==============================\n")
        f.flush()
    print("[DEBUG] Log escrito com sucesso.")

def imputar_na(X):
    """
    Preenche os valores ausentes nas colunas numéricas com a média.
    Se ainda houver NaNs, preenche com 0.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    X.fillna(0, inplace=True)
    if X.isna().sum().sum() > 0:
        raise ValueError("❌ Ainda existem NaNs após a imputação!")
    return X

def main(args):
    # Defina os caminhos dos arquivos processados completos
    train_file = "data/processed/dados_ieee_treinamento_completo.csv"
    test_file = "data/processed/dados_ieee_teste_completo.csv"
    
    print("Diretório atual:", os.getcwd())
    
    # Extrair amostras estratificadas para treino e teste
    print("🔹 Extraindo amostra estratificada para treinamento...")
    df_train_sample = sample_stratified_from_file(train_file, "isFraud", SAMPLE_SIZE_TRAIN)
    print(f"✅ Amostra de treinamento coletada: {df_train_sample.shape[0]} linhas.")
    
    print("🔹 Extraindo amostra estratificada para teste...")
    df_test_sample = sample_stratified_from_file(test_file, "isFraud", SAMPLE_SIZE_TEST)
    print(f"✅ Amostra de teste coletada: {df_test_sample.shape[0]} linhas.")
    
    # Separar features e target para o conjunto de treinamento
    X_train = df_train_sample.drop("isFraud", axis=1)
    y_train = df_train_sample["isFraud"]
    
    # Imputar valores ausentes no conjunto de treinamento
    print("🔹 Imputando valores ausentes no conjunto de treinamento...")
    X_train = imputar_na(X_train)
    
    # Aplicar SMOTE somente no conjunto de treinamento
    print("🔹 Aplicando SMOTE no conjunto de treinamento...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    df_train_smote = pd.concat([pd.DataFrame(X_train_res, columns=X_train.columns),
                                pd.Series(y_train_res, name="isFraud")], axis=1)
    
    # Preparar o conjunto de teste (amostra extraída) sem SMOTE
    X_test = df_test_sample.drop("isFraud", axis=1)
    y_test = df_test_sample["isFraud"]
    df_test_final = pd.concat([X_test, y_test], axis=1)
    
    # Escolher o modelo com base no argumento
    if args.model == "rf":
        print("🔹 Treinando modelo Random Forest...")
        modelo, predicoes, y_pred, accuracy, class_report, conf_matrix = treinar_rf(
            df_train_smote,
            df_test_final,
            target_column="isFraud",
            salvar_modelo=True
        )
        model_name = "Random Forest (IEEE SMOTE Treino)"
    elif args.model == "xgb":
        print("🔹 Treinando modelo XGBoost...")
        modelo, predicoes, y_pred, accuracy, class_report, conf_matrix = treinar_xgb(
            df_train_smote,
            df_test_final,
            target_column="isFraud",
            salvar_modelo=True
        )
        model_name = "XGBoost (IEEE SMOTE Treino)"
    else:
        raise ValueError("Modelo desconhecido. Escolha 'rf' para Random Forest ou 'xgb' para XGBoost.")
    
    print("✅ Modelo treinado e salvo com sucesso!")
    
    registrar_resultados(model_name, accuracy, class_report, conf_matrix, train_file, test_file)
    
    os.makedirs("results/plots", exist_ok=True)
    print("📊 Gerando gráficos de avaliação...")
    plot_confusion_matrix(y_test, predicoes)
    plot_roc_curve(modelo, X_test, y_test)
    plot_precision_recall_curve(modelo, X_test, y_test)
    print("✅ Gráficos gerados e salvos em results/plots/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina um modelo para a base IEEE balanceada.")
    parser.add_argument('--model', type=str, default='rf', help="Modelo a ser treinado: 'rf' para Random Forest, 'xgb' para XGBoost")
    args = parser.parse_args()
    
    main(args)
    print("✅ Experimento IEEE concluído com sucesso!")

# # Treinamento com a base IEEE utilizando o modelo Random Forest
#EXECUTE> python experiments/train_ieee.py --model rf

# # Treinamento com a base IEEE utilizando o modelo XGBoost
#EXECUTE> python experiments/train_ieee.py --model xgb
