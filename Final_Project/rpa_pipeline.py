# process_creditcard_data: Processa o dataset de fraude com cartão de crédito.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import normalize_data
from config import TRAIN_OUTPUT, TEST_OUTPUT  # Caminhos definidos no config.py

def process_creditcard_data(df):
    """
    Prepara o dataset de fraude com cartão de crédito.
    - Separa as features e o target (coluna "Class").
    - Divide os dados em treino (70%) e teste (30%).
    - Normaliza os dados.
    - Salva os conjuntos intermediários e finais na pasta data/processed/.
    Retorna:
      df_train, df_test: DataFrames de treinamento e teste finais.
    """
    # Separar features e target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Dividir os dados (70% treino, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Normalizar os dados
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    # Salvar os conjuntos de treino e teste intermediários
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/Y_train.csv", index=False)
    y_test.to_csv("data/processed/Y_test.csv", index=False)

    # Concatenar e salvar os datasets finais
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    df_train.to_csv(TRAIN_OUTPUT, index=False)
    df_test.to_csv(TEST_OUTPUT, index=False)
    
    print(f"✅ Dados processados e salvos em {TRAIN_OUTPUT} e {TEST_OUTPUT}")
    return df_train, df_test


# process_ieee_data: Processa a base IEEE CIS Fraud.
import os
from sklearn.model_selection import train_test_split
from preprocessing import remove_irrelevant_columns, normalize_data
from config import TRAIN_OUTPUT, TEST_OUTPUT  # Se desejar usar variáveis do config, ou definir novos nomes

def process_ieee_data(df):
    """
    Processa a base IEEE CIS Fraud removendo colunas irrelevantes, normalizando os dados,
    dividindo em conjuntos de treino e teste e salvando os resultados com novos nomes.
    Retorna:
      df_train_ieee, df_test_ieee: DataFrames de treinamento e teste finais para a base IEEE.
    """
    # Remover colunas irrelevantes (ajuste conforme necessário para a base IEEE)
    colunas_irrelevantes = ['TransactionID', 'TransactionDT']
    df = remove_irrelevant_columns(df, colunas_irrelevantes)
    
    # Normalizar os dados
    df = normalize_data(df)
    
    # Separar features e target (supondo que o rótulo seja "isFraud")
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    
    # Dividir os dados (70% treino, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Salvar arquivos intermediários para referência (opcional)
    X_train.to_csv("data/processed/X_ieee_train.csv", index=False)
    X_test.to_csv("data/processed/X_ieee_test.csv", index=False)
    y_train.to_csv("data/processed/Y_ieee_train.csv", index=False)
    y_test.to_csv("data/processed/Y_ieee_test.csv", index=False)
    
    # Concatenar features e target para os datasets finais
    df_train_ieee = pd.concat([X_train, y_train], axis=1)
    df_test_ieee  = pd.concat([X_test, y_test], axis=1)
    
    # Definir os caminhos para salvar os dados IEEE processados
    train_output_ieee = "data/processed/dados_ieee_treinamento_completo.csv"
    test_output_ieee  = "data/processed/dados_ieee_teste_completo.csv"
    
    # Garantir que o diretório "data/processed" exista
    os.makedirs("data/processed", exist_ok=True)
    
    # Salvar os datasets finais
    df_train_ieee.to_csv(train_output_ieee, index=False)
    df_test_ieee.to_csv(test_output_ieee, index=False)
    
    print(f"✅ Dados IEEE processados e salvos em:\n   - {train_output_ieee}\n   - {test_output_ieee}")
    
    return df_train_ieee, df_test_ieee


# Exemplo de uso:
# Se você tiver um arquivo raw da base IEEE, por exemplo "data/raw/ieee_data.csv", você pode fazer:
# df_ieee = pd.read_csv("data/raw/ieee_data.csv")
# df_train_ieee, df_test_ieee = process_ieee_data(df_ieee)
