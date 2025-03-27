import logging_config  # Isso configura o logging automaticamente

from data_loader import load_ieee_data, load_creditcard_data
from eda import check_missing_values, check_column_match, check_dtypes
from preprocessing import remove_irrelevant_columns, normalize_data
from rpa_pipeline import process_creditcard_data
import pandas as pd

### IEEE CIS Fraud Dataset ###
print("🔹 Carregando dados IEEE CIS Fraud...")
df_train, df_test = load_ieee_data()

# Análise exploratória
print("🔹 Verificando valores ausentes...")
check_missing_values(df_train)
check_missing_values(df_test)

print("🔹 Verificando colunas...")
check_column_match(df_train, df_test)

# Pré-processamento
print("🔹 Removendo colunas irrelevantes...")
colunas_irrelevantes = ['TransactionID', 'TransactionDT']
df_train = remove_irrelevant_columns(df_train, colunas_irrelevantes)
df_test = remove_irrelevant_columns(df_test, colunas_irrelevantes)

print("🔹 Normalizando dados...")
df_train = normalize_data(df_train)
df_test = normalize_data(df_test)

print("✅ IEEE CIS Fraud processado!\n")

### Fraud Credit Card Dataset ###
print("🔹 Carregando dataset de fraude com cartão de crédito...")
df_credit = load_creditcard_data()

print("🔹 Processando dados...")
# A função process_creditcard_data deve processar os dados e salvar os arquivos processados
process_creditcard_data(df_credit)

print("✅ Pipeline finalizado com sucesso!")
