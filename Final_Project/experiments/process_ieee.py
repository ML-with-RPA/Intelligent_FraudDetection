import sys, os
# Adiciona o diretório raiz ao sys.path para que os módulos sejam encontrados
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import remove_irrelevant_columns, normalize_data

def process_ieee_data():
    """
    Processa a base IEEE CIS Fraud:
      - Carrega o arquivo raw unificado da base IEEE (data/raw/ieee_data.csv)
      - Remove colunas irrelevantes
      - Converte colunas não numéricas (exceto "isFraud") em variáveis dummy
      - Normaliza apenas as features (mantendo "isFraud" inalterado)
      - Remove linhas onde o target ("isFraud") está faltando
      - Divide os dados em conjuntos de treino e teste e salva os arquivos processados
    Retorna:
      - df_train_ieee, df_test_ieee
    """
    print("🔹 Processando dados IEEE...")
    
    raw_path = "data/raw/ieee_data.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {raw_path}")
    
    # Use low_memory=False para evitar warnings
    df = pd.read_csv(raw_path, low_memory=False)
    
    # Remover colunas irrelevantes
    colunas_irrelevantes = ['TransactionID', 'TransactionDT']
    df = remove_irrelevant_columns(df, colunas_irrelevantes)
    
    # Converter todas as colunas não numéricas (exceto "isFraud") para string
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if "isFraud" in non_numeric_cols:
        non_numeric_cols.remove("isFraud")
    df[non_numeric_cols] = df[non_numeric_cols].astype(str)
    
    # Converter as colunas categóricas em dummies
    df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)
    
    # Remover linhas onde o target "isFraud" é NaN
    df = df.dropna(subset=["isFraud"])
    
    # Separar target e features
    target = df["isFraud"]
    features = df.drop("isFraud", axis=1)
    
    # Normalizar apenas as features
    features = normalize_data(features)
    
    # Reconstituir o DataFrame com o target inalterado
    df = pd.concat([features, target], axis=1)
    
    # Dividir os dados em treino (70%) e teste (30%)
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Criar diretório para dados processados
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Salvar arquivos intermediários
    X_train.to_csv(os.path.join(processed_dir, "X_ieee_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_ieee_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "Y_ieee_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "Y_ieee_test.csv"), index=False)
    
    # Concatenar para os datasets finais
    df_train_ieee = pd.concat([X_train, y_train], axis=1)
    df_test_ieee  = pd.concat([X_test, y_test], axis=1)
    
    # Definir caminhos de saída para os dados processados IEEE
    train_output_ieee = os.path.join(processed_dir, "dados_ieee_treinamento_completo.csv")
    test_output_ieee  = os.path.join(processed_dir, "dados_ieee_teste_completo.csv")
    
    df_train_ieee.to_csv(train_output_ieee, index=False)
    df_test_ieee.to_csv(test_output_ieee, index=False)
    
    print(f"✅ Dados IEEE processados e salvos em:\n   - {train_output_ieee}\n   - {test_output_ieee}")
    
    return df_train_ieee, df_test_ieee

if __name__ == "__main__":
    process_ieee_data()
