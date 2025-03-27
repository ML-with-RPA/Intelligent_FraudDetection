import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# Caminhos dos arquivos
INPUT_FILE = "data/processed/dados_ieee_treinamento_completo.csv"
OUTPUT_FILE = "data/processed/dados_ieee_treinamento_smote.csv"

NUM_SAMPLES = 1000  # Número de amostras desejado (ajuste conforme necessário)

def load_sample():
    """
    Lê o dataset em chunks e extrai uma amostra estratificada de NUM_SAMPLES linhas
    baseada na coluna 'isFraud', sem carregar o dataset completo em memória.
    """
    print("🔹 Lendo amostra estratificada diretamente do dataset...")
    chunksize = 100000  # Tamanho do chunk para leitura parcial
    sample_list = []
    
    for chunk in pd.read_csv(INPUT_FILE, chunksize=chunksize, low_memory=False):
        if "isFraud" not in chunk.columns:
            raise ValueError("A coluna 'isFraud' não foi encontrada no dataset.")
        
        # Amostragem estratificada por 'isFraud'
        sample_chunk = chunk.groupby("isFraud", group_keys=False).apply(
            lambda x: x.sample(min(len(x), NUM_SAMPLES // 2), random_state=42)
        )
        sample_list.append(sample_chunk)
        
        if sum(len(s) for s in sample_list) >= NUM_SAMPLES:
            break

    df_sample = pd.concat(sample_list).head(NUM_SAMPLES)
    print(f"✅ Amostra coletada: {df_sample.shape[0]} linhas.")
    return df_sample

def preprocess_data(df):
    """
    Preenche os valores ausentes nas colunas numéricas com a média e, se ainda houver NaNs, preenche com 0.
    """
    print("🔹 Preenchendo valores ausentes com a média...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Preencher quaisquer NaNs restantes com 0
    df.fillna(0, inplace=True)
    
    if df.isna().sum().sum() > 0:
        raise ValueError("❌ Ainda existem NaNs após o preenchimento!")
    
    return df

def apply_smote(X, y):
    """
    Aplica SMOTE para balancear as classes e retorna os dados reamostrados.
    """
    print("🔹 Aplicando SMOTE para balancear as classes...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def main():
    # Carrega uma amostra estratificada do dataset processado
    df_sample = load_sample()
    
    # Preenche valores ausentes com a média e, se necessário, com 0
    df_sample = preprocess_data(df_sample)
    
    # Separar features e target
    X = df_sample.drop(columns=["isFraud"])
    y = df_sample["isFraud"]
    
    # Aplicar SMOTE
    X_res, y_res = apply_smote(X, y)
    
    # Recria o DataFrame final com os dados reamostrados e adiciona a coluna target
    df_resampled = pd.concat([pd.DataFrame(X_res, columns=X.columns), 
                              pd.Series(y_res, name="isFraud")], axis=1)
    
    # Salva o dataset balanceado
    df_resampled.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Dataset balanceado salvo em: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
