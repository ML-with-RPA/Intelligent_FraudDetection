from sklearn.preprocessing import StandardScaler
import numpy as np

def remove_irrelevant_columns(df, columns):
    """
    Remove as colunas especificadas do dataset.
    
    Parâmetros:
      - df: DataFrame a ser processado.
      - columns: Lista de nomes de colunas a serem removidas.
    
    Retorna:
      - DataFrame sem as colunas irrelevantes.
    """
    return df.drop(columns=columns, errors='ignore')

def normalize_data(df):
    """
    Normaliza as colunas numéricas do dataset utilizando StandardScaler.
    
    Para evitar divisão por zero, colunas com desvio padrão zero são removidas temporariamente.
    """
    scaler = StandardScaler()
    # Seleciona as colunas numéricas
    colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
    df_numeric = df[colunas_numericas]
    
    # Remover colunas com desvio padrão zero
    cols_validas = df_numeric.std()[df_numeric.std() != 0].index
    df_numeric = df_numeric[cols_validas]
    
    # Normalizar as colunas válidas com tratamento de avisos
    with np.errstate(invalid='ignore'):
        df_numeric_normalized = scaler.fit_transform(df_numeric)
    
    # Atualiza o DataFrame copiando os dados normalizados para as colunas válidas
    df_normalized = df.copy()
    df_normalized[cols_validas] = df_numeric_normalized
    return df_normalized

