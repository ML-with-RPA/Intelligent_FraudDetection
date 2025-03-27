import os
import pandas as pd

def merge_ieee_data():
    """
    Mescla os arquivos raw da base IEEE CIS Fraud Detection e salva os resultados.
    
    Arquivos raw esperados (na pasta data/raw/):
      - train_transaction.csv
      - train_identity.csv
      - test_transaction.csv
      - test_identity.csv
    
    Os datasets mesclados de treino e teste serão salvos na pasta data/processed/ com os nomes:
      - dados_ieee_treinamento_completo.csv
      - dados_ieee_teste_completo.csv
    
    Além disso, o dataset raw unificado será retornado para ser salvo em data/raw/ieee_data.csv.
    
    Retorna:
      - df_train_merged: DataFrame mesclado de treinamento.
      - df_test_merged: DataFrame mesclado de teste.
      - df_ieee: DataFrame unificado (concatenação de treino e teste).
    """
    
    # Definir caminhos dos arquivos raw
    train_trans_path = "data/raw/train_transaction.csv"
    train_id_path    = "data/raw/train_identity.csv"
    test_trans_path  = "data/raw/test_transaction.csv"
    test_id_path     = "data/raw/test_identity.csv"
    
    # Verificar se os arquivos existem
    for path in [train_trans_path, train_id_path, test_trans_path, test_id_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Arquivo não encontrado: {path}")
    
    # Carregar os dados de treino
    print("Carregando dados de transação e identidade para o treino...")
    df_train_trans = pd.read_csv(train_trans_path)
    df_train_id    = pd.read_csv(train_id_path)
    
    # Mesclar os dados de treino usando 'TransactionID' como chave (left join)
    print("Realizando merge dos dados de treino...")
    df_train_merged = pd.merge(df_train_trans, df_train_id, on="TransactionID", how="left")
    
    # Carregar os dados de teste
    print("Carregando dados de transação e identidade para o teste...")
    df_test_trans = pd.read_csv(test_trans_path)
    df_test_id    = pd.read_csv(test_id_path)
    
    # Mesclar os dados de teste
    print("Realizando merge dos dados de teste...")
    df_test_merged = pd.merge(df_test_trans, df_test_id, on="TransactionID", how="left")
    
    # Gerar o dataset raw unificado (concatenação de treino e teste)
    df_ieee = pd.concat([df_train_merged, df_test_merged], axis=0)
    
    # Salvar os datasets mesclados na pasta data/processed
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    train_output = os.path.join(processed_dir, "dados_ieee_treinamento_completo.csv")
    test_output  = os.path.join(processed_dir, "dados_ieee_teste_completo.csv")
    df_train_merged.to_csv(train_output, index=False)
    df_test_merged.to_csv(test_output, index=False)
    
    print(f"✅ Dados de treino mesclados salvos em: {train_output}")
    print(f"✅ Dados de teste mesclados salvos em: {test_output}")
    
    return df_train_merged, df_test_merged, df_ieee

if __name__ == "__main__":
    _, _, df_ieee = merge_ieee_data()
    # Salvar o dataset raw unificado em data/raw/ieee_data.csv
    os.makedirs("data/raw", exist_ok=True)
    raw_output = os.path.join("data/raw", "ieee_data.csv")
    df_ieee.to_csv(raw_output, index=False)
    print(f"✅ Arquivo raw unificado salvo em {raw_output}")
    print("✅ Experimento IEEE Base de Dados Completa e Prontas para uso!")
