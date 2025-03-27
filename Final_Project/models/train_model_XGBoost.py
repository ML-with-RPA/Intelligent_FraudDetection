import time
import joblib
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def treinar_modelo_xgboost(df_train, df_test, target_column="isFraud", salvar_modelo=False):
    """
    Treina um modelo XGBoost e retorna o modelo treinado, predições e métricas de avaliação.
    
    Parâmetros:
      - df_train: DataFrame de treinamento (com a coluna target).
      - df_test: DataFrame de teste (com a coluna target).
      - target_column: Nome da coluna target. Default: "isFraud".
      - salvar_modelo: Se True, salva o modelo treinado em 'results/saved_models/modelo_xgboost.pkl'.
      
    Retorna:
      - modelo: Modelo XGBoost treinado.
      - predicoes: Vetor de predições para o conjunto de teste.
      - y_test: Vetor de rótulos reais.
      - accuracy: Acurácia do modelo.
      - report: Relatório de classificação.
      - conf_matrix: Matriz de confusão.
    """
    try:
        # Verificação de segurança das colunas
        if target_column not in df_train.columns:
            raise ValueError(f"Coluna alvo '{target_column}' não encontrada no treino!")
        if target_column not in df_test.columns:
            raise ValueError(f"Coluna alvo '{target_column}' não encontrada no teste!")

        # Debug: Verificar estrutura dos dados
        print("\n🔍 Debug - Primeiras linhas do treino:")
        print(df_train.head(2))
        print("\n🔍 Debug - Primeiras linhas do teste:")
        print(df_test.head(2))

        # Separar features e target
        X_train = df_train.drop(columns=[target_column]).astype('float32')  # Forçar tipo numérico
        y_train = df_train[target_column].astype('int32')
        X_test = df_test.drop(columns=[target_column]).astype('float32')
        y_test = df_test[target_column].astype('int32')

        # Inicializa e treina o modelo
        modelo = XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            tree_method='hist'  # Otimização para grandes datasets
        )
        
        start_time = time.time()
        modelo.fit(X_train, y_train)
        print(f"✅ Modelo treinado em {time.time() - start_time:.2f}s")

        # Predições e métricas
        predicoes = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, predicoes)
        report = classification_report(y_test, predicoes)
        conf_matrix = confusion_matrix(y_test, predicoes)

        print(f"\n🎯 Acurácia: {accuracy:.4f}")
        print("📊 Relatório de Classificação:\n", report)
        print("📌 Matriz de Confusão:\n", conf_matrix)

        # Salvar modelo se solicitado
        if salvar_modelo:
            os.makedirs("results/saved_models", exist_ok=True)
            model_path = os.path.join("results/saved_models", "modelo_xgboost.pkl")
            joblib.dump(modelo, model_path)
            print(f"💾 Modelo salvo em: {model_path}")

        return modelo, predicoes, y_test, accuracy, report, conf_matrix

    except Exception as e:
        print(f"\n❌ Erro crítico durante o treinamento:")
        print(f"Tipo do erro: {type(e).__name__}")
        print(f"Mensagem: {str(e)}")
        print("\n🛠️ Ações recomendadas:")
        print("- Verifique se a coluna target existe nos DataFrames")
        print("- Confira os tipos de dados das features (devem ser numéricos)")
        print("- Valores NaN/Infinitos nas features? Use SimpleImputer")
        raise
    