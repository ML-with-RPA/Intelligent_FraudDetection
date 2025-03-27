import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def treinar_modelo_random_forest(df_train, df_test, target_column="Class", salvar_modelo=False):
    """
    Treina um modelo Random Forest e retorna o modelo treinado, predições e métricas.
    
    Parâmetros:
      - df_train: DataFrame de treinamento (com a coluna target).
      - df_test: DataFrame de teste (com a coluna target).
      - target_column: Nome da coluna target. Default: "Class".
      - salvar_modelo: Se True, salva o modelo treinado em 'results/saved_models/modelo_random_forest.pkl'.
      
    Retorna:
      - modelo: Modelo Random Forest treinado.
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
        X_train = df_train.drop(columns=[target_column]).astype('float32')  # Otimização de memória
        y_train = df_train[target_column].astype('int32')
        X_test = df_test.drop(columns=[target_column]).astype('float32')
        y_test = df_test[target_column].astype('int32')

        # Configuração otimizada para grandes datasets
        modelo = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            n_jobs=-1,  # Usar todos os núcleos
            random_state=42,
            class_weight='balanced'  # Útil para dados desbalanceados
        )
        
        # Treinamento com medição de tempo
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

        # Salvamento do modelo
        if salvar_modelo:
            os.makedirs("results/saved_models", exist_ok=True)
            model_path = os.path.join("results/saved_models", "modelo_random_forest.pkl")
            joblib.dump(modelo, model_path)
            print(f"💾 Modelo salvo em: {model_path}")

        return modelo, predicoes, y_test, accuracy, report, conf_matrix

    except Exception as e:
        print(f"\n❌ Erro crítico durante o treinamento:")
        print(f"Tipo do erro: {type(e).__name__}")
        print(f"Mensagem: {str(e)}")
        print("\n🛠️ Ações recomendadas:")
        print("- Verifique os tipos de dados das colunas (df.dtypes)")
        print("- Execute df.isna().sum() para identificar valores faltantes")
        print("- Reduza n_estimators/max_depth para economizar memória")
        raise
