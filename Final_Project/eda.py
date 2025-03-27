def check_missing_values(df):
    """
    Mostra a contagem de valores ausentes (NaN) por coluna no dataset.
    """
    print("Valores ausentes por coluna:\n", df.isna().sum())


def check_column_match(df_train, df_test):
    """
    Verifica se os datasets de treinamento e teste possuem as mesmas colunas (mesmo nome e ordem).
    """
    if list(df_train.columns) == list(df_test.columns):
        print("✅ As colunas de df_train e df_test são as mesmas.")
    else:
        print("⚠️ Diferenças detectadas entre os datasets.")
        cols_train = set(df_train.columns)
        cols_test = set(df_test.columns)
        diff_train = cols_train - cols_test
        diff_test = cols_test - cols_train
        if diff_train:
            print("Colunas presentes em df_train e não em df_test:", diff_train)
        if diff_test:
            print("Colunas presentes em df_test e não em df_train:", diff_test)


def check_dtypes(df):
    """
    Mostra os tipos de dados de cada coluna no dataset.
    """
    print("Tipos de dados:\n", df.dtypes)


def check_unique_values(df):
    """
    Mostra a contagem de valores únicos por coluna no dataset.
    """
    print("Valores únicos por coluna:\n", df.nunique())


def check_skewness(df):
    """
    Mostra a skewness (assimetria) de cada coluna no dataset.
    """
    print("Skewness de cada coluna:\n", df.skew())


def check_outliers(df):
    """
    Mostra uma estimativa da contagem de outliers por coluna.
    Aqui, definimos outlier como valores com z-score > 3 ou < -3.
    """
    def contar_outliers(x):
        return ((x - x.mean()).abs() > 3 * x.std()).sum()
    
    # Aplicar apenas para colunas numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    outliers = df[numeric_cols].apply(contar_outliers)
    print("Outliers por coluna:\n", outliers)


def check_correlation(df):
    """
    Mostra a matriz de correlação entre as colunas numéricas do dataset.
    """
    print("Matriz de Correlação:\n", df.corr())


def plot_correlation_heatmap(df):
    """
    Plota um heatmap da matriz de correlação das colunas numéricas do dataset.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt 
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Matriz de Correlação")
    plt.show()


# Exemplo de uso:
# if __name__ == "__main__":
#     import pandas as pd
#     df = pd.read_csv("seu_arquivo.csv")
#     check_missing_values(df)
#     check_column_match(df, df)  # Exemplo de verificação com o mesmo dataframe
#     check_dtypes(df)
#     check_unique_values(df)
#     check_skewness(df)
#     check_outliers(df)
#     check_correlation(df)
#     plot_correlation_heatmap(df)
