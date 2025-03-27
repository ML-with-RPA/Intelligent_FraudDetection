import pandas as pd
from config import DATA_PATH_TRAIN, DATA_PATH_TEST, CREDIT_CARD_DATA

def load_ieee_data():
    """Carrega os dados IEEE CIS Fraud"""
    df_train = pd.read_csv(DATA_PATH_TRAIN)
    df_test = pd.read_csv(DATA_PATH_TEST)
    return df_train, df_test

def load_creditcard_data():
    """Carrega o dataset de fraude com cartão de crédito"""
    df = pd.read_csv(CREDIT_CARD_DATA)
    return df
