import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score
)

import json
from sklearn.model_selection import learning_curve

def plot_correlation_heatmap(df, save_path=None, figsize=(20, 15)):
    """
    Plota e salva um heatmap da matriz de correlação com aprimoramentos.
    """
    plt.figure(figsize=figsize)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=.5, cbar_kws={"shrink": 0.8}
    )
    
    plt.title("Matriz de Correlação entre Variáveis", pad=20, fontsize=14)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Gráfico salvo em: {save_path}")
    
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name="Modelo", save_path=None):
    """
    Plota e salva a matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Legítima", "Fraude"]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, annot_kws={"size": 14}
    )
    
    plt.title(f"Matriz de Confusão - {model_name}", fontsize=14)
    plt.xlabel("Predição")
    plt.ylabel("Real")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Gráfico salvo em: {save_path}")
    
    plt.show()
    plt.close()

def plot_roc_curve(model, X_test, y_test, model_name="Modelo", save_path=None):
    """
    Plota e salva a curva ROC.
    """
    if not hasattr(model, "predict_proba"):
        print("⚠️ Modelo não suporta probabilidades. ROC não gerada.")
        return

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.title(f'Curva ROC - {model_name}', fontsize=14)
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.legend(loc="lower right")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Gráfico salvo em: {save_path}")

    plt.show()
    plt.close()

def plot_precision_recall_curve(model, X_test, y_test, model_name="Modelo", save_path=None):
    """
    Plota e salva a curva Precision-Recall.
    """
    if not hasattr(model, "predict_proba"):
        print("⚠️ Modelo não suporta probabilidades. Curva não gerada.")
        return

    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Curva Precision-Recall (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="upper right")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Gráfico salvo em: {save_path}")

    plt.show()
    plt.close()
def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Não Fraude', 'Fraude'],
               yticklabels=['Não Fraude', 'Fraude'])
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve

def plot_feature_importance(model, feature_names, save_path):
    """Gera gráfico de importância das features"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Importância das Features")
        plt.barh(range(len(importances)), importances[indices], align='center')
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.xlabel("Importância Relativa")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def plot_learning_curve(estimator, X, y, save_path):
    """Gera curva de aprendizado"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Treino')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Validação')
    plt.title('Curva de Aprendizado')
    plt.xlabel('Tamanho do Conjunto de Treino')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_metrics_comparison(metrics_rf, metrics_xgb, save_path):
    """Compara métricas entre modelos"""
    labels = ['Acurácia', 'Precision', 'Recall']
    rf_values = [metrics_rf['accuracy'], metrics_rf['precision'], metrics_rf['recall']]
    xgb_values = [metrics_xgb['accuracy'], metrics_xgb['precision'], metrics_xgb['recall']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, rf_values, width, label='Random Forest')
    plt.bar(x + width/2, xgb_values, width, label='XGBoost')
    plt.xticks(x, labels)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title('Comparação de Modelos')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    