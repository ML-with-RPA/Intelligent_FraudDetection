# 🏦 Automacao Inteligente na Detecao de Fraudes 🚀  

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema integrado de Machine Learning e RPA para deteccao de fraudes financeiras em tempo real com notificacao automatica.

## 📌 Objetivo  
Desenvolver uma solucao automatizada que:
1. Analisa transacoes financeiras usando modelos de ML
2. Detecta padroes suspeitos com alta precisao
3. Executa acoes corretivas via RPA
4. Envia alertas automaticos por e-mail

## 🛠 Tecnologias Utilizadas  
| Categoria          | Ferramentas                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| Linguagem          | Python 3.9+                                                                 |
| ML                 | Scikit-learn, XGBoost, Random Forest, SMOTE                                 |
| Analise de Dados   | Pandas, NumPy                                                               |
| Visualizacao       | Matplotlib, Seaborn, Plotly                                                 |
| Automacao          | win32com (Outlook), SMTP                                                    |
| Estatistica        | Statsmodels (McNemar), SciPy                                                |

## 📂 Estrutura do Projeto  

```bash
📂 automacao-fraudes
 ┣ 📂 data
 ┃ ┣ 📂 raw                   # Dados brutos (nao versionado)
 ┃ ┗ 📂 processed             # Dados pre-processados
 ┣ 📂 experiments
 ┃ ┣ 📜 train.py              # Treinamento (Credit Card)
 ┃ ┣ 📜 train_ieee.py         # Treinamento (IEEE)
 ┃ ┗ 📜 evaluate.py           # Avaliacao de modelos
 ┣ 📂 results
 ┃ ┣ 📂 logs                  # Logs de execucao
 ┃ ┣ 📂 plots                 # Graficos (ROC, matriz confusao)
 ┃ ┗ 📂 saved_models          # Modelos treinados (.pkl)
 ┣ 📜 main.py                 # Pipeline principal
 ┣ 📜 preprocessing.py        # Pre-processamento
 ┣ 📜 eda.py                  # Analise exploratoria
 ┣ 📜 modelo_ml.py            # Definicao dos modelos
 ┣ 📜 envioEmail.py           # Automacao de e-mails
 ┣ 📜 requirements.txt        # Dependencias
 ┗ 📜 README.md               # Este arquivo
```

## ⚙️ Configuracao e Execucao  
### Pre-requisitos  
- Python 3.9+
- Conta no Kaggle para download dos datasets

### Instalacao  
```bash
# Clone o repositorio
git clone https://github.com/seu-usuario/automacao-fraudes.git
cd automacao-fraudes

# Instale as dependencias
pip install -r requirements.txt
```

### Download dos Dados  
**IEEE CIS Fraud Dataset:**
- Disponivel em: Kaggle
- Arquivos necessarios: train_transaction.csv, test_transaction.csv
- Salvar em: `data/raw/ieee/`

**Credit Card Fraud Dataset:**
- Disponivel em: Kaggle
- Arquivo: `creditcard.csv`
- Salvar em: `data/raw/creditcard/`

### Execucao  
```bash
# Pipeline completo (pre-processamento + treino + avaliacao)
python main.py

# Treinar modelos especificos
python experiments/train.py --base creditcard --model xgb
python experiments/train_ieee.py --model rf
```

## 📊 Métricas e Estatísticas  
### Comparacao de Modelos  
| Modelo        | Acuracia | Precisao | Recall | F1-Score | AUC  |
|--------------|---------|---------|--------|---------|------|
| XGBoost     | 0.998   | 0.952   | 0.821  | 0.881   | 0.983|
| Random Forest | 0.996   | 0.912   | 0.784  | 0.843   | 0.962|

### Teste de McNemar  
```python
# Resultado:
- Estatistica χ²: 4.92
- p-valor: 0.027 (p < 0.05 → diferenca significativa)
```

### Graficos Gerados  
- **Matriz de Confusao**  
- **Curva ROC**  
- **Distribuicao de Features**  

## 📈 Resultados  
✅ Reducao de **15%** em falsos negativos vs abordagem manual  
✅ Economia de **40%** no tempo de analise via automacao RPA  
✅ Precisao de **95.2%** na deteccao de fraudes (XGBoost)  

## 📧 Notificacao Automatica  
### Exemplo de e-mail gerado:  
```
Assunto: [ALERTA] Fraude Detectada - Sistema AntiFraude

Corpo:
Foram identificadas 3 transacoes suspeitas:
- ID: 12345 | Valor: R$ 2.450,00
- ID: 67890 | Valor: R$ 3.780,50

Anexos:
- relatorio_fraudes.xlsx
- graficos_analise.zip
```

## 📚 Referencias  
- Chen, T. (2016). XGBoost: A Scalable Tree Boosting System  
- Chawla, N.V. (2002). SMOTE: Synthetic Minority Over-sampling Technique  
- Gartner (2021). Market Guide for RPA  

## 🔮 Proximos Passos  
✅ Integracao com APIs bancarias  
✅ Dashboard em tempo real  
✅ Modelos de Deep Learning  

## 🤝 Contribuicao  
Contribuicoes sao bem-vindas! Siga os passos:  
1. **Fork** o projeto  
2. Crie sua branch (`git checkout -b feature/nova-feature`)  
3. Commit suas mudancas (`git commit -m 'Adiciona nova feature'`)  
4. Push para a branch (`git push origin feature/nova-feature`)  
5. Abra um **Pull Request**  

## 📄 Licenca  
Distribuido sob licenca **MIT**. Veja `LICENSE` para mais informacoes.  

---  

Desenvolvido com ❤️ por [Guilherme de Almeida Pereira](https://www.linkedin.com/in/guilhermedealmeidapereira/)  


