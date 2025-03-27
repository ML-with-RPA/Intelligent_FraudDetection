# 🏦 Automacao Inteligente na Detecção de Fraudes 🚀  

Este projeto integra **Machine Learning (ML)** e **Robotic Process Automation (RPA)** para identificar e agir contra fraudes financeiras em tempo real.  

## 📌 Objetivo  
Desenvolver uma solução automatizada que analisa transações financeiras, detecta padrões suspeitos e executa ações corretivas utilizando **RPA**, incluindo envio de notificações por e-mail quando fraudes são detectadas.  

## 🛠 Tecnologias Utilizadas  
- **Python 3.9+**  
- **Pandas** → Manipulação e análise de dados  
- **Scikit-learn** → Modelos de Machine Learning  
- **XGBoost/Random Forest** → Algoritmos de classificação  
- **SMOTE** → Balanceamento de classes  
- **Seaborn/Matplotlib** → Visualização de dados  
- **OpenPyXL** → Manipulação de arquivos Excel  
- **SMTP (E-mail Automation)** → Envio de alertas sobre possíveis fraudes  

## 📂 Estrutura do Projeto  
```bash
📂 automacao-fraudes
 ┣ 📂 data                    # Bases de dados processadas e brutas
 ┣ 📂 experiments             # Scripts para experimentos com ML
 ┃ ┣ 📜 train.py              # Treinamento de modelos (Credit Card)
 ┃ ┣ 📜 train_ieee.py         # Treinamento de modelos (IEEE)
 ┃ ┗ 📜 evaluate.py           # Avaliação de modelos
 ┣ 📂 resultado               # Relatórios e logs das execuções
 ┣ 📜 main.py                 # Código principal para processamento de dados e automação RPA
 ┣ 📜 modelo_ml.py            # Implementação dos modelos de Machine Learning
 ┣ 📜 notificacao_email.py     # Envio automático de alertas sobre fraudes detectadas
 ┣ 📜 requirements.txt         # Lista de dependências do projeto
 ┗ 📜 README.md               # Documentação do projeto
```

## ⚙️ Como Executar  

1. **Instalar as dependências**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Treinar um modelo**  
   - Com base *Credit Card*  
     ```bash
     python experiments/train.py --base creditcard --model xgb  # XGBoost  
     python experiments/train.py --base creditcard --model rf   # Random Forest  
     ```
   - Com base *IEEE*  
     ```bash
     python experiments/train_ieee.py --model xgb  # XGBoost  
     python experiments/train_ieee.py --model rf   # Random Forest  
     ```

3. **Executar a análise e detecção de fraudes**  
   ```bash
   python main.py
   ```

4. **Receber notificações automáticas de fraudes detectadas**  
   O sistema enviará um e-mail com os detalhes de qualquer fraude identificada no banco de dados processado.  

## 📊 Principais Recursos  
✔ Treinamento de modelos de ML com bases de dados *Credit Card* e *IEEE*  
✔ Balanceamento de classes com SMOTE para evitar viés nos modelos  
✔ Comparação entre os modelos *XGBoost* e *Random Forest*  
✔ Automação de alertas via e-mail sempre que uma fraude for detectada  
✔ Relatórios de classificação e matrizes de confusão para análise de desempenho  

## 🔍 Próximas Melhorias  
🔹 Otimização dos hiperparâmetros dos modelos para melhor desempenho  
🔹 Implementação de uma API para facilitar a integração com outras aplicações  
🔹 Adição de um dashboard interativo para visualização de fraudes em tempo real  

