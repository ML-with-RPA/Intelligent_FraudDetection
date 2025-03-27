# Dados Processados (Processed Data)  

Este diretório contém os dados processados gerados pelo pipeline de pré-processamento.  

### **Como Gerar os Arquivos**:  
1. **Pré-requisitos**:  
   - Certifique-se de que os dados brutos estão em `data/raw/`.  
   - Execute o script de pré-processamento:  
     ```bash  
     python main.py  
     ```  

2. **Arquivos Gerados**:  
   - `dados_treinamento_completo.csv`: Dados de treino normalizados.  
   - `dados_teste_completo.csv`: Dados de teste normalizados.  
   - `X_train.csv` / `Y_train.csv`: Features e target de treino (Credit Card).  
   - `X_test.csv` / `Y_test.csv`: Features e target de teste (Credit Card).  

### **Nota**:  
- Esses arquivos são gerados automaticamente e não devem ser commitados no GitHub.  