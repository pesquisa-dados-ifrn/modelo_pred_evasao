# Predição de Evasão de Alunos - IFRN

Este projeto visa construir um modelo de machine learning para prever a possibilidade de evasão de alunos do Instituto Federal do Rio Grande do Norte (IFRN), utilizando um conjunto de dados que contém informações desses alunos. O modelo tem como objetivo identificar possíveis riscos de evasão e auxiliar na criação de estratégias de intervenção.

## Tecnologias Utilizadas

1. **Python**: Linguagem principal utilizada para manipulação de dados e construção do modelo.
2. **Pandas**: Usada para manipulação de dados, criação de DataFrames e processamento de valores ausentes.
3. **Missingno**: Ferramenta de visualização de dados ausentes.
4. **Matplotlib e Seaborn**: Bibliotecas de visualização de gráficos, usadas para criar gráficos como histogramas, matriz de confusão, e curva ROC.
5. **Scikit-learn**: Biblioteca de machine learning utilizada para:
   - **Imputação de valores faltantes**.
   - **Codificação de variáveis categóricas** com OneHotEncoder.
   - **Divisão do dataset** em conjuntos de treino e teste.
   - **Algoritmo RandomForest** para classificação.
   - **Métricas de avaliação**: Acurácia, relatório de classificação, matriz de confusão.
6. **Imbalanced-learn (SMOTE)**: Técnica de balanceamento de classes para lidar com dados desbalanceados.
7. **ydata-profiling**: Geração de relatórios detalhados sobre os dados para visualização e análise exploratória.
8. **Pickle**: Para salvar o modelo treinado e os valores únicos da variável alvo, permitindo reutilização futura do modelo.
9. **GridSearchCV (Scikit-learn)**: Técnica utilizada para otimização de hiperparâmetros do modelo de Random Forest.

## Estrutura do Projeto

1. **Pré-processamento de Dados**:
   - Leitura do dataset `merge2018-tratado.csv`.
   - Identificação e remoção de colunas com mais de 17% de valores ausentes.
   - Tratamento de variáveis numéricas (imputação de valores faltantes com mediana).
   - Tratamento de variáveis categóricas (imputação com valor mais frequente e transformação para variáveis dummy com OneHotEncoder).
   - Balanceamento do conjunto de dados usando a técnica SMOTE.

2. **Divisão em Conjuntos de Treino e Teste**:
   - Divisão estratificada para garantir a representatividade da classe alvo.

3. **Execução do Grid Search**:
   - Utilização de `GridSearchCV` para otimização dos hiperparâmetros do modelo Random Forest.

4. **Criação e Treinamento do Modelo**:
   - Modelo de classificação utilizando o `RandomForestClassifier`.
   - Treinamento do modelo nos dados balanceados.
   - Avaliação com métricas como acurácia, relatório de classificação, e matriz de confusão.

5. **Salvamento do Modelo**:
   - O modelo treinado é salvo em formato pickle como `modelo_treinado.pickle` para futuras predições.
   - Os valores únicos da variável alvo são salvos como `output_class.pickle`.

6. **Geração de Relatórios**:
   - Relatório de análise exploratória dos dados gerado com `ydata_profiling`.
   - Relatório gerado como `relatorio_dados_evasao_modificado.html`.

## Como Executar

1. Clone o repositório.
2. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o notebook `modelo_pred.ipynb` para treinar o modelo.
4. O modelo será salvo como `modelo_treinado.pickle`, e os valores únicos da variável alvo como `output_class.pickle`.

## Futuro

Este código será integrado em uma aplicação Streamlit para permitir que usuários façam novas predições, com uma interface interativa e simples de usar.
