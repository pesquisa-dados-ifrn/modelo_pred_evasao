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
   - **Métricas de avaliação**: Acurácia, relatório de classificação, matriz de confusão, e AUC-ROC.
6. **Pickle**: Para salvar o modelo treinado e os valores únicos da variável alvo, permitindo reutilização futura do modelo.
7. **Resampling com Scikit-learn**: Aumenta a classe minoritária usando técnicas de upsampling.

## Estrutura do Projeto

1. **Pré-processamento de Dados**:
   - Leitura do dataset.
   - Identificação e remoção de colunas com mais de 17% de valores ausentes.
   - Tratamento de variáveis categóricas e numéricas.
   - Balanceamento do conjunto de dados usando a técnica de upsampling.

2. **Divisão em Conjuntos de Treino e Teste**:
   - Divisão estratificada para garantir a representatividade da classe alvo.

3. **Criação do Modelo**:
   - Modelo de classificação utilizando o RandomForestClassifier.
   - Treinamento do modelo nos dados balanceados.
   - Avaliação com métricas como acurácia e matriz de confusão.

4. **Salvamento do Modelo**:
   - O modelo treinado é salvo em formato pickle para futuras predições.

## Futuro

Este código será integrado em uma aplicação Streamlit para permitir que usuários façam novas predições, com uma interface interativa e simples de usar.

## Como Executar

1. Clone o repositório.
2. Instale as dependências necessárias (consulte `requirements.txt` se houver).
3. Execute o script principal para treinar o modelo.
4. O modelo será salvo como `modelo_pred.pickle`, e os valores únicos da variável alvo como `output_class.pickle`.

---

Este README oferece uma visão geral clara e informativa para quem deseja entender e rodar o projeto.