# Predição de Evasão de Alunos - IFRN

Este projeto visa construir um modelo de machine learning para prever a possibilidade de evasão de alunos do Instituto Federal do Rio Grande do Norte (IFRN), utilizando um conjunto de dados que contém informações desses alunos. O modelo tem como objetivo identificar possíveis riscos de evasão e auxiliar na criação de estratégias de intervenção.

## Tecnologias Utilizadas

1. **Python**: Linguagem principal utilizada para manipulação de dados e construção do modelo.
2. **Pandas**: Usada para manipulação de dados, criação de DataFrames e processamento de valores ausentes.
3. **Missingno**: Ferramenta de visualização de dados ausentes.
4. **Matplotlib e Seaborn**: Bibliotecas de visualização de gráficos, usadas para criar gráficos como histogramas, matriz de confusão e curva ROC.
5. **Scikit-learn**: Biblioteca de machine learning utilizada para:
   - **Imputação de valores faltantes**.
   - **Codificação de variáveis categóricas** com OneHotEncoder.
   - **Divisão do dataset** em conjuntos de treino e teste.
   - **Algoritmos de classificação**: RandomForest, CatBoost, LGBM e XGBoost.
   - **Métricas de avaliação**: Acurácia, relatório de classificação, matriz de confusão.
6. **Imbalanced-learn (SMOTE)**: Técnica de balanceamento de classes para lidar com dados desbalanceados.
7. **ydata-profiling**: Geração de relatórios detalhados sobre os dados para visualização e análise exploratória.
8. **Pickle**: Para salvar o modelo treinado e os valores únicos da variável alvo, permitindo reutilização futura do modelo.

## Estrutura do Projeto

### 1. **Pré-processamento de Dados**:
   - Leitura do dataset `merge2018-tratado.csv`.
   - Identificação e remoção de colunas com mais de 17% de valores ausentes.
   - Tratamento de variáveis numéricas (imputação de valores faltantes com mediana).
   - Tratamento de variáveis categóricas (imputação com valor mais frequente e transformação para variáveis dummy com OneHotEncoder).
   - Balanceamento do conjunto de dados usando a técnica SMOTE.

### 2. **Tomada de Decisões no Pré-processamento**:

   Inicialmente, realizamos uma análise exploratória dos dados brutos, ainda não tratados, para entender melhor com o que estamos lidando. Por se tratar de um banco de dados extenso, foi necessário, em primeiro lugar, encontrar uma forma de reduzir a quantidade de dados a serem trabalhados. Já tendo discutido anteriormente sobre a estrutura da base, era perceptível a presença de uma grande quantidade de valores ausentes. Como ainda sou relativamente novo na área de análise de dados e não conhecia ferramentas mais sofisticadas para lidar com esse problema, decidi seguir uma abordagem mais simples: a exclusão de colunas com dados ausentes.

   No entanto, remover colunas arbitrariamente poderia comprometer a integridade da base de dados. Por isso, busquei um critério razoável para determinar o limite máximo de valores ausentes aceitável. Após análise, defini um valor de 17% — ou seja, colunas com 17% ou mais de valores ausentes (aproximadamente ⅙ da coluna) foram removidas. Essa decisão também facilita a seleção das features a serem utilizadas pelos modelos nos passos posteriores, reduzindo o esforço necessário para o tratamento dos dados.

   A próxima escolha fundamental foi a criação da variável alvo. Como o problema exige, devemos definir uma variável que identifique os alunos evadidos dos cursos do IFRN. Para isso, utilizamos a feature "Situação no Curso", que contém os seguintes valores:  
   'Evasão', 'Matriculado', 'Concluído', 'Cancelado', 'Formado', 'Jubilado', 'Trancado Voluntariamente', 'Trancado', 'Cancelamento por Desligamento', 'Transferido Interno', 'Transferido Externo', 'Cancelamento por Duplicidade', 'Cancelamento Compulsório', 'Matrícula Vínculo Institucional' e 'Intercâmbio'.

   Os valores "Cancelamento por Duplicidade", "Cancelamento Compulsório" e "Matrícula Vínculo Institucional" foram removidos do conjunto de dados antes da criação da variável alvo. Essas categorias representam situações administrativas específicas que não refletem diretamente o fenômeno da evasão estudantil. Como o objetivo do modelo é identificar padrões que levem à evasão voluntária ou circunstancial dos alunos, esses registros foram considerados irrelevantes para a análise e descartados na etapa de pré-processamento.

   Entre as categorias restantes, aquelas que foram consideradas como evasão foram:  
   'Evasão', 'Cancelado', 'Jubilado' e 'Cancelamento por Desligamento'.  
   As demais categorias podem estar associadas a condições adversas temporárias que levaram o aluno a se afastar dos estudos, mas sem necessariamente indicar uma evasão definitiva. Por exemplo, alunos classificados na categoria "Trancado Voluntariamente" ainda podem retomar suas atividades acadêmicas no futuro e concluir o curso. Dessa forma, nossa definição de evasão se restringe a estudantes que se afastaram e não retornaram às atividades escolares.

### 3. **Divisão em Conjuntos de Treino e Teste**:
   - Divisão estratificada para garantir a representatividade da classe alvo.

### 4. **Criação e Treinamento do Modelo**:
   - Modelos de classificação utilizando o `RandomForestClassifier`, `CatBoostClassifier`, `LGBMClassifier` e `XGBClassifier`.
   - Treinamento do modelo nos dados balanceados.
   - Avaliação com métricas como validação cruzada, acurácia, relatório de classificação e matriz de confusão.

### 5. **Otimizando o Modelo**:
   
   Antes de prosseguir com as previsões, buscamos otimizar os hiperparâmetros utilizando GridSearch em conjunto com KFold Cross-Validation. Para isso, realizamos diversos testes ao longo de quase um mês, variando tanto os parâmetros dos modelos quanto os valores do KFold. No entanto, os resultados obtidos não foram tão satisfatórios quanto o esperado. Durante os testes, observamos que os modelos sem o uso de GridSearch e KFold apresentavam desempenhos semelhantes, mas com um tempo de resposta significativamente menor. Como essas técnicas demandam alto custo computacional e tempo de execução, optamos por não utilizá-las nesta etapa, uma vez que a diferença de desempenho era mínima ou praticamente nula.

### 6. **Salvamento do Modelo**:
   - O modelo treinado é salvo em formato pickle como `modelo_treinado.pickle` para futuras predições.
   - Os valores únicos da variável alvo são salvos como `output_class.pickle`.

### 7. **Geração de Relatórios**:
   - Relatório de análise exploratória dos dados gerado com `ydata_profiling`.
   - Relatório gerado como `relatorio_dados_evasao_modificado.html`.

### 8. **Interface Streamlit**:
   - O código será integrado a uma aplicação Streamlit no futuro para permitir que usuários façam novas predições de evasão, com uma interface interativa e simples de usar.

## Como Executar

1. Clone o repositório.
2. Instale as dependências necessárias:
   ```
   pip install -r requirements.txt
   ```
3. Execute o notebook `modelo_pred.ipynb` para treinar o modelo.
4. O modelo será salvo como `modelo_treinado.pickle` e os valores únicos da variável alvo como `output_class.pickle`.

## Futuro

Este código será integrado em uma aplicação Streamlit para permitir que usuários façam novas predições de evasão de alunos, com uma interface interativa que possibilitará a visualização dos resultados em tempo real.
