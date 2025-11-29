# 🎬 Análise de Sentimentos em Avaliações de Filmes - Documentação Completa

## Sumário
1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Importações e Dependências](#importações-e-dependências)
3. [Arquitetura do Pipeline](#arquitetura-do-pipeline)
4. [Detalhamento das Funções](#detalhamento-das-funções)
5. [Modos de Execução](#modos-de-execução)
6. [Arquivos de Saída](#arquivos-de-saída)
7. [Desempenho dos Modelos](#desempenho-dos-modelos)
8. [Como Usar](#como-usar)

---

## Visão Geral do Projeto

Este projeto implementa um **Sistema de Análise de Sentimentos** para avaliações de filmes. Ele treina classificadores de machine learning para prever se uma avaliação de filme expressa um sentimento **POSITIVO** ou **NEGATIVO** com base no texto da avaliação.

### Principais Funcionalidades:
- Carrega e analisa 1,44 milhão de avaliações de filmes (amostra de 100.000 para processamento mais rápido)
- Treina e compara 4 modelos diferentes de machine learning
- Gera visualizações para análise exploratória de dados
- Fornece análise de importância de features (quais palavras indicam sentimento positivo/negativo)
- Inclui uma interface web interativa para previsões em tempo real
- Salva o melhor modelo para uso futuro

### Informações do Dataset:
- **Fonte:** Avaliações de críticos de filmes de diversas publicações
- **Tamanho:** 1,44 milhão de avaliações
- **Variável Alvo:** `scoreSentiment` (POSITIVE ou NEGATIVE)
- **Feature Principal:** `reviewText` (o texto completo da avaliação)

---

## Importações e Dependências

### Manipulação e Análise de Dados

```python
import pandas as pd
```
- **Propósito:** Biblioteca de manipulação e análise de dados
- **Usado para:** Carregar dados CSV, manipular DataFrames, limpeza de dados, filtragem e transformação
- **Operações principais:** `read_csv()`, `dropna()`, `value_counts()`, `groupby()`

```python
import numpy as np
```
- **Propósito:** Biblioteca de computação numérica
- **Usado para:** Operações com arrays, funções matemáticas, ordenação de índices
- **Operações principais:** `np.arange()`, `np.argsort()`, `np.linspace()`

### Visualização

```python
import matplotlib.pyplot as plt
```
- **Propósito:** Biblioteca principal de plotagem
- **Usado para:** Criar todos os gráficos e visualizações
- **Operações principais:** `subplots()`, `bar()`, `barh()`, `scatter()`, `savefig()`

```python
import seaborn as sns
```
- **Propósito:** Visualização estatística de dados (construída sobre matplotlib)
- **Usado para:** Criar mapas de calor para matrizes de confusão
- **Operações principais:** `heatmap()` com anotações

### Machine Learning - Seleção e Avaliação de Modelos

```python
from sklearn.model_selection import train_test_split
```
- **Propósito:** Dividir dados em conjuntos de treino e teste
- **Usado para:** Criar divisão 80/20 treino-teste com estratificação
- **Parâmetros:** `test_size=0.2`, `random_state=42`, `stratify=y`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
- **Propósito:** Converter texto em features numéricas usando TF-IDF
- **Como funciona:**
  - **TF (Frequência do Termo):** Quantas vezes uma palavra aparece em um documento
  - **IDF (Frequência Inversa do Documento):** Quão rara/importante uma palavra é em todos os documentos
  - **TF-IDF = TF × IDF:** Equilibra frequência com importância
- **Parâmetros utilizados:**
  - `max_features=10000`: Mantém as 10.000 palavras mais importantes
  - `ngram_range=(1, 2)`: Usa palavras únicas (unigramas) e pares de palavras (bigramas)
  - `min_df=5`: Ignora palavras que aparecem em menos de 5 documentos
  - `max_df=0.95`: Ignora palavras que aparecem em mais de 95% dos documentos
  - `stop_words='english'`: Remove palavras comuns em inglês (the, is, at, etc.)

### Machine Learning - Classificadores

```python
from sklearn.linear_model import LogisticRegression
```
- **Propósito:** Classificador linear para classificação binária/multiclasse
- **Como funciona:** Encontra uma fronteira de decisão linear usando função logística
- **Pontos fortes:** Rápido, interpretável, funciona bem com dados esparsos de alta dimensão (como texto)
- **Parâmetros:** `max_iter=1000`, `random_state=42`, `n_jobs=-1`

```python
from sklearn.naive_bayes import MultinomialNB
```
- **Propósito:** Classificador probabilístico baseado no teorema de Bayes
- **Como funciona:** Assume que as features são condicionalmente independentes dada a classe
- **Pontos fortes:** Muito rápido, funciona bem com dados de texto, bom baseline
- **Parâmetros:** `alpha=0.1` (suavização de Laplace)

```python
from sklearn.svm import LinearSVC
```
- **Propósito:** Máquina de Vetores de Suporte com kernel linear
- **Como funciona:** Encontra o hiperplano que maximiza a margem entre as classes
- **Pontos fortes:** Eficaz em espaços de alta dimensão, eficiente em memória
- **Parâmetros:** `random_state=42`, `max_iter=2000`

```python
from sklearn.ensemble import RandomForestClassifier
```
- **Propósito:** Conjunto de árvores de decisão
- **Como funciona:** Treina múltiplas árvores de decisão e agrega suas previsões
- **Pontos fortes:** Lida com relações não-lineares, resistente a overfitting
- **Parâmetros:** `n_estimators=100`, `random_state=42`, `n_jobs=-1`

### Machine Learning - Métricas

```python
from sklearn.metrics import (
    accuracy_score,      # Correção geral: (VP + VN) / Total
    precision_score,     # Dos positivos previstos, quantos estão corretos: VP / (VP + FP)
    recall_score,        # Dos positivos reais, quantos foram encontrados: VP / (VP + FN)
    f1_score,           # Média harmônica de precisão e recall
    classification_report,  # Relatório detalhado com todas as métricas por classe
    confusion_matrix,    # Matriz mostrando VP, VN, FP, FN
    roc_curve,          # Dados da curva ROC (Receiver Operating Characteristic)
    auc                 # Área Sob a Curva ROC
)
```

**Legenda:**
- VP = Verdadeiro Positivo
- VN = Verdadeiro Negativo
- FP = Falso Positivo
- FN = Falso Negativo

### Machine Learning - Pipeline

```python
from sklearn.pipeline import Pipeline
```
- **Propósito:** Encadear múltiplas etapas de processamento
- **Usado para:** Combinar vetorização TF-IDF com classificadores
- **Benefício:** Garante pré-processamento consistente durante treino e previsão

### Bibliotecas Utilitárias

```python
import warnings
warnings.filterwarnings('ignore')
```
- **Propósito:** Suprimir mensagens de aviso para saída mais limpa

```python
import re
```
- **Propósito:** Expressões regulares para correspondência de padrões de texto
- **Usado para:** Remover URLs, tags HTML, pontuação do texto

```python
import string
```
- **Propósito:** Constantes e utilitários de strings
- **Usado para:** Acesso a caracteres de pontuação (importado mas não usado diretamente)

```python
import joblib
```
- **Propósito:** Serialização eficiente de objetos Python
- **Usado para:** Salvar e carregar modelos treinados de/para o disco

```python
import argparse
```
- **Propósito:** Análise de argumentos de linha de comando
- **Usado para:** Implementar diferentes modos de execução (train, ui, demo)

---

## Arquitetura do Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PIPELINE DE ANÁLISE DE SENTIMENTOS                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 1: CARREGAMENTO DE DADOS                                               │
│ ──────────────────────────────                                              │
│ • Carregar dataset CSV (1,44M avaliações)                                   │
│ • Opcional: Amostrar 100.000 avaliações para processamento mais rápido      │
│ • Saída: DataFrame bruto                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 2: ANÁLISE EXPLORATÓRIA DE DADOS                                      │
│ ──────────────────────────────────────                                      │
│ • Exibir forma e colunas do dataset                                         │
│ • Verificar valores ausentes                                                │
│ • Mostrar distribuição de sentimentos (POSITIVE vs NEGATIVE)                │
│ • Calcular estatísticas do texto (comprimento, contagem de palavras)        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 3: VISUALIZAÇÃO DE DADOS                                               │
│ ──────────────────────────────                                              │
│ • Gráfico de barras da distribuição de sentimentos                          │
│ • Box Plot do comprimento das avaliações por sentimento                     │
│ • Histograma da contagem de palavras por sentimento                         │
│ • Sentimento por tipo de crítico (Top Critic vs Regular)                    │
│ • Saída: data/sentiment_analysis_eda.png                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 4: PRÉ-PROCESSAMENTO DE TEXTO                                          │
│ ────────────────────────────────────                                        │
│ • Remover linhas com reviewText ou scoreSentiment ausentes                  │
│ • Manter apenas sentimentos POSITIVE e NEGATIVE                             │
│ • Limpar texto:                                                             │
│   - Converter para minúsculas                                               │
│   - Remover URLs (http://, https://, www.)                                  │
│   - Remover tags HTML (<...>)                                               │
│   - Remover pontuação (exceto ! e ?)                                        │
│   - Remover espaços em branco extras                                        │
│ • Criar labels binários (1 = POSITIVE, 0 = NEGATIVE)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 5: DIVISÃO TREINO-TESTE                                                │
│ ─────────────────────────────                                               │
│ • Dividir dados: 80% treino, 20% teste                                      │
│ • Divisão estratificada (mantém proporções das classes)                     │
│ • Random state: 42 (para reprodutibilidade)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 6: TREINAMENTO DE MODELOS                                              │
│ ───────────────────────────────                                             │
│ Para cada modelo (Regressão Logística, Naive Bayes, SVM Linear, Random Forest):│
│ • Criar Pipeline: TfidfVectorizer → Classificador                           │
│ • Treinar com dados de treino                                               │
│ • Prever dados de teste                                                     │
│ • Calcular métricas (Acurácia, Precisão, Recall, F1)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 7: VISUALIZAÇÃO DE RESULTADOS                                          │
│ ────────────────────────────────────                                        │
│ • Comparação de desempenho dos modelos (gráfico de barras agrupadas)        │
│ • Matriz de confusão do melhor modelo (mapa de calor)                       │
│ • Comparação de F1 Score (gráfico de barras horizontal)                     │
│ • Visão geral da acurácia dos modelos (gráfico de dispersão)                │
│ • Saída: data/model_comparison.png                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 8: IMPORTÂNCIA DAS FEATURES                                            │
│ ─────────────────────────────────                                           │
│ • Extrair nomes das features TF-IDF e coeficientes da Regressão Logística   │
│ • Identificar top 20 palavras para sentimento POSITIVO (maiores coef.)      │
│ • Identificar top 20 palavras para sentimento NEGATIVO (menores coef.)      │
│ • Criar visualização                                                        │
│ • Saída: data/feature_importance.png                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 9: SALVAR MELHOR MODELO                                                │
│ ─────────────────────────────                                               │
│ • Identificar melhor modelo pelo F1 score                                   │
│ • Serializar e salvar em data/best_model.joblib                             │
│ • Modelo pode ser carregado posteriormente para previsões                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detalhamento das Funções

### 1. Funções de Carregamento de Dados

#### `load_data(filepath: str, sample_size: int = None) -> pd.DataFrame`

**Propósito:** Carregar o dataset de avaliações de filmes de um arquivo CSV.

**Parâmetros:**
- `filepath`: Caminho para o arquivo CSV
- `sample_size`: Número de avaliações para amostrar (None para dataset completo)

**Processo:**
1. Ler arquivo CSV usando pandas
2. Se sample_size for especificado, amostrar aleatoriamente esse número de linhas
3. Imprimir status do carregamento

**Retorna:** DataFrame com avaliações de filmes

---

### 2. Funções de Análise Exploratória de Dados

#### `explore_data(df: pd.DataFrame) -> None`

**Propósito:** Realizar e exibir análise exploratória de dados.

**Saídas no console:**
- Forma do dataset (linhas × colunas)
- Nomes das colunas
- Contagem de valores ausentes para reviewText e scoreSentiment
- Distribuição de sentimentos (contagens e porcentagens)
- Estatísticas do texto:
  - Comprimento médio em caracteres
  - Contagem média de palavras
  - Contagens mínima/máxima de palavras

#### `visualize_data(df: pd.DataFrame) -> None`

**Propósito:** Criar uma grade 2×2 de visualizações.

**Visualizações criadas:**

| Posição | Tipo de Gráfico | Descrição |
|---------|-----------------|-----------|
| Superior-Esquerda | Gráfico de Barras | Distribuição de sentimentos com porcentagens |
| Superior-Direita | Box Plot | Comprimento das avaliações por sentimento |
| Inferior-Esquerda | Histograma | Distribuição de contagem de palavras por sentimento |
| Inferior-Direita | Barras Agrupadas | Sentimento por tipo de crítico (Top vs Regular) |

**Arquivo de saída:** `data/sentiment_analysis_eda.png`

**Esquema de cores:**
- 🟢 Verde (`#2ecc71`): Sentimento POSITIVO
- 🔴 Vermelho (`#e74c3c`): Sentimento NEGATIVO

---

### 3. Funções de Pré-processamento de Texto

#### `clean_text(text: str) -> str`

**Propósito:** Limpar e normalizar texto de avaliação para machine learning.

**Etapas de limpeza:**
1. Tratar valores NaN → retornar string vazia
2. Converter para tipo string
3. Converter para minúsculas
4. Remover URLs (padrões http, https, www)
5. Remover tags HTML
6. Remover pontuação (exceto ! e ? que carregam sentimento)
7. Remover espaços em branco extras

**Exemplo:**
```
Entrada: "This movie was AMAZING!!! Check it out at http://example.com <br>"
Saída:   "this movie was amazing!"
```

#### `preprocess_data(df: pd.DataFrame) -> pd.DataFrame`

**Propósito:** Preparar todo o dataset para treinamento do modelo.

**Etapas:**
1. Remover linhas com reviewText ou scoreSentiment ausentes
2. Filtrar apenas sentimentos POSITIVE e NEGATIVE
3. Aplicar `clean_text()` em todas as avaliações
4. Remover avaliações que ficam vazias após limpeza
5. Criar labels binários: POSITIVE=1, NEGATIVE=0

**Retorna:** DataFrame limpo com novas colunas:
- `cleaned_text`: Texto da avaliação pré-processado
- `sentiment_label`: Label binário (0 ou 1)

---

### 4. Funções de Treinamento de Modelos

#### `create_models() -> dict`

**Propósito:** Criar dicionário de pipelines de modelos para treinar.

**Modelos criados:**

| Modelo | Features TF-IDF | Parâmetros do Classificador |
|--------|-----------------|----------------------------|
| Regressão Logística | 10.000 | max_iter=1000, n_jobs=-1 |
| Naive Bayes | 10.000 | alpha=0.1 |
| SVM Linear | 10.000 | max_iter=2000 |
| Random Forest | 5.000 | n_estimators=100, n_jobs=-1 |

**Configuração TF-IDF (compartilhada):**
- `ngram_range=(1, 2)`: Unigramas e bigramas
- `min_df=5`: Frequência mínima do documento
- `max_df=0.95`: Frequência máxima do documento
- `stop_words='english'`: Remove stop words em inglês

#### `train_and_evaluate(...) -> dict`

**Propósito:** Treinar todos os modelos e avaliar seu desempenho.

**Processo para cada modelo:**
1. Ajustar pipeline nos dados de treino (X_train, y_train)
2. Prever nos dados de teste (X_test)
3. Calcular métricas:
   - **Acurácia:** Correção geral
   - **Precisão:** Valor preditivo positivo
   - **Recall:** Taxa de verdadeiro positivo (sensibilidade)
   - **F1-Score:** Média harmônica de precisão e recall

**Retorna:** Dicionário contendo:
```python
{
    'Nome do Modelo': {
        'model': pipeline_treinado,
        'accuracy': float,
        'precision': float,
        'recall': float,
        'f1': float,
        'y_pred': array
    }
}
```

---

### 5. Funções de Visualização

#### `plot_results(results: dict, y_test: pd.Series) -> None`

**Propósito:** Criar visualizações de comparação de modelos.

**Visualizações:**

| Posição | Gráfico | Descrição |
|---------|---------|-----------|
| Superior-Esquerda | Barras Agrupadas | Todas as métricas para todos os modelos |
| Superior-Direita | Mapa de Calor | Matriz de confusão do melhor modelo |
| Inferior-Esquerda | Barras Horizontais | F1 scores ranqueados |
| Inferior-Direita | Dispersão | Visão geral da acurácia com gradiente de cor |

**Arquivo de saída:** `data/model_comparison.png`

#### `print_classification_reports(results: dict, y_test: pd.Series) -> None`

**Propósito:** Imprimir relatórios detalhados de classificação do sklearn.

**Saída por modelo:**
```
              precision    recall  f1-score   support

    NEGATIVE       0.75      0.55      0.63      6297
    POSITIVE       0.80      0.91      0.85     12742

    accuracy                           0.79     19039
   macro avg       0.78      0.73      0.74     19039
weighted avg       0.79      0.79      0.78     19039
```

---

### 6. Funções de Importância de Features

#### `show_feature_importance(results: dict, top_n: int = 20) -> None`

**Propósito:** Identificar e visualizar as palavras mais importantes para sentimento.

**Como funciona:**
1. Extrair vocabulário TF-IDF (nomes das features)
2. Extrair coeficientes da Regressão Logística
3. Ordenar coeficientes para encontrar:
   - **Maiores coeficientes** → Palavras indicadoras de POSITIVO
   - **Menores coeficientes** → Palavras indicadoras de NEGATIVO

**Saída no console:**
```
Top 20 palavras indicando sentimento POSITIVO:
----------------------------------------
  entertaining         (coef: 4.2430)
  enjoyable            (coef: 4.0315)
  ...

Top 20 palavras indicando sentimento NEGATIVO:
----------------------------------------
  fails                (coef: -5.9928)
  unfortunately        (coef: -5.4558)
  ...
```

**Arquivo de saída:** `data/feature_importance.png`

---

### 7. Funções de Previsão

#### `predict_sentiment(model, text: str) -> dict`

**Propósito:** Prever sentimento para uma única avaliação.

**Processo:**
1. Limpar o texto de entrada
2. Usar modelo para prever classe
3. Obter scores de probabilidade (se disponível)
4. Formatar resultado

**Retorna:**
```python
{
    'text': 'texto da avaliação truncado...',
    'sentiment': 'POSITIVE' ou 'NEGATIVE',
    'confidence': 0.95  # ou None para SVM
}
```

#### `interactive_demo(model) -> None`

**Propósito:** Executar demo de linha de comando com previsões de exemplo.

**Avaliações de exemplo testadas:**
1. "This movie was absolutely fantastic!..." → Esperado: POSITIVE
2. "Terrible film. Waste of time..." → Esperado: NEGATIVE
3. "A decent movie with some good moments..." → Pode ser qualquer um
4. "One of the best films I've ever seen..." → Esperado: POSITIVE
5. "I couldn't even finish watching this garbage..." → Esperado: NEGATIVE

---

### 8. Funções de Persistência de Modelos

#### `save_best_model(results: dict, filepath: str) -> None`

**Propósito:** Salvar o modelo com melhor desempenho no disco.

**Critério de seleção:** Maior F1 score

**Arquivo de saída:** `data/best_model.joblib`

**Uso para carregar:**
```python
model = joblib.load('data/best_model.joblib')
prediction = model.predict(["Texto da sua avaliação aqui"])
```

---

### 9. Função de Interface Interativa

#### `run_interactive_ui() -> None`

**Propósito:** Lançar uma interface web para previsão de sentimentos.

**Tecnologia:** Gradio (biblioteca Python de interface web)

**Funcionalidades:**
- Área de entrada de texto para inserir avaliações
- Botão "Analisar Sentimento"
- Exibição de resultados com:
  - Previsão (POSITIVO ✅ ou NEGATIVO ❌)
  - Porcentagem de confiança
  - Barra de probabilidade
- Avaliações de exemplo para testar
- Painel de informações do modelo

**URL:** `http://127.0.0.1:7860`

---

## Modos de Execução

A aplicação suporta 3 modos de execução via argumentos de linha de comando:

### Modo 1: Treino (`--mode train`)

```bash
python app.py --mode train
```

**O que faz:**
1. Carrega e explora o dataset
2. Cria visualizações (EDA)
3. Pré-processa dados de texto
4. Treina 4 modelos diferentes
5. Avalia e compara modelos
6. Mostra importância das features
7. Executa previsões demo
8. Salva o melhor modelo

**Arquivos de saída gerados:**
- `data/sentiment_analysis_eda.png`
- `data/model_comparison.png`
- `data/feature_importance.png`
- `data/best_model.joblib`

**Modo padrão** (executa se nenhum --mode for especificado)

---

### Modo 2: Interface (`--mode ui`)

```bash
python app.py --mode ui
```

**O que faz:**
1. Carrega o melhor modelo salvo
2. Lança interface web Gradio
3. Abre navegador automaticamente

**Requisitos:**
- Deve executar `--mode train` primeiro para criar o modelo
- Pacote Gradio deve estar instalado

**Acesso:** Abrir `http://127.0.0.1:7860` no navegador

---

### Modo 3: Demo (`--mode demo`)

```bash
python app.py --mode demo
```

**O que faz:**
1. Carrega o melhor modelo salvo
2. Executa demo interativo de linha de comando
3. Mostra previsões para avaliações de exemplo

**Requisitos:**
- Deve executar `--mode train` primeiro para criar o modelo

---

## Arquivos de Saída

### 1. `data/sentiment_analysis_eda.png`

**Tipo:** Visualização de Análise Exploratória de Dados (grade 2×2)

**Conteúdo:**
- Distribuição de sentimentos com porcentagens
- Distribuição do comprimento das avaliações por sentimento
- Distribuição da contagem de palavras por sentimento
- Análise de sentimentos por tipo de crítico

**Tamanho:** ~300 DPI, alta qualidade para relatórios

---

### 2. `data/model_comparison.png`

**Tipo:** Visualização de avaliação de modelos (grade 2×2)

**Conteúdo:**
- Comparação de todas as métricas (acurácia, precisão, recall, F1)
- Matriz de confusão do melhor modelo
- Ranking de F1 score
- Visão geral da acurácia

---

### 3. `data/feature_importance.png`

**Tipo:** Visualização de importância de features (layout 1×2)

**Conteúdo:**
- Top 10 palavras indicando sentimento POSITIVO
- Top 10 palavras indicando sentimento NEGATIVO

---

### 4. `data/best_model.joblib`

**Tipo:** Pipeline scikit-learn serializado

**Conteúdo:**
- Pipeline treinado completo incluindo:
  - TfidfVectorizer (com vocabulário ajustado)
  - Classificador (com pesos treinados)

**Tamanho:** Varia (~10-50 MB dependendo do tamanho do vocabulário)

**Carregamento:**
```python
import joblib
model = joblib.load('data/best_model.joblib')
result = model.predict(["Ótimo filme!"])  # Retorna [1] para POSITIVE
```

---

## Desempenho dos Modelos

### Resultados Típicos (em amostra de 100.000):

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| **Regressão Logística** | 79,06% | 80,27% | 91,11% | **85,35%** |
| Naive Bayes | 77,19% | 77,17% | 93,63% | 84,60% |
| SVM Linear | 78,30% | 81,42% | 87,55% | 84,38% |
| Random Forest | 75,36% | 77,71% | 88,60% | 82,80% |

### Melhor Modelo: Regressão Logística

**Por que a Regressão Logística tem melhor desempenho:**
1. Funciona bem com dados esparsos de alta dimensão (vetores TF-IDF)
2. Fronteira de decisão linear é apropriada para classificação de texto
3. Treinamento e previsão rápidos
4. Altamente interpretável (coeficientes mostram importância das features)

### Observações Principais:
- Todos os modelos favorecem previsões POSITIVAS (maior recall para positivo)
- Isso é devido ao desbalanceamento de classes (~67% positivo, ~33% negativo)
- Regressão Logística tem o melhor equilíbrio entre precisão e recall

---

## Como Usar

### Instalação

```bash
# Instalar pacotes necessários
pip install pandas numpy scikit-learn matplotlib seaborn joblib gradio
```

### Treinando o Modelo

```bash
# Navegar para o diretório do projeto
cd movie-recomendation-system-ml

# Executar treinamento (modo padrão)
python app.py
# ou explicitamente
python app.py --mode train
```

### Usando a Interface Web

```bash
# Primeiro, garanta que o modelo está treinado
python app.py --mode train

# Depois lance a UI
python app.py --mode ui
```

### Fazendo Previsões em Código

```python
import joblib
from app import clean_text

# Carregar modelo
model = joblib.load('data/best_model.joblib')

# Prever
review = "This movie was absolutely wonderful!"
cleaned = clean_text(review)
prediction = model.predict([cleaned])[0]
probability = model.predict_proba([cleaned])[0]

print(f"Sentimento: {'POSITIVO' if prediction == 1 else 'NEGATIVO'}")
print(f"Confiança: {max(probability):.1%}")
```

### Opções de Configuração

Na função `main()`, você pode modificar:

```python
DATA_PATH = 'data/dataset.csv'  # Caminho para o dataset
SAMPLE_SIZE = 100000            # Defina como None para dataset completo
TEST_SIZE = 0.2                 # Proporção da divisão treino/teste
RANDOM_STATE = 42               # Para reprodutibilidade
```

---

## Resumo

Este projeto demonstra um pipeline completo de machine learning para análise de sentimentos:

1. **Engenharia de Dados:** Carregamento, limpeza e pré-processamento de dados de texto
2. **Engenharia de Features:** Vetorização TF-IDF com unigramas e bigramas
3. **Treinamento de Modelos:** Comparação de 4 algoritmos diferentes
4. **Avaliação:** Métricas abrangentes e visualizações
5. **Interpretação:** Análise de importância das features
6. **Deploy:** Interface web para previsões em tempo real
7. **Persistência:** Serialização de modelos para uso em produção

O modelo com melhor desempenho (Regressão Logística) alcança **85% de F1-Score**, tornando-o adequado para tarefas práticas de classificação de sentimentos.
