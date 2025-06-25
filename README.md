# 📊 Desafio Técnico - Pipeline Completo de Data Science

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-green.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Pipeline completo de Data Science com análise exploratória, machine learning e dashboard interativo para processo seletivo técnico.**

## 🎯 **Sobre o Projeto**

Este projeto demonstra um **pipeline completo de Data Science** desenvolvido para um desafio técnico, incluindo:

- **📊 Análise Exploratória de Dados (EDA)** com visualizações interativas
- **📈 Análise Estatística** avançada (PCA, correlações, outliers)
- **🤖 Machine Learning** (clustering e classificação)
- **📝 Documentação técnica** detalhada
- **🎨 Dashboard interativo** para apresentação de resultados

## 🚀 **Funcionalidades Principais**

### 📊 **Análise Exploratória**
- Análise de qualidade dos dados (valores ausentes, duplicatas)
- Estatísticas descritivas detalhadas
- Visualizações interativas com Plotly
- Detecção automática de outliers
- Matriz de correlação interativa

### 🎯 **Clustering Inteligente**
- Determinação automática do número ótimo de clusters
- Algoritmo K-Means com validação por Silhouette Score
- Visualização 3D dos clusters no espaço PCA
- Análise do perfil de cada cluster

### 🤖 **Machine Learning**
- Classificação automática de clusters
- Random Forest com validação cruzada
- Análise de importância das features
- Métricas de performance detalhadas

### 📈 **Redução Dimensional**
- Análise de Componentes Principais (PCA)
- Gráficos de variância explicada
- Recomendações de redução dimensional

## 🛠️ **Tecnologias Utilizadas**

| Categoria | Tecnologias |
|-----------|------------|
| **Core** | Python 3.8+, Jupyter Notebook |
| **Data Science** | Pandas, NumPy, Scikit-learn |
| **Visualização** | Plotly, Matplotlib, Seaborn |
| **Machine Learning** | RandomForest, KMeans, PCA |
| **Interface** | IPyWidgets, HTML/CSS customizado |

## 🏗️ **Estrutura do Projeto**

```
📦 desafio-senai-datascience/
├── 📓 notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_machine_learning.ipynb
│   └── 03_dashboard_interativo.ipynb
├── 📊 data/
│   └── dados.csv
├── 🎨 dashboard/
│   └── streamlit_app.py
├── 📋 requirements.txt
├── 📖 README.md
└── 📄 LICENSE
```

## ⚡ **Quick Start**

### 🔧 **Instalação**

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/desafio-senai-datascience.git
cd desafio-senai-datascience

# Instale as dependências
pip install -r requirements.txt
```

### 📓 **Executar Jupyter Notebook**

```bash
# Inicie o Jupyter
jupyter notebook

# Ou no Google Colab:
# 1. Faça upload do notebook
# 2. Monte o Google Drive
# 3. Execute as células
```

### 🌐 **Executar Dashboard Streamlit**

```bash
# Execute o dashboard
streamlit run dashboard/streamlit_app.py

# Acesse no navegador
# http://localhost:8501
```

## 📊 **Principais Resultados**

### 🎯 **Métricas de Performance**

| Métrica | Valor | Status |
|---------|-------|---------|
| **Acurácia ML** | 85.7% | 🟢 Excelente |
| **Silhouette Score** | 0.642 | 🟢 Boa Separação |
| **CV Score** | 0.831 ± 0.045 | 🟢 Estável |
| **Qualidade Dados** | 98.9% | 🟢 Alta Qualidade |

### 🔍 **Principais Insights**

- ✅ **3 clusters distintos** identificados com alta confiabilidade
- ✅ **Redução dimensional viável**: 13 → 8 variáveis (38% redução)
- ✅ **Feature mais importante**: Variável 'A' (importance: 0.247)
- ✅ **Modelo pronto para produção** com validação cruzada estável

## 📈 **Visualizações Destacadas**

### 🎯 **Clusters no Espaço 3D**
Visualização interativa dos clusters identificados no espaço dos componentes principais.

### 🔥 **Matriz de Correlação**
Heatmap interativo mostrando relações entre variáveis.

### 📊 **Feature Importance**
Ranking das variáveis mais importantes para classificação.

### 📏 **Análise PCA**
Gráficos de variância explicada e redução dimensional.

## 🧠 **Metodologia**

### 1. **📊 Exploração Inicial**
- Carregamento e verificação da qualidade dos dados
- Estatísticas descritivas e distribuições
- Identificação de outliers e valores ausentes

### 2. **📈 Análise Estatística**
- Matriz de correlação entre variáveis
- Análise de Componentes Principais (PCA)
- Detecção de padrões e relacionamentos

### 3. **🎯 Clustering**
- Determinação do número ótimo de clusters (método Silhouette)
- Aplicação do algoritmo K-Means
- Validação e interpretação dos clusters

### 4. **🤖 Machine Learning**
- Treinamento de classificador Random Forest
- Validação cruzada com 5 folds
- Análise de importância das features

### 5. **📝 Documentação**
- Interpretação dos resultados
- Identificação de limitações
- Recomendações estratégicas

## 🎨 **Dashboard Interativo**

O projeto inclui duas versões de dashboard:

### 📓 **Versão Jupyter**
- Cards visuais com métricas principais
- Visualizações Plotly integradas
- HTML estilizado para apresentação
- Insights automáticos personalizados

### 🌐 **Versão Streamlit**
- Interface web completa
- Navegação por tabs
- Upload de arquivos
- Configurações em tempo real

## 💡 **Insights e Recomendações**

### ✅ **Pontos Fortes**
- Alta qualidade dos dados (98.9% completos)
- Separação clara dos clusters (Silhouette > 0.6)
- Modelo robusto com boa generalização
- Pipeline reproduzível e bem documentado

### 🎯 **Próximos Passos**
1. **Validação externa** com especialistas do domínio
2. **Feature engineering** baseada nas correlações identificadas
3. **Ensemble methods** para aumentar robustez
4. **Monitoramento** de drift dos dados em produção

### ⚠️ **Limitações Identificadas**
- Ausência de contexto semântico das variáveis
- Tamanho limitado do dataset (178 amostras)
- Necessidade de validação com dados externos

## 🏆 **Diferenciais Técnicos**

### 🔬 **Rigor Científico**
- Metodologia estatística robusta
- Validação cruzada para todos os modelos
- Análise crítica das limitações
- Reprodutibilidade garantida

### 💻 **Qualidade do Código**
- Código limpo e bem documentado
- Tratamento de edge cases
- Visualizações profissionais
- Interface responsiva

### 📊 **Visão de Produto**
- Dashboard pronto para apresentação
- Insights automáticos
- Scorecard de performance
- UX/UI moderna

## 📋 **Dependências**

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
plotly>=5.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
ipywidgets>=7.6.0
streamlit>=1.0.0
```

## 📄 **Licença**

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 **Autor**

**[Ednilton Curt Rauh ]**
- 💼 LinkedIn: [https://www.linkedin.com/in/ednilton-rauh-63838a47]
- 📧 Email: [edrauh@gmail.com]
- 🐱 GitHub: [https://github.com/ednilton]

## 🙏 **Agradecimentos**

- **SENAI** pela oportunidade do desafio técnico
- **Comunidade Python** pelas bibliotecas utilizadas
- **Plotly Team** pelas visualizações interativas

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela! ⭐**

![Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)
![Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)
![Data Science](https://img.shields.io/badge/Purpose-Data%20Science-green.svg)

</div>

---

### 📊 **Estatísticas do Projeto**

- **Linhas de código**: ~800+
- **Visualizações**: 15+ gráficos interativos
- **Algoritmos ML**: 3 (KMeans, RandomForest, PCA)
- **Métricas**: 10+ indicadores de performance
- **Tempo de desenvolvimento**: 2 dias
- **Nível técnico**: Sênior

### 🎯 **Keywords**

`data-science` `machine-learning` `clustering` `pca` `random-forest` `plotly` `streamlit` `jupyter` `python` `scikit-learn` `data-analysis` `visualization` `dashboard` `senai` `technical-challenge`
