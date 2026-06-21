# 📈 InvestIA — Sistema Inteligente de Recomendação de Investimentos

Plataforma web desenvolvido como projeto final da disciplina de **Inteligência Artificial** (SATC, 2026) que combina **Lógica Fuzzy** e **Machine Learning** para auxiliar investidores na escolha de Fundos de Investimento Imobiliário (FIIs) adequados ao seu perfil.

---

## ✨ Funcionalidades

- **Questionário de Perfil** — 8 perguntas classificam o investidor como Conservador, Moderado ou Arrojado via Lógica Fuzzy
- **Recomendação de FIIs** — modelo XGBoost analisa atributos financeiros do ativo e indica o perfil ideal, comparando com o perfil do usuário
- **Projeção Patrimonial** — simula crescimento do patrimônio por até 40 anos com aportes, dividendos e comparativo contra a poupança
- **Minha Carteira** — o usuário cadastra seus ativos (ticker + cotas) e recebe análise ML de compatibilidade e projeção baseada na carteira real
- **Dados ao vivo** — DY mensal buscado em tempo real no [Status Invest](https://statusinvest.com.br), com fallback para médias históricas do dataset

---

## 🗂️ Estrutura do Projeto

```
PROJETO_FINAL_IA/
├── app.py                        # Interface Streamlit (questionário + 3 abas)
├── ML_perfil_invest.py           # Treinamento XGBoost + função recomendar_fii()
├── projecao.py                   # Motor de projeção patrimonial + scraping
├── dataset_fiis_b3_refinado.csv  # Dataset de FIIs da B3 com rótulos de perfil
├── modelo_xgb.pkl                # Modelo XGBoost serializado (gerado automaticamente)
├── modelo_colunas.pkl            # Schema de features do modelo
├── graficos_projecao/            # Gráficos gerados nas simulações
├── Exemplo RAG e MCP/            # Exemplo complementar com RAG e MCP
└── requirements.txt              # Dependências do projeto
```

---

## 🧠 Tecnologias de IA

| Técnica | Biblioteca | Aplicação |
|---|---|---|
| Lógica Fuzzy | `scikit-fuzzy` | Classificação do perfil do investidor |
| Gradient Boosting | `xgboost` | Recomendação de FIIs por perfil |
| Feature Engineering | `pandas` / `numpy` | Log, razões e índices financeiros |
| Web Scraping | `beautifulsoup4` | DY ao vivo do Status Invest |
| Visualização | `matplotlib` | Gráficos de projeção patrimonial |
| Interface | `streamlit` | Aplicação web interativa |

---

## 🚀 Como executar

**1. Clone o repositório**
```bash
git clone https://github.com/seu-usuario/PROJETO_FINAL_IA.git
cd PROJETO_FINAL_IA
```

**2. Instale as dependências**
```bash
pip install -r requirements.txt
```

**3. Execute a aplicação**
```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador.

> **Nota:** os arquivos `modelo_xgb.pkl` e `modelo_colunas.pkl` já estão incluídos. O modelo só será retreinado se esses arquivos forem removidos.

---

## 📊 Fluxo do Sistema

```
Usuário responde questionário (8 perguntas)
        ↓
Lógica Fuzzy define o perfil (Conservador / Moderado / Arrojado)
        ↓
┌──────────────────────────────────────────────────────┐
│  Aba 1: Recomendação de FIIs                         │
│  → XGBoost classifica o ticker informado             │
│  → Verifica compatibilidade com perfil do usuário    │
├──────────────────────────────────────────────────────┤
│  Aba 2: Minha Carteira                               │
│  → ML analisa cada ativo e aponta incompatibilidades │
│  → Projeção baseada na carteira real do usuário      │
└──────────────────────────────────────────────────────┘
```

---

## 🤖 Modelo de Machine Learning

- **Algoritmo:** XGBoost (multiclasse — 3 perfis)
- **Features principais:** `DY_Anual`, `P_VP`, `Log_Patrimonio`, `Log_Liquidez`, `Indice_Valor`, `DY_Ajustado`, `Desconto_VP`, `Liquidez_Patrimonio` + segmento (One-Hot)
- **Divisão:** 80% treino / 20% teste (estratificado)
- **Validação:** Cross-validation 5-fold com `f1_weighted`

---

## 📋 Requisitos

- Python 3.11+
- Conexão com a internet (para scraping de DY ao vivo)

---

## 📁 Dataset

O dataset `dataset_fiis_b3_refinado.csv` contém FIIs reais da B3 com atributos financeiros como DY anual, P/VP, liquidez diária, vacância e patrimônio líquido, além do rótulo `Perfil_Ideal_Investidor` usado no treinamento supervisionado.

---

## 👥 Equipe

Dado projeto foi desenvolvido por:

Davi - João Gustavo - Lucas - Lorenzo - Miguel

---

## ⚠️ Aviso

Este sistema é uma ferramenta **educacional e de apoio à decisão**. Não realiza operações financeiras reais nem substitui a orientação de um profissional certificado.