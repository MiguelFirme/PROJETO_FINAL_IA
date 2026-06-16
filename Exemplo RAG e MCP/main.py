import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from projecao import projetar_patrimonio

# Importar somente se voce criar modelo_fii.py
# from modelo_fii import recomendar_fii


# ==================================================
# CONFIGURACAO DA PAGINA
# ==================================================
st.set_page_config(
    page_title="InvestIA",
    page_icon="📈",
    layout="wide"
)

# ==================================================
# ESTADO DA SESSAO
# ==================================================
if "resultado_projecao" not in st.session_state:
    st.session_state.resultado_projecao = None

if "ultima_simulacao" not in st.session_state:
    st.session_state.ultima_simulacao = None


# ==================================================
# FUNCOES AUXILIARES
# ==================================================
def formatar_moeda(valor):
    try:
        formatado = f"{valor:,.2f}"
        formatado = formatado.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {formatado}"
    except Exception:
        return "R$ 0,00"


def exibir_resumo_inteligente(dados_simulacao, resultado):
    resumo     = resultado.get("resumo", {})
    perfil     = dados_simulacao["perfil"]
    ap_inicial = formatar_moeda(dados_simulacao["aporte_inicial"])
    ap_mensal  = formatar_moeda(dados_simulacao["aporte_mensal"])
    anos       = dados_simulacao["anos"]
    texto_ia   = resultado.get("texto_para_llm", "Resumo inteligente indisponivel no momento.")

    # Altura dinamica baseada no tamanho do texto da IA
    altura = max(420, 380 + len(texto_ia) // 3)

    # components.html renderiza HTML puro, sem interferencia do Markdown do Streamlit
    html = f"""
    <style>
      body {{ margin: 0; padding: 0; background: transparent; font-family: sans-serif; }}
      .card {{
        background: linear-gradient(135deg, #071426, #0a1f3d);
        border: 1px solid #1f6feb;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 0 25px rgba(0,100,255,0.15);
      }}
      .titulo {{
        color: white;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 6px;
      }}
      .sub {{
        color: #b8c6db;
        font-size: 16px;
        margin-bottom: 22px;
      }}
      .perfil-box {{
        background: rgba(20,40,80,0.55);
        border: 1px solid #2d7ff9;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 22px;
        color: #e0e8f4;
        font-size: 15px;
      }}
      .perfil-text {{
        color: white;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
      }}
      .perfil-nome {{ color: #2ea8ff; }}
      .sep {{ margin: 0 10px; color: #4a6fa5; }}
      .ia-box {{
        background: rgba(10,25,50,0.85);
        border: 1px solid #0094ff;
        border-radius: 14px;
        padding: 22px;
      }}
      .ia-title {{
        color: white;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 14px;
      }}
      .ia-text {{
        color: #f4f4f4;
        font-size: 16px;
        line-height: 1.8;
      }}
    </style>
    <div class="card">
      <div class="titulo">✨ Resumo Inteligente</div>
      <div class="sub">Analise personalizada do seu plano de investimentos</div>

      <div class="perfil-box">
        <div class="perfil-text">
          Perfil <span class="perfil-nome">{perfil}</span>
        </div>
        💰 Aporte Inicial: <b>{ap_inicial}</b>
        <span class="sep">|</span>
        📅 Aporte Mensal: <b>{ap_mensal}</b>
        <span class="sep">|</span>
        ⏳ Horizonte: <b>{anos} anos</b>
      </div>

      <div class="ia-box">
        <div class="ia-title">⭐ O que a IA descobriu para voce</div>
        <div class="ia-text">{texto_ia}</div>
      </div>
    </div>
    """
    components.html(html, height=altura, scrolling=True)

    st.subheader("📊 Destaques da Projecao")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Patrimonio Projetado",
        formatar_moeda(resumo.get("patrimonio_final_carteira", 0))
    )
    d2.metric(
        "Total Investido",
        formatar_moeda(resumo.get("total_investido", 0))
    )
    d3.metric(
        "Ganho Obtido",
        formatar_moeda(resumo.get("ganho_sobre_investido", 0))
    )
    d4.metric(
        "Renda Passiva",
        formatar_moeda(resumo.get("renda_passiva_mensal_final", 0))
    )


def exibir_detalhes_projecao(resultado):
    st.divider()

    st.subheader("Composicao da Carteira")
    df_classes = pd.DataFrame(resultado.get("carteira_por_classe", []))

    if not df_classes.empty:
        # Renomear colunas para exibicao mais legivel
        renomear = {
            "classe":               "Classe",
            "peso_pct":             "Peso (%)",
            "dy_anual_pct":         "DY Anual (%)",
            "renda_mensal_estimada":"Renda Mensal Estimada (R$)",
        }
        df_exibir = df_classes.rename(columns={
            k: v for k, v in renomear.items() if k in df_classes.columns
        })
        st.dataframe(df_exibir, use_container_width=True)
    else:
        st.info("Nao ha dados de composicao da carteira para exibir.")

    st.divider()

    st.subheader("Evolucao Patrimonial")
    serie = pd.DataFrame(resultado.get("serie_temporal", []))

    if not serie.empty and "ano" in serie.columns:
        colunas_plot = [col for col in ["carteira", "poupanca"] if col in serie.columns]
        if colunas_plot:
            st.line_chart(serie.set_index("ano")[colunas_plot])
        else:
            st.info("Nao ha colunas suficientes para gerar o grafico de evolucao.")
    else:
        st.info("Nao ha serie temporal disponivel.")

    st.divider()

    st.subheader("Graficos Gerados")
    graficos = resultado.get("graficos", {})
    exibiu_algum = False

    for chave, legenda in [
        ("comparativo", "Carteira x Poupanca"),
        ("composicao", "Composicao da Carteira"),
        ("renda_passiva", "Renda Passiva"),
    ]:
        if chave in graficos:
            try:
                st.image(graficos[chave], caption=legenda)
                exibiu_algum = True
            except Exception as e:
                st.warning(f"Nao foi possivel exibir o grafico '{legenda}': {e}")

    if not exibiu_algum:
        st.info("Nenhum grafico foi gerado para esta simulacao.")


# ==================================================
# CABECALHO
# ==================================================
st.title("📈 InvestIA")

st.markdown("""
### Plataforma Inteligente de Investimentos Imobiliarios

Sistema baseado em Inteligencia Artificial para:

- Simular crescimento patrimonial
- Estimar renda passiva futura
- Comparar desempenho com a poupanca
- Recomendar FIIs adequados com apoio de Machine Learning
""")

st.divider()

# ==================================================
# ABAS
# ==================================================
tab1, tab2 = st.tabs([
    "📊 Projecao Patrimonial",
    "🤖 Recomendacao de FIIs"
])

# ==================================================
# ABA 1 - PROJECAO PATRIMONIAL
# ==================================================
with tab1:
    st.header("Simulacao Patrimonial")

    col1, col2 = st.columns(2)

    with col1:
        perfil = st.selectbox(
            "Perfil do Investidor",
            ["Conservador", "Moderado", "Arrojado"]
        )

        aporte_inicial = st.number_input(
            "Aporte Inicial (R$)",
            min_value=0.0,
            value=10000.0,
            step=1000.0
        )

    with col2:
        aporte_mensal = st.number_input(
            "Aporte Mensal (R$)",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        anos = st.slider(
            "Horizonte de Investimento (anos)",
            min_value=1,
            max_value=40,
            value=20
        )

    reinvestir = st.checkbox(
        "Reinvestir Dividendos",
        value=True
    )

    if st.button("🚀 Gerar Projecao", use_container_width=True):
        with st.spinner("Calculando projecao..."):
            try:
                res = projetar_patrimonio(
                    perfil=perfil,
                    aporte_inicial=aporte_inicial,
                    aporte_mensal=aporte_mensal,
                    anos=anos,
                    reinvestir_dividendos=reinvestir
                )

                if res.get("ok"):
                    st.session_state.resultado_projecao = res
                    st.session_state.ultima_simulacao = {
                        "perfil": perfil,
                        "aporte_inicial": aporte_inicial,
                        "aporte_mensal": aporte_mensal,
                        "anos": anos,
                        "reinvestir": reinvestir
                    }
                    st.success("Projecao gerada com sucesso!")
                else:
                    st.session_state.resultado_projecao = None
                    st.session_state.ultima_simulacao = None
                    st.error(res.get("erro", "Erro ao gerar projecao."))

            except Exception as erro:
                st.session_state.resultado_projecao = None
                st.session_state.ultima_simulacao = None
                st.error(f"Erro inesperado ao gerar a projecao: {erro}")

    if st.session_state.resultado_projecao and st.session_state.ultima_simulacao:
        resultado = st.session_state.resultado_projecao
        resumo = resultado.get("resumo", {})

        st.divider()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Patrimonio Final",
            formatar_moeda(resumo.get("patrimonio_final_carteira", 0))
        )
        c2.metric(
            "Renda Passiva Mensal",
            formatar_moeda(resumo.get("renda_passiva_mensal_final", 0))
        )
        c3.metric(
            "Total Investido",
            formatar_moeda(resumo.get("total_investido", 0))
        )
        c4.metric(
            "Vantagem vs Poupanca",
            formatar_moeda(resumo.get("vantagem_vs_poupanca", 0))
        )

        st.divider()

        exibir_resumo_inteligente(
            st.session_state.ultima_simulacao,
            st.session_state.resultado_projecao
        )

        exibir_detalhes_projecao(st.session_state.resultado_projecao)

# ==================================================
# ABA 2 - RECOMENDACAO DE FIIs
# ==================================================
with tab2:
    st.header("Recomendacao Inteligente de FIIs")

    st.write(
        "Digite o ticker de um Fundo Imobiliario para que o modelo de IA determine o perfil recomendado."
    )

    ticker_input = st.text_input(
        "Ticker",
        placeholder="Ex: HGLG11"
    )

    if st.button("🔍 Analisar FII", use_container_width=True):
        ticker = ticker_input.strip().upper()

        if not ticker:
            st.warning("Digite um ticker.")
        else:
            try:
                # resultado = recomendar_fii(ticker)

                resultado_fii = {
                    "Ticker": ticker,
                    "Perfil_Recomendado": "Moderado",
                    "Confianca": "92.50%"
                }

                st.success("Analise concluida!")

                col1, col2, col3 = st.columns(3)

                col1.metric("Ticker", resultado_fii["Ticker"])
                col2.metric("Perfil Recomendado", resultado_fii["Perfil_Recomendado"])
                col3.metric("Confianca", resultado_fii["Confianca"])

            except Exception as erro:
                st.error(f"Erro ao analisar: {erro}")

# ==================================================
# RODAPE
# ==================================================
st.divider()

st.caption(
    "InvestIA • Projeto de Inteligencia Artificial para Recomendacao de Investimentos Imobiliarios"
)