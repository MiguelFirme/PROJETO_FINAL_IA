import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from projecao import projetar_patrimonio, projetar_carteira_propria
from formulario import calcular_perfil_investidor


from ML_perfil_invest import recomendar_fii


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

if "perfil_definido" not in st.session_state:
    st.session_state.perfil_definido = None

if "questionario_concluido" not in st.session_state:
    st.session_state.questionario_concluido = False

if "itens_carteira" not in st.session_state:
    st.session_state.itens_carteira = [{"ticker": "", "cotas": 0}]

if "resultado_carteira" not in st.session_state:
    st.session_state.resultado_carteira = None


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
# QUESTIONARIO DE PERFIL
# ==================================================
PERGUNTAS = [
    {
        "texto": "Por quanto tempo você pretende manter seus recursos aplicados?",
        "opcoes": [
            ("A) Menos de 1 ano", 1),
            ("B) Entre 1 e 3 anos", 2),
            ("C) Entre 3 e 5 anos", 3),
            ("D) Mais de 5 anos", 4),
        ]
    },
    {
        "texto": "Qual é a sua necessidade de liquidez (facilidade de resgate)?",
        "opcoes": [
            ("A) Preciso ter acesso imediato ao dinheiro", 1),
            ("B) Posso esperar alguns meses", 2),
            ("C) Posso esperar alguns anos", 3),
            ("D) Não tenho necessidade de liquidez no curto prazo", 4),
        ]
    },
    {
        "texto": "Como você reagiria se seu investimento sofresse uma queda de 15% em pouco tempo?",
        "opcoes": [
            ("A) Resgataria imediatamente para evitar perdas", 1),
            ("B) Reduziria parte da posição para diminuir risco", 2),
            ("C) Manteria investido aguardando recuperação", 3),
            ("D) Aproveitaria para investir mais", 4),
        ]
    },
    {
        "texto": "Qual nível de risco você está disposto a assumir para buscar maior rentabilidade?",
        "opcoes": [
            ("A) Nenhum risco, quero segurança total", 1),
            ("B) Baixo risco, aceito pequenas oscilações", 2),
            ("C) Médio risco, aceito oscilações moderadas", 3),
            ("D) Alto risco, aceito grandes oscilações", 4),
        ]
    },
    {
        "texto": "Qual percentual do seu patrimônio você estaria confortável em aplicar em produtos de maior risco?",
        "opcoes": [
            ("A) Até 10%", 1),
            ("B) Entre 10% e 25%", 2),
            ("C) Entre 25% e 50%", 3),
            ("D) Mais de 50%", 4),
        ]
    },
    {
        "texto": "Qual é o principal objetivo do seu investimento?",
        "opcoes": [
            ("A) Preservar o valor do dinheiro", 1),
            ("B) Realizar um projeto ou compra nos próximos anos", 2),
            ("C) Construir patrimônio para o futuro", 3),
            ("D) Maximizar ganhos e viver de renda/legado", 4),
        ]
    },
    {
        "texto": "Qual retorno anual você considera satisfatório?",
        "opcoes": [
            ("A) Igual ou pouco acima da poupança", 1),
            ("B) Inflação + 2%", 2),
            ("C) Retornos médios de mercado (8–12% ao ano)", 3),
            ("D) Retornos muito altos, mesmo que arriscados", 4),
        ]
    },
    {
        "texto": "Qual é sua prioridade ao investir?",
        "opcoes": [
            ("A) Segurança e liquidez", 1),
            ("B) Equilibrar segurança e crescimento", 2),
            ("C) Crescimento consistente do patrimônio", 3),
            ("D) Máxima rentabilidade, mesmo com risco elevado", 4),
        ]
    },
]

def calcular_perfil(pontuacao):
    if pontuacao <= 16:
        return "Conservador"
    elif pontuacao <= 24:
        return "Moderado"
    else:
        return "Arrojado"

def exibir_questionario():
    st.title("📈 InvestIA")
    st.markdown("### Antes de começar, vamos traçar o seu perfil de investidor")
    st.markdown("Responda as perguntas abaixo com honestidade. Não há resposta certa ou errada.")
    st.divider()

    respostas = []
    completo = True

    for i, pergunta in enumerate(PERGUNTAS):
        opcoes_texto = [op[0] for op in pergunta["opcoes"]]
        escolha = st.radio(
            f"**{i+1}. {pergunta['texto']}**",
            opcoes_texto,
            index=None,
            key=f"pergunta_{i}"
        )
        if escolha is None:
            completo = False
            respostas.append(None)
        else:
            pontos = dict(pergunta["opcoes"])[escolha]
            respostas.append(pontos)

    st.divider()

    if st.button("✅ Definir meu perfil", use_container_width=True, type="primary"):
        if not completo:
            st.warning("Por favor, responda todas as perguntas antes de continuar.")
        else:
            
            ht_valor = (respostas[0] + respostas[1]) / 2
            tr_valor = (respostas[2] + respostas[3] + respostas[4]) / 3
            of_valor = (respostas[5] + respostas[6] + respostas[7]) / 3

            
            perfil = calcular_perfil_investidor(ht_valor, tr_valor, of_valor)

            
            st.session_state.perfil_definido = perfil
            st.session_state.questionario_concluido = True
            st.rerun()
# ==================================================
# FLUXO PRINCIPAL
# ==================================================
if not st.session_state.questionario_concluido:
    exibir_questionario()
else:
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
    tab1, tab2, tab3 = st.tabs([
        "📊 Projecao Patrimonial",
        "🤖 Recomendacao de FIIs",
        "📋 Minha Carteira"
    ])

    # ==================================================
    # ABA 1 - PROJECAO PATRIMONIAL
    # ==================================================
    with tab1:
        st.header("Simulacao Patrimonial")

        col_perfil_info, col_refazer = st.columns([4, 1])
        with col_perfil_info:
            perfil_cor = {"Conservador": "🟢", "Moderado": "🟡", "Arrojado": "🔴"}
            icone = perfil_cor.get(st.session_state.perfil_definido, "⚪")
            st.info(f"{icone} Perfil identificado pelo questionário: **{st.session_state.perfil_definido}**")
        with col_refazer:
            if st.button("🔄 Refazer questionário", use_container_width=True):
                st.session_state.questionario_concluido = False
                st.session_state.perfil_definido = None
                st.rerun()

        col1, col2 = st.columns(2)

        with col1:
            _opcoes_perfil = ["Conservador", "Moderado", "Arrojado"]
            _idx_perfil = _opcoes_perfil.index(st.session_state.perfil_definido) \
                if st.session_state.perfil_definido in _opcoes_perfil else 0
            perfil = st.selectbox(
                "Perfil do Investidor",
                _opcoes_perfil,
                index=_idx_perfil,
                help="Perfil definido pelo questionário. Você pode alterar manualmente se desejar."
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
                    resultado_fii = recomendar_fii(ticker)

                    if not resultado_fii.get("ok"):
                        st.warning(resultado_fii.get("erro", "Ticker nao encontrado no dataset."))
                    else:
                        st.success("Analise concluida!")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Ticker",             resultado_fii["Ticker"])
                        col2.metric("Perfil Recomendado", resultado_fii["Perfil_Recomendado"])
                        col3.metric("Confianca",          resultado_fii["Confianca"])

                        perfil_usuario = st.session_state.perfil_definido
                        perfil_fii     = resultado_fii["Perfil_Recomendado"]
                        compativel     = perfil_usuario == perfil_fii

                        if compativel:
                            st.success(f"✅ Este FII é compativel com o seu perfil **{perfil_usuario}**!")
                        else:
                            st.warning(
                                f"⚠️ Seu perfil é **{perfil_usuario}**, mas este FII é recomendado "
                                f"para o perfil **{perfil_fii}**. Avalie se faz sentido na sua carteira."
                            )

                except Exception as erro:
                    st.error(f"Erro ao analisar: {erro}")

    # ==================================================
    # ABA 3 - MINHA CARTEIRA
    # ==================================================
    with tab3:
        st.header("📋 Minha Carteira")

        perfil_cor = {"Conservador": "🟢", "Moderado": "🟡", "Arrojado": "🔴"}
        icone = perfil_cor.get(st.session_state.perfil_definido, "⚪")
        st.info(f"{icone} Perfil do investidor: **{st.session_state.perfil_definido}**")

        st.markdown("### Informe seus ativos")
        st.caption(
            "Digite o ticker e a quantidade de cotas de cada FII. "
            "O ML vai classificar cada ativo e destacar os que não batem com seu perfil."
        )

        # ── formulário dinâmico ────────────────────────────────────────────
        itens = st.session_state.itens_carteira

        for idx in range(len(itens)):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                itens[idx]["ticker"] = st.text_input(
                    "Ticker",
                    value=itens[idx]["ticker"],
                    key=f"ticker_{idx}",
                    placeholder="Ex: KNCR11",
                    label_visibility="collapsed" if idx > 0 else "visible",
                )
            with c2:
                itens[idx]["cotas"] = st.number_input(
                    "Cotas",
                    min_value=0,
                    value=int(itens[idx]["cotas"]),
                    step=1,
                    key=f"cotas_{idx}",
                    label_visibility="collapsed" if idx > 0 else "visible",
                )
            with c3:
                if idx > 0:
                    if st.button("✕", key=f"del_{idx}", help="Remover ativo"):
                        st.session_state.itens_carteira.pop(idx)
                        st.rerun()

        col_add, col_clear = st.columns([1, 1])
        with col_add:
            if st.button("➕ Adicionar ativo", use_container_width=True):
                st.session_state.itens_carteira.append({"ticker": "", "cotas": 0})
                st.rerun()
        with col_clear:
            if st.button("🗑️ Limpar carteira", use_container_width=True):
                st.session_state.itens_carteira = [{"ticker": "", "cotas": 0}]
                st.session_state.resultado_carteira = None
                st.rerun()

        st.divider()

        # ── parâmetros da projeção ─────────────────────────────────────────
        anos_carteira = st.slider(
            "⏳ Horizonte de Investimento (anos)",
            min_value=1, max_value=40, value=10,
            key="anos_carteira"
        )
        st.caption("📌 Os dividendos recebidos serão automaticamente reinvestidos na carteira mês a mês.")

        # aporte extra zerado e reinvestimento sempre ativo
        aporte_mensal_carteira = 0.0
        reinvestir_carteira    = True

        # ── botão analisar ─────────────────────────────────────────────────
        if st.button("🔍 Analisar Carteira e Gerar Projeção", use_container_width=True, type="primary"):
            ativos_validos = [
                i for i in st.session_state.itens_carteira
                if i["ticker"].strip() and i["cotas"] > 0
            ]
            if not ativos_validos:
                st.warning("Adicione ao menos um ativo com ticker e quantidade de cotas.")
            else:
                with st.spinner(f"Buscando dados e classificando {len(ativos_validos)} ativo(s)..."):
                    res = projetar_carteira_propria(
                        carteira=ativos_validos,
                        perfil_usuario=st.session_state.perfil_definido,
                        aporte_mensal=aporte_mensal_carteira,
                        anos=anos_carteira,
                        reinvestir_dividendos=reinvestir_carteira,
                    )
                    if res.get("ok"):
                        st.session_state.resultado_carteira = res
                        st.success("Análise concluída!")
                    else:
                        st.error(res.get("erro", "Erro ao analisar carteira."))

        # ── resultado ─────────────────────────────────────────────────────
        if st.session_state.resultado_carteira:
            res      = st.session_state.resultado_carteira
            resumo   = res["resumo"]
            itens_ml = res["itens_carteira"]

            st.divider()

            # métricas rápidas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Patrimônio Atual",       formatar_moeda(resumo["patrimonio_inicial"]))
            m2.metric("Patrimônio Projetado",   formatar_moeda(resumo["patrimonio_final_carteira"]))
            m3.metric("Renda Passiva Final",    formatar_moeda(resumo["renda_passiva_mensal_final"]))
            m4.metric("Vantagem vs Poupança",   formatar_moeda(resumo["vantagem_vs_poupanca"]))

            st.divider()

            # ── tabela de ativos com sinalização ML ───────────────────────
            st.subheader("🔬 Análise ML dos Ativos")

            n_incomp = sum(1 for i in itens_ml if not i["compativel"])
            if n_incomp > 0:
                st.warning(
                    f"⚠️ **{n_incomp} ativo(s)** não são compatíveis com seu perfil "
                    f"**{st.session_state.perfil_definido}** segundo o modelo de IA."
                )
            else:
                st.success("✅ Todos os ativos são compatíveis com o seu perfil!")

            # Tabela com cor via HTML
            linhas_html = ""
            for item in itens_ml:
                cor_bg  = "#3d0000" if not item["compativel"] else "#003d00"
                cor_txt = "#ff6b6b" if not item["compativel"] else "#6bff6b"
                icone_compat = "❌" if not item["compativel"] else "✅"
                dy_anual = ((1 + item["dy_mensal"]) ** 12 - 1) * 100

                linhas_html += f"""
                <tr style="background:{cor_bg}">
                    <td style="color:{cor_txt};font-weight:bold;padding:8px 12px">{item['ticker']}</td>
                    <td style="padding:8px 12px">{int(item['cotas'])}</td>
                    <td style="padding:8px 12px">R$ {item['preco']:,.2f}</td>
                    <td style="padding:8px 12px">R$ {item['valor_total']:,.2f}</td>
                    <td style="padding:8px 12px">{item['peso_pct']:.1f}%</td>
                    <td style="padding:8px 12px">{dy_anual:.2f}% a.a.</td>
                    <td style="color:{cor_txt};padding:8px 12px">{item['perfil_ml']}</td>
                    <td style="padding:8px 12px">{item['confianca']}</td>
                    <td style="padding:8px 12px;font-size:18px">{icone_compat}</td>
                </tr>"""

            tabela_html = f"""
            <style>
              .tb-carteira {{
                width:100%; border-collapse:collapse;
                font-family:sans-serif; font-size:14px;
                background:#0e1117; border-radius:12px; overflow:hidden;
              }}
              .tb-carteira th {{
                background:#1a2a4a; color:#a0c4ff;
                padding:10px 12px; text-align:left; font-weight:600;
              }}
              .tb-carteira td {{ color:#e0e8f4; border-top:1px solid #1a2a4a; }}
            </style>
            <table class="tb-carteira">
              <thead>
                <tr>
                  <th>Ticker</th><th>Cotas</th><th>Preço Atual</th>
                  <th>Valor Total</th><th>Peso</th><th>DY Anual</th>
                  <th>Perfil ML</th><th>Confiança</th><th>Ok?</th>
                </tr>
              </thead>
              <tbody>{linhas_html}</tbody>
            </table>"""

            components.html(tabela_html, height=60 + len(itens_ml) * 44, scrolling=False)

            st.divider()

            # ── resumo inteligente e gráficos ─────────────────────────────
            exibir_resumo_inteligente(
                {
                    "perfil":          st.session_state.perfil_definido,
                    "aporte_inicial":  resumo["patrimonio_inicial"],
                    "aporte_mensal":   aporte_mensal_carteira,
                    "anos":            anos_carteira,
                    "reinvestir":      reinvestir_carteira,
                },
                res,
            )

            exibir_detalhes_projecao(res)

    # ==================================================
    # RODAPE
    # ==================================================
    st.divider()

    st.caption(
        "InvestIA • Projeto de Inteligencia Artificial para Recomendacao de Investimentos Imobiliarios"
    )
