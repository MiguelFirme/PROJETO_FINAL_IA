"""
projecao.py
-----------
Calcula a projeção patrimonial levando em conta:
  - Valorização anual estimada da cota (por perfil)
  - Dividendo mensal (DY% ao mês, reinvestido ou não)
  - Aportes mensais

Busca dados ao vivo no Status Invest para uma cesta de FIIs
representativa de cada perfil. Usa médias do dataset como fallback.

Expõe:
    projetar_patrimonio(perfil, aporte_inicial, aporte_mensal,
                        anos, reinvestir_dividendos) -> dict
"""

import io
import base64
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from time import sleep

from ML_perfil_invest import PERFIS_DY

# ------------------------------------------------------------------
# CESTA DE FIIs REPRESENTATIVOS POR PERFIL
# (busca ao vivo; se falhar usa fallback do dataset)
# ------------------------------------------------------------------

CESTA_PERFIL: dict[str, list[str]] = {
    "Conservador": ["KNCR11", "MXRF11", "BTLG11", "RZTR11", "CPTS11"],
    "Moderado":    ["HGLG11", "XPML11", "VISC11", "BCFF11", "RBRF11"],
    "Arrojado":    ["VGIA11", "RURA11", "TGAR11", "DEVA11", "RZAG11"],
}

# Valorização anual estimada da cota por perfil
# (conservador = menor risco, menor upside; arrojado = maior volatilidade)
VALORIZACAO_ANUAL: dict[str, float] = {
    "Conservador": 0.04,   # ~4% a.a.
    "Moderado":    0.07,   # ~7% a.a.
    "Arrojado":    0.10,   # ~10% a.a.
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


# ------------------------------------------------------------------
# SCRAPING STATUS INVEST
# ------------------------------------------------------------------

def _buscar_dy_live(ticker: str) -> float | None:
    """Retorna o DY mensal (0 a 1) do ticker ou None se falhar."""
    urls = [
        f"https://statusinvest.com.br/fundos-imobiliarios/{ticker.lower()}",
        f"https://statusinvest.com.br/fiagros/{ticker.lower()}",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            rend_tag = soup.select_one("b.sub-value")
            if rend_tag:
                valor = float(rend_tag.text.replace(",", "."))
                return valor / 100  # ex: 1.28 → 0.0128
        except Exception:
            continue
    return None


def _dy_medio_perfil(perfil: str) -> float:
    """
    Tenta buscar o DY mensal médio ao vivo para a cesta do perfil.
    Se ≥ 2 tickers responderem, usa a média deles.
    Caso contrário, usa a média do dataset (PERFIS_DY) convertida p/ mensal.
    """
    tickers = CESTA_PERFIL.get(perfil, [])
    dys = []

    for ticker in tickers:
        dy = _buscar_dy_live(ticker)
        if dy is not None:
            dys.append(dy)
        sleep(1.5)          # respeita o rate limit do site

    if len(dys) >= 2:
        return float(np.mean(dys))

    # fallback: DY anual do dataset → mensal composto
    dy_anual = PERFIS_DY.get(perfil, 0.09)
    return (1 + dy_anual) ** (1 / 12) - 1


# ------------------------------------------------------------------
# MOTOR DE PROJEÇÃO
# ------------------------------------------------------------------

def _simular(
    patrimonio_inicial: float,
    aporte_mensal: float,
    anos: int,
    dy_mensal: float,
    valorizacao_anual: float,
    reinvestir: bool,
) -> list[dict]:
    """
    Simula mês a mês e retorna lista de snapshots anuais.

    A cada mês:
      1. Aplica valorização mensal da cota ao patrimônio
      2. Calcula dividendo = patrimônio * dy_mensal
      3. Se reinvestir: soma dividendo ao patrimônio
         Senão: acumula dividendo separado (renda passiva)
      4. Soma o aporte mensal
    """
    val_mensal = (1 + valorizacao_anual) ** (1 / 12) - 1

    patrimonio    = patrimonio_inicial
    renda_acum    = 0.0       # dividendos NÃO reinvestidos acumulados
    total_aportes = patrimonio_inicial
    serie = []

    for mes in range(1, anos * 12 + 1):
        # 1. valorização da cota
        patrimonio *= (1 + val_mensal)

        # 2. dividendo
        dividendo = patrimonio * dy_mensal

        # 3. reinvestimento ou renda passiva
        if reinvestir:
            patrimonio += dividendo
        else:
            renda_acum += dividendo

        # 4. aporte
        patrimonio    += aporte_mensal
        total_aportes += aporte_mensal

        # snapshot anual
        if mes % 12 == 0:
            ano = mes // 12
            patrimonio_total = patrimonio + renda_acum
            renda_passiva_mes = patrimonio * dy_mensal   # estimativa do mês seguinte

            serie.append({
                "ano":            ano,
                "carteira":       round(patrimonio_total, 2),
                "total_aportes":  round(total_aportes, 2),
                "renda_passiva":  round(renda_passiva_mes, 2),
            })

    return serie


def _simular_poupanca(
    patrimonio_inicial: float,
    aporte_mensal: float,
    anos: int,
) -> list[dict]:
    """Poupança: ~6% a.a. = 0.4868% a.m."""
    taxa_mensal = (1 + 0.06) ** (1 / 12) - 1
    patrimonio  = patrimonio_inicial
    serie = []

    for mes in range(1, anos * 12 + 1):
        patrimonio = patrimonio * (1 + taxa_mensal) + aporte_mensal
        if mes % 12 == 0:
            serie.append({"ano": mes // 12, "poupanca": round(patrimonio, 2)})

    return serie


# ------------------------------------------------------------------
# GRÁFICOS
# ------------------------------------------------------------------

def _grafico_comparativo(serie_carteira: list[dict], serie_poupanca: list[dict]) -> str:
    anos      = [s["ano"] for s in serie_carteira]
    carteira  = [s["carteira"] for s in serie_carteira]
    poupanca  = [s["poupanca"] for s in serie_poupanca]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(anos, carteira, label="Carteira FIIs", color="#2ea8ff", linewidth=2.5)
    ax.plot(anos, poupanca, label="Poupança",      color="#f0a500", linewidth=2,  linestyle="--")
    ax.set_xlabel("Anos")
    ax.set_ylabel("R$")
    ax.set_title("Crescimento Patrimonial: Carteira FIIs vs Poupança")
    ax.legend()
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"R$ {x:,.0f}".replace(",", "."))
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf


def _grafico_renda_passiva(serie: list[dict]) -> str:
    anos   = [s["ano"] for s in serie]
    rendas = [s["renda_passiva"] for s in serie]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(anos, rendas, color="#2ea8ff", alpha=0.85)
    ax.set_xlabel("Anos")
    ax.set_ylabel("R$ / mês")
    ax.set_title("Estimativa de Renda Passiva Mensal por Ano")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"R$ {x:,.0f}".replace(",", "."))
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf


def _grafico_composicao(perfil: str) -> io.BytesIO:
    classes = _classes_por_perfil(perfil)
    labels  = [c["classe"] for c in classes]
    pesos   = [c["peso_pct"] for c in classes]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        pesos, labels=labels, autopct="%1.1f%%",
        startangle=140,
        colors=["#2ea8ff", "#1a6fc4", "#0d4a8c", "#5bc8ff", "#a0d8ff"],
    )
    ax.set_title(f"Composição da Carteira — Perfil {perfil}")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf


# ------------------------------------------------------------------
# COMPOSIÇÃO POR PERFIL
# ------------------------------------------------------------------

def _classes_por_perfil(perfil: str) -> list[dict]:
    composicoes = {
        "Conservador": [
            {"classe": "Papel / CRI (CDI)",  "peso_pct": 50, "dy_anual_pct": 11.5},
            {"classe": "Logística / Tijolo", "peso_pct": 30, "dy_anual_pct":  9.0},
            {"classe": "Híbrido",            "peso_pct": 20, "dy_anual_pct":  9.8},
        ],
        "Moderado": [
            {"classe": "Shoppings",          "peso_pct": 35, "dy_anual_pct":  9.5},
            {"classe": "Logística / Tijolo", "peso_pct": 30, "dy_anual_pct":  9.0},
            {"classe": "Papel / CRI",        "peso_pct": 20, "dy_anual_pct": 10.5},
            {"classe": "FoF",                "peso_pct": 15, "dy_anual_pct":  8.5},
        ],
        "Arrojado": [
            {"classe": "Fiagro / Agro",      "peso_pct": 40, "dy_anual_pct": 14.0},
            {"classe": "Desenvolvimento",    "peso_pct": 30, "dy_anual_pct": 12.5},
            {"classe": "Papel High Yield",   "peso_pct": 30, "dy_anual_pct": 13.0},
        ],
    }
    return composicoes.get(perfil, composicoes["Moderado"])


# ------------------------------------------------------------------
# TEXTO LLM (template rico baseado nos números calculados)
# ------------------------------------------------------------------

def _gerar_texto_llm(
    perfil: str,
    aporte_inicial: float,
    aporte_mensal: float,
    anos: int,
    dy_mensal: float,
    resumo: dict,
    reinvestir: bool,
) -> str:
    dy_anual_pct = ((1 + dy_mensal) ** 12 - 1) * 100
    pat_final    = resumo["patrimonio_final_carteira"]
    total_inv    = resumo["total_investido"]
    ganho        = resumo["ganho_sobre_investido"]
    renda        = resumo["renda_passiva_mensal_final"]
    vantagem     = resumo["vantagem_vs_poupanca"]
    multiplicador = pat_final / total_inv if total_inv > 0 else 1

    reinv_texto = (
        "reinvestindo todos os dividendos recebidos mês a mês"
        if reinvestir
        else "sacando os dividendos como renda passiva mensal"
    )

    return (
        f"Com um perfil <b>{perfil}</b> e {reinv_texto}, sua simulação aponta para um "
        f"patrimônio projetado de <b>R$ {pat_final:,.2f}</b> ao final de <b>{anos} anos</b>. "
        f"Você terá investido no total <b>R$ {total_inv:,.2f}</b> e o mercado terá feito "
        f"o resto: um ganho estimado de <b>R$ {ganho:,.2f}</b>, multiplicando seu capital "
        f"<b>{multiplicador:.1f}×</b>.<br><br>"
        f"O DY médio utilizado na simulação foi de <b>{dy_anual_pct:.2f}% ao ano</b> "
        f"({dy_mensal*100:.3f}% ao mês), baseado em dados reais buscados ao vivo no "
        f"Status Invest para FIIs representativos do seu perfil.<br><br>"
        f"Ao final do período, a estimativa de renda passiva mensal gerada pela carteira é de "
        f"<b>R$ {renda:,.2f}/mês</b> — o equivalente a "
        f"<b>{renda / 1412:.1f}× o salário mínimo</b> atual.<br><br>"
        f"Comparado à poupança, sua carteira de FIIs entregaria uma vantagem de "
        f"<b>R$ {vantagem:,.2f}</b> a mais ao longo desse período. "
        f"Isso mostra o poder dos dividendos mensais <i>compostos</i> ao longo do tempo."
    ).replace(",", "X").replace(".", ",").replace("X", ".")


# ------------------------------------------------------------------
# FUNÇÃO PÚBLICA PRINCIPAL
# ------------------------------------------------------------------

def projetar_patrimonio(
    perfil: str,
    aporte_inicial: float,
    aporte_mensal: float,
    anos: int,
    reinvestir_dividendos: bool = True,
) -> dict:
    """
    Retorna dict completo compatível com app.py:
    {
        "ok": True,
        "resumo": {...},
        "serie_temporal": [...],
        "carteira_por_classe": [...],
        "texto_para_llm": "...",
        "graficos": {
            "comparativo": <BytesIO>,
            "composicao":  <BytesIO>,
            "renda_passiva": <BytesIO>,
        }
    }
    """
    try:
        # 1. Busca DY ao vivo
        dy_mensal        = _dy_medio_perfil(perfil)
        valorizacao_anual = VALORIZACAO_ANUAL.get(perfil, 0.07)

        # 2. Simulação da carteira FIIs
        serie_fiis = _simular(
            patrimonio_inicial=aporte_inicial,
            aporte_mensal=aporte_mensal,
            anos=anos,
            dy_mensal=dy_mensal,
            valorizacao_anual=valorizacao_anual,
            reinvestir=reinvestir_dividendos,
        )

        # 3. Simulação da poupança (comparativo)
        serie_poup = _simular_poupanca(
            patrimonio_inicial=aporte_inicial,
            aporte_mensal=aporte_mensal,
            anos=anos,
        )

        # 4. Série temporal unificada (para o line_chart do app.py)
        serie_temporal = []
        poup_dict = {s["ano"]: s["poupanca"] for s in serie_poup}
        for s in serie_fiis:
            serie_temporal.append({
                "ano":      s["ano"],
                "carteira": s["carteira"],
                "poupanca": poup_dict.get(s["ano"], 0),
            })

        # 5. Resumo
        ultimo          = serie_fiis[-1]
        total_investido = aporte_inicial + aporte_mensal * anos * 12
        pat_final       = ultimo["carteira"]
        poup_final      = serie_poup[-1]["poupanca"]

        resumo = {
            "patrimonio_final_carteira":  round(pat_final, 2),
            "total_investido":            round(total_investido, 2),
            "ganho_sobre_investido":      round(pat_final - total_investido, 2),
            "renda_passiva_mensal_final": round(ultimo["renda_passiva"], 2),
            "poupanca_final":             round(poup_final, 2),
            "vantagem_vs_poupanca":       round(pat_final - poup_final, 2),
        }

        # 6. Composição por classe
        classes = _classes_por_perfil(perfil)
        for c in classes:
            dy_anual = c["dy_anual_pct"] / 100
            c["renda_mensal_estimada"] = round(
                pat_final * (c["peso_pct"] / 100) * ((1 + dy_anual) ** (1/12) - 1), 2
            )

        # 7. Gráficos
        buf_comp  = _grafico_comparativo(serie_fiis, serie_poup)
        buf_pizza = _grafico_composicao(perfil)
        buf_renda = _grafico_renda_passiva(serie_fiis)

        # 8. Texto IA
        texto = _gerar_texto_llm(
            perfil, aporte_inicial, aporte_mensal,
            anos, dy_mensal, resumo, reinvestir_dividendos,
        )

        return {
            "ok":                True,
            "resumo":            resumo,
            "serie_temporal":    serie_temporal,
            "carteira_por_classe": classes,
            "texto_para_llm":    texto,
            "graficos": {
                "comparativo":   buf_comp,
                "composicao":    buf_pizza,
                "renda_passiva": buf_renda,
            },
            "dy_mensal_usado":   round(dy_mensal * 100, 4),
        }

    except Exception as e:
        return {"ok": False, "erro": str(e)}

# ------------------------------------------------------------------
# PROJEÇÃO POR CARTEIRA PRÓPRIA
# ------------------------------------------------------------------

def _buscar_preco_e_dy(ticker: str) -> tuple[float | None, float | None]:
    """Busca preço atual e DY mensal do ticker no Status Invest."""
    urls = [
        f"https://statusinvest.com.br/fundos-imobiliarios/{ticker.lower()}",
        f"https://statusinvest.com.br/fiagros/{ticker.lower()}",
        f"https://statusinvest.com.br/acoes/{ticker.lower()}",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            preco_tag = soup.select_one("strong.value")
            rend_tag  = soup.select_one("b.sub-value")
            if preco_tag and rend_tag:
                preco     = float(preco_tag.text.replace(",", "."))
                dy_mensal = float(rend_tag.text.replace(",", ".")) / 100
                return preco, dy_mensal
        except Exception:
            continue
    return None, None


def _grafico_composicao_carteira(itens: list[dict]) -> io.BytesIO:
    """Pizza com o peso de cada ticker na carteira."""
    labels = [i["ticker"] for i in itens]
    valores = [i["valor_total"] for i in itens]

    cores = plt.cm.Blues(np.linspace(0.35, 0.85, len(labels)))

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        valores, labels=labels, autopct="%1.1f%%",
        startangle=140, colors=cores,
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title("Composição da Carteira — Peso por Ticker")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf


def _grafico_comparativo_carteira(serie_fiis, serie_poup) -> io.BytesIO:
    anos     = [s["ano"] for s in serie_fiis]
    carteira = [s["carteira"] for s in serie_fiis]
    poupanca = [s["poupanca"] for s in serie_poup]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(anos, carteira, label="Sua Carteira", color="#2ea8ff", linewidth=2.5)
    ax.plot(anos, poupanca, label="Poupança",     color="#f0a500", linewidth=2, linestyle="--")
    ax.set_xlabel("Anos")
    ax.set_ylabel("R$")
    ax.set_title("Evolução Patrimonial: Sua Carteira vs Poupança")
    ax.legend()
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"R$ {x:,.0f}".replace(",", "."))
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf


def projetar_carteira_propria(
    carteira: list[dict],   # [{"ticker": "KNCR11", "cotas": 25}, ...]
    perfil_usuario: str,
    aporte_mensal: float,
    anos: int,
    reinvestir_dividendos: bool = True,
) -> dict:
    """
    Projeta o patrimônio com base na carteira real do usuário.

    Busca preço + DY ao vivo de cada ticker, calcula o patrimônio
    inicial real e o DY mensal ponderado pelo peso de cada ativo.

    Retorna o mesmo formato de projetar_patrimonio() mais:
        "itens_carteira": lista com dados de cada ticker
            {ticker, cotas, preco, dy_mensal, valor_total,
             peso_pct, perfil_ml, compativel}
    """
    try:
        from modelo_fii import recomendar_fii

        itens_enriquecidos = []
        patrimonio_inicial  = 0.0

        for item in carteira:
            ticker = item["ticker"].strip().upper()
            cotas  = float(item["cotas"])

            preco, dy_mensal = _buscar_preco_e_dy(ticker)
            sleep(1.5)

            # fallback: sem dados ao vivo
            if preco is None:
                preco     = 0.0
                dy_mensal = PERFIS_DY.get(perfil_usuario, 0.09) / 12

            valor_total = preco * cotas
            patrimonio_inicial += valor_total

            # ML: perfil do FII
            rec = recomendar_fii(ticker)
            perfil_fii  = rec.get("Perfil_Recomendado", "Desconhecido") if rec.get("ok") else "Desconhecido"
            confianca   = rec.get("Confianca", "-")                      if rec.get("ok") else "-"
            compativel  = (perfil_fii == perfil_usuario)

            itens_enriquecidos.append({
                "ticker":      ticker,
                "cotas":       cotas,
                "preco":       preco,
                "dy_mensal":   dy_mensal,
                "valor_total": valor_total,
                "perfil_ml":   perfil_fii,
                "confianca":   confianca,
                "compativel":  compativel,
            })

        if patrimonio_inicial == 0:
            return {"ok": False, "erro": "Não foi possível obter preços para nenhum ticker."}

        # Peso de cada ativo e DY ponderado
        for item in itens_enriquecidos:
            item["peso_pct"] = round(item["valor_total"] / patrimonio_inicial * 100, 2)

        dy_ponderado = sum(
            i["dy_mensal"] * (i["valor_total"] / patrimonio_inicial)
            for i in itens_enriquecidos
        )

        # Valorização anual: média ponderada por perfil de cada FII
        _val_map = {"Conservador": 0.04, "Moderado": 0.07, "Arrojado": 0.10}
        valorizacao_anual = sum(
            _val_map.get(i["perfil_ml"], 0.07) * (i["valor_total"] / patrimonio_inicial)
            for i in itens_enriquecidos
        )

        # Simulação
        serie_fiis = _simular(
            patrimonio_inicial=patrimonio_inicial,
            aporte_mensal=aporte_mensal,
            anos=anos,
            dy_mensal=dy_ponderado,
            valorizacao_anual=valorizacao_anual,
            reinvestir=reinvestir_dividendos,
        )

        serie_poup = _simular_poupanca(
            patrimonio_inicial=patrimonio_inicial,
            aporte_mensal=aporte_mensal,
            anos=anos,
        )

        poup_dict = {s["ano"]: s["poupanca"] for s in serie_poup}
        serie_temporal = [
            {
                "ano":      s["ano"],
                "carteira": s["carteira"],
                "poupanca": poup_dict.get(s["ano"], 0),
            }
            for s in serie_fiis
        ]

        ultimo          = serie_fiis[-1]
        total_investido = patrimonio_inicial + aporte_mensal * anos * 12
        pat_final       = ultimo["carteira"]
        poup_final      = serie_poup[-1]["poupanca"]

        resumo = {
            "patrimonio_inicial":         round(patrimonio_inicial, 2),
            "patrimonio_final_carteira":  round(pat_final, 2),
            "total_investido":            round(total_investido, 2),
            "ganho_sobre_investido":      round(pat_final - total_investido, 2),
            "renda_passiva_mensal_final": round(ultimo["renda_passiva"], 2),
            "poupanca_final":             round(poup_final, 2),
            "vantagem_vs_poupanca":       round(pat_final - poup_final, 2),
            "dy_ponderado_mensal_pct":    round(dy_ponderado * 100, 4),
        }

        # Gráficos
        buf_comp  = _grafico_comparativo_carteira(serie_fiis, serie_poup)
        buf_pizza = _grafico_composicao_carteira(itens_enriquecidos)
        buf_renda = _grafico_renda_passiva(serie_fiis)

        # Texto resumo
        n_incompativeis = sum(1 for i in itens_enriquecidos if not i["compativel"])
        texto = _gerar_texto_carteira(
            perfil_usuario, anos, dy_ponderado,
            resumo, reinvestir_dividendos, n_incompativeis,
        )

        return {
            "ok":               True,
            "resumo":           resumo,
            "serie_temporal":   serie_temporal,
            "itens_carteira":   itens_enriquecidos,
            "carteira_por_classe": [
                {
                    "classe":               i["ticker"],
                    "peso_pct":             i["peso_pct"],
                    "dy_anual_pct":         round(((1 + i["dy_mensal"]) ** 12 - 1) * 100, 2),
                    "renda_mensal_estimada": round(i["valor_total"] * i["dy_mensal"], 2),
                }
                for i in itens_enriquecidos
            ],
            "texto_para_llm":  texto,
            "graficos": {
                "comparativo":   buf_comp,
                "composicao":    buf_pizza,
                "renda_passiva": buf_renda,
            },
        }

    except Exception as e:
        return {"ok": False, "erro": str(e)}


def _gerar_texto_carteira(
    perfil: str,
    anos: int,
    dy_ponderado: float,
    resumo: dict,
    reinvestir: bool,
    n_incompativeis: int,
) -> str:
    dy_anual_pct  = ((1 + dy_ponderado) ** 12 - 1) * 100
    pat_inicial   = resumo["patrimonio_inicial"]
    pat_final     = resumo["patrimonio_final_carteira"]
    total_inv     = resumo["total_investido"]
    ganho         = resumo["ganho_sobre_investido"]
    renda         = resumo["renda_passiva_mensal_final"]
    vantagem      = resumo["vantagem_vs_poupanca"]
    multiplicador = pat_final / total_inv if total_inv > 0 else 1
    reinv_texto   = "reinvestindo os dividendos" if reinvestir else "sacando os dividendos mensalmente"

    aviso_incomp = ""
    if n_incompativeis > 0:
        aviso_incomp = (
            f"<br><br>⚠️ <b>{n_incompativeis} ativo(s)</b> da sua carteira foram sinalizados como "
            f"incompatíveis com o seu perfil <b>{perfil}</b>. Avalie se deseja rebalancear "
            f"esses ativos para melhorar a aderência ao seu nível de risco."
        )

    return (
        f"Sua carteira atual possui um patrimônio inicial de <b>R$ {pat_inicial:,.2f}</b>, "
        f"com DY médio ponderado de <b>{dy_anual_pct:.2f}% ao ano</b> "
        f"({dy_ponderado*100:.3f}% ao mês) baseado nos dados reais de cada ativo.<br><br>"
        f"{reinv_texto.capitalize()} ao longo de <b>{anos} anos</b>, a projeção aponta para um "
        f"patrimônio de <b>R$ {pat_final:,.2f}</b> — multiplicando seu capital "
        f"<b>{multiplicador:.1f}×</b> sobre o total investido de <b>R$ {total_inv:,.2f}</b>.<br><br>"
        f"O ganho estimado é de <b>R$ {ganho:,.2f}</b>, gerando uma renda passiva mensal de "
        f"<b>R$ {renda:,.2f}/mês</b> ao final do período. "
        f"Comparado à poupança, sua carteira entregaria <b>R$ {vantagem:,.2f}</b> a mais."
        f"{aviso_incomp}"
    ).replace(",", "X").replace(".", ",").replace("X", ".")
