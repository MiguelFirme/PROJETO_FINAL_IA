
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend sem tela: funciona em servidor/agente/CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("projecao")

# Caminho padrao do dataset (o mesmo usado pelo ML do Miguel).
DATASET_PADRAO = "dataset_fiis_b3_refinado.csv"

# Onde os graficos sao salvos por padrao.
DIR_GRAFICOS_PADRAO = "graficos_projecao"

# --------------------------------------------------------------------------
# Estilo visual dos graficos
# --------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (11, 6), "figure.dpi": 150, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "legend.fontsize": 10,
})
COR_CARTEIRA, COR_POUPANCA, COR_APORTES = "#4C72B0", "#DD8452", "#55A868"
PALETA = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# ==========================================================================
#  FUNCAO-FERRAMENTA  —  E ISTO QUE A LLM CHAMA
# ==========================================================================
def projetar_patrimonio(
    perfil: str,
    aporte_inicial: float,
    aporte_mensal: float,
    anos: int,
    reinvestir_dividendos: bool = True,
    taxa_poupanca_aa: float = 0.0617,
    inflacao_aa: float = 0.0450,
    gerar_graficos: bool = True,
    dataset: str = DATASET_PADRAO,
    dir_saida: str = DIR_GRAFICOS_PADRAO,
) -> dict[str, Any]:

    try:
        projetor = ProjetorPatrimonial.de_csv(dataset)
        params = ParametrosProjecao(
            aporte_inicial=float(aporte_inicial),
            aporte_mensal=float(aporte_mensal),
            anos=int(anos),
            reinvestir_dividendos=bool(reinvestir_dividendos),
            taxa_poupanca_aa=float(taxa_poupanca_aa),
            inflacao_aa=float(inflacao_aa),
        )
        resultado = projetor.projetar(perfil, params)

        graficos: dict[str, str] = {}
        if gerar_graficos:
            graficos = projetor.gerar_todos_os_graficos(resultado, dir_saida)

        return _montar_saida_json(resultado, projetor, graficos)

    except (ValueError, FileNotFoundError) as exc:
        logger.error("Falha na projecao: %s", exc)
        return {"ok": False, "erro": str(exc)}


def descrever_ferramenta() -> dict[str, Any]:
 
    return {
        "name": "projetar_patrimonio",
        "description": (
            "Projeta o patrimonio futuro e a renda passiva mensal de um "
            "investidor em FIIs, comparando com a poupanca. Use quando o "
            "usuario quiser saber quanto tera acumulado ou quanto de renda "
            "passiva tera no futuro, dado seu perfil, aporte inicial, aporte "
            "mensal e horizonte em anos."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "perfil": {
                    "type": "string",
                    "enum": ["Conservador", "Moderado", "Arrojado"],
                    "description": "Perfil do investidor, geralmente vindo do modelo de ML.",
                },
                "aporte_inicial": {"type": "number", "description": "Capital inicial em R$."},
                "aporte_mensal": {"type": "number", "description": "Aporte mensal em R$."},
                "anos": {"type": "integer", "description": "Horizonte em anos."},
                "reinvestir_dividendos": {
                    "type": "boolean",
                    "description": "True para acumular; False para viver de renda.",
                },
            },
            "required": ["perfil", "aporte_inicial", "aporte_mensal", "anos"],
        },
    }


# ==========================================================================
#  Montagem da saida JSON (numeros + texto + graficos)
# ==========================================================================
def _montar_saida_json(
    resultado: "ResultadoProjecao",
    projetor: "ProjetorPatrimonial",
    graficos: Mapping[str, str],
) -> dict[str, Any]:
    """Converte o resultado interno em um dicionario pronto para a LLM."""
    p = resultado.params
    ganho = resultado.patrimonio_final_carteira - resultado.total_investido
    vantagem = resultado.patrimonio_final_carteira - resultado.patrimonio_final_poupanca

    rendimento = projetor.rendimento_por_classe_df(resultado)
    classes = [
        {
            "classe": row["Segmento"],
            "peso_pct": round(row["peso"] * 100, 2),
            "dy_anual_pct": round(row["dy_medio_classe"] * 100, 2),
            "renda_mensal_estimada": round(row["proventos_mensais"], 2),
        }
        for _, row in rendimento.iterrows()
    ]

    return {
        "ok": True,
        "perfil": resultado.perfil,
        "parametros": {
            "aporte_inicial": p.aporte_inicial,
            "aporte_mensal": p.aporte_mensal,
            "anos": p.anos,
            "reinvestir_dividendos": p.reinvestir_dividendos,
        },
        "resumo": {
            "dy_carteira_anual_pct": round(resultado.dy_carteira_aa * 100, 2),
            "valorizacao_cota_anual_pct": round(resultado.valorizacao_carteira_aa * 100, 2),
            "total_investido": round(resultado.total_investido, 2),
            "patrimonio_final_carteira": round(resultado.patrimonio_final_carteira, 2),
            "patrimonio_final_poupanca": round(resultado.patrimonio_final_poupanca, 2),
            "patrimonio_real_hoje": round(resultado.patrimonio_real_carteira, 2),
            "ganho_sobre_investido": round(ganho, 2),
            "vantagem_vs_poupanca": round(vantagem, 2),
            "renda_passiva_mensal_final": round(resultado.renda_passiva_mensal_final, 2),
        },
        "carteira_por_classe": classes,
        "texto_para_llm": resultado.texto_narrativo(),
        "graficos": dict(graficos),
        "serie_temporal": resultado.serie_resumida(),  # alguns pontos p/ contexto
    }


# ==========================================================================
#  Entrada: parametros da projecao
# ==========================================================================
@dataclass(frozen=True)
class ParametrosProjecao:
    aporte_inicial: float = 0.0
    aporte_mensal: float = 0.0
    anos: int = 10
    taxa_poupanca_aa: float = 0.0617
    reinvestir_dividendos: bool = True
    inflacao_aa: float = 0.0450

    def __post_init__(self) -> None:
        if self.aporte_inicial < 0 or self.aporte_mensal < 0:
            raise ValueError("Aportes nao podem ser negativos.")
        if self.anos <= 0:
            raise ValueError("O horizonte (anos) deve ser positivo.")

    @property
    def meses(self) -> int:
        return self.anos * 12


# ==========================================================================
#  Saida interna: resultado da projecao
# ==========================================================================
@dataclass
class ResultadoProjecao:
    perfil: str
    params: ParametrosProjecao
    dy_carteira_aa: float
    valorizacao_carteira_aa: float
    composicao: pd.DataFrame
    evolucao: pd.DataFrame
    renda_passiva_mensal_final: float
    total_investido: float
    patrimonio_final_carteira: float
    patrimonio_final_poupanca: float

    @property
    def patrimonio_real_carteira(self) -> float:
        fator = (1 + self.params.inflacao_aa) ** self.params.anos
        return self.patrimonio_final_carteira / fator

    def texto_narrativo(self) -> str:
        """Texto pronto para a LLM narrar ao usuario (linguagem natural)."""
        ganho = self.patrimonio_final_carteira - self.total_investido
        vantagem = self.patrimonio_final_carteira - self.patrimonio_final_poupanca
        return (
            f"Com perfil {self.perfil}, aportando {_brl(self.params.aporte_inicial)} "
            f"inicialmente e {_brl(self.params.aporte_mensal)} por mes durante "
            f"{self.params.anos} anos, o patrimonio projetado e de "
            f"{_brl(self.patrimonio_final_carteira)}. Desse total, "
            f"{_brl(self.total_investido)} foram aportados e {_brl(ganho)} vieram "
            f"de rendimentos. Na poupanca, o mesmo plano renderia apenas "
            f"{_brl(self.patrimonio_final_poupanca)} — uma vantagem de "
            f"{_brl(vantagem)} para a carteira de FIIs. Ao fim do periodo, a "
            f"renda passiva mensal estimada seria de "
            f"{_brl(self.renda_passiva_mensal_final)} por mes, considerando um "
            f"dividend yield medio de {self.dy_carteira_aa:.2%} ao ano. Em valor "
            f"de hoje (descontada a inflacao), o patrimonio equivale a "
            f"{_brl(self.patrimonio_real_carteira)}."
        )

    def serie_resumida(self, pontos: int = 6) -> list[dict[str, float]]:
        """Amostra alguns pontos da serie temporal (para a LLM ter contexto)."""
        ev = self.evolucao
        indices = np.linspace(0, len(ev) - 1, pontos).astype(int)
        return [
            {
                "ano": round(float(ev.iloc[i]["ano"]), 1),
                "carteira": round(float(ev.iloc[i]["carteira"]), 2),
                "poupanca": round(float(ev.iloc[i]["poupanca"]), 2),
            }
            for i in indices
        ]


# ==========================================================================
#  Motor de projecao
# ==========================================================================
class ProjetorPatrimonial:
    VALORIZACAO_POR_PERFIL: Mapping[str, float] = {
        "Conservador": 0.020, "Moderado": 0.035, "Arrojado": 0.050,
    }
    COLUNAS_OBRIGATORIAS = {"Ticker", "Segmento", "DY_Anual", "Perfil_Ideal_Investidor"}

    def __init__(self, dados: pd.DataFrame) -> None:
        faltando = self.COLUNAS_OBRIGATORIAS - set(dados.columns)
        if faltando:
            raise ValueError(f"Dataset sem as colunas obrigatorias: {sorted(faltando)}")
        self.dados = dados.copy()
        self.dados["DY_Anual"] = (
            pd.to_numeric(self.dados["DY_Anual"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )

    @classmethod
    def de_csv(cls, caminho: str | Path) -> "ProjetorPatrimonial":
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(f"Dataset nao encontrado: {caminho}")
        return cls(pd.read_csv(caminho))

    def montar_carteira(self, perfil: str, dy_minimo: float = 0.0) -> pd.DataFrame:
        perfil = self._normalizar_perfil(perfil)
        carteira = self.dados[self.dados["Perfil_Ideal_Investidor"] == perfil].copy()
        carteira = carteira[carteira["DY_Anual"] > dy_minimo]
        if carteira.empty:
            raise ValueError(f"Nenhum FII com DY > {dy_minimo:.1%} para o perfil '{perfil}'.")
        composicao = (
            carteira.groupby("Segmento")
            .agg(qtd_fiis=("Ticker", "count"), dy_medio_classe=("DY_Anual", "mean"))
            .reset_index()
        )
        composicao["peso"] = composicao["qtd_fiis"] / composicao["qtd_fiis"].sum()
        return composicao.sort_values("peso", ascending=False).reset_index(drop=True)

    def dy_medio_carteira(self, composicao: pd.DataFrame) -> float:
        return float(np.average(composicao["dy_medio_classe"], weights=composicao["peso"]))

    def projetar(self, perfil: str, params: ParametrosProjecao) -> ResultadoProjecao:
        perfil = self._normalizar_perfil(perfil)
        composicao = self.montar_carteira(perfil)
        dy_aa = self.dy_medio_carteira(composicao)
        valorizacao_aa = self.VALORIZACAO_POR_PERFIL[perfil]

        r_div_m = (1 + dy_aa) ** (1 / 12) - 1
        r_val_m = (1 + valorizacao_aa) ** (1 / 12) - 1
        r_poup_m = (1 + params.taxa_poupanca_aa) ** (1 / 12) - 1

        meses = params.meses
        idx = np.arange(meses + 1)
        invest_acum = np.zeros(meses + 1)
        carteira = np.zeros(meses + 1)
        poupanca = np.zeros(meses + 1)
        invest_acum[0] = carteira[0] = poupanca[0] = params.aporte_inicial

        r_cart_m = (1 + r_val_m) * (1 + r_div_m) - 1 if params.reinvestir_dividendos else r_val_m

        for t in range(1, meses + 1):
            invest_acum[t] = invest_acum[t - 1] + params.aporte_mensal
            carteira[t] = carteira[t - 1] * (1 + r_cart_m) + params.aporte_mensal
            poupanca[t] = poupanca[t - 1] * (1 + r_poup_m) + params.aporte_mensal

        evolucao = pd.DataFrame({
            "mes": idx, "ano": idx / 12,
            "investido_acumulado": invest_acum, "carteira": carteira, "poupanca": poupanca,
        })
        renda_passiva_final = carteira[-1] * dy_aa / 12

        return ResultadoProjecao(
            perfil=perfil, params=params, dy_carteira_aa=dy_aa,
            valorizacao_carteira_aa=valorizacao_aa, composicao=composicao,
            evolucao=evolucao, renda_passiva_mensal_final=float(renda_passiva_final),
            total_investido=float(invest_acum[-1]),
            patrimonio_final_carteira=float(carteira[-1]),
            patrimonio_final_poupanca=float(poupanca[-1]),
        )

    def rendimento_por_classe_df(self, resultado: ResultadoProjecao) -> pd.DataFrame:
        comp = resultado.composicao.copy()
        comp["patrimonio_alocado"] = comp["peso"] * resultado.patrimonio_final_carteira
        comp["proventos_anuais"] = comp["patrimonio_alocado"] * comp["dy_medio_classe"]
        comp["proventos_mensais"] = comp["proventos_anuais"] / 12
        return comp

    # ---------------------------------------------------------- graficos
    def gerar_todos_os_graficos(
        self, resultado: ResultadoProjecao, dir_saida: str | Path
    ) -> dict[str, str]:
        """Gera os 3 graficos e devolve {nome: caminho} para a LLM referenciar."""
        pasta = Path(dir_saida)
        pasta.mkdir(parents=True, exist_ok=True)
        slug = resultado.perfil.lower()
        caminhos = {
            "comparativo": pasta / f"comparativo_{slug}.png",
            "composicao": pasta / f"composicao_{slug}.png",
            "renda_passiva": pasta / f"renda_passiva_{slug}.png",
        }
        self._grafico_comparativo(resultado, caminhos["comparativo"])
        self._grafico_composicao(resultado, caminhos["composicao"])
        self._grafico_renda(resultado, caminhos["renda_passiva"])
        return {k: str(v) for k, v in caminhos.items()}

    def _grafico_comparativo(self, resultado: ResultadoProjecao, destino: Path) -> None:
        ev = resultado.evolucao
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(ev["ano"], ev["carteira"], color=COR_CARTEIRA, linewidth=2.5, label="Carteira recomendada")
        ax.plot(ev["ano"], ev["poupanca"], color=COR_POUPANCA, linewidth=2.0, linestyle="--", label="Poupanca")
        ax.plot(ev["ano"], ev["investido_acumulado"], color=COR_APORTES, linewidth=1.5, linestyle=":", label="Total aportado")
        ax.fill_between(ev["ano"], ev["poupanca"], ev["carteira"],
                        where=ev["carteira"] >= ev["poupanca"], color=COR_CARTEIRA, alpha=0.08)
        ax.set_title(f"Projecao patrimonial — perfil {resultado.perfil}: vantagem de "
                     f"{_brl(resultado.patrimonio_final_carteira - resultado.patrimonio_final_poupanca)} sobre a poupanca")
        ax.set_xlabel("Anos"); ax.set_ylabel("Patrimonio acumulado (R$)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _brl(x, curto=True)))
        ax.legend(loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
        self._salvar(fig, destino)

    def _grafico_composicao(self, resultado: ResultadoProjecao, destino: Path) -> None:
        comp = resultado.composicao.sort_values("peso")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(comp))))
        cores = [PALETA[i % len(PALETA)] for i in range(len(comp))]
        barras = ax.barh(comp["Segmento"], comp["peso"] * 100, color=cores)
        for b in barras:
            ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=9)
        ax.set_title(f"Composicao da carteira por classe de ativo — {resultado.perfil}")
        ax.set_xlabel("Peso na carteira (%)"); ax.spines[["top", "right"]].set_visible(False)
        self._salvar(fig, destino)

    def _grafico_renda(self, resultado: ResultadoProjecao, destino: Path) -> None:
        ev = resultado.evolucao.copy()
        ev["renda"] = ev["carteira"] * resultado.dy_carteira_aa / 12
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.fill_between(ev["ano"], ev["renda"], color=COR_CARTEIRA, alpha=0.25)
        ax.plot(ev["ano"], ev["renda"], color=COR_CARTEIRA, linewidth=2.5)
        ax.annotate(f"{_brl(resultado.renda_passiva_mensal_final)}/mes",
                    xy=(ev["ano"].iloc[-1], resultado.renda_passiva_mensal_final),
                    xytext=(-10, 10), textcoords="offset points", ha="right",
                    fontweight="bold", color=COR_CARTEIRA)
        ax.set_title(f"Evolucao da renda passiva mensal estimada — {resultado.perfil}")
        ax.set_xlabel("Anos"); ax.set_ylabel("Renda passiva mensal (R$)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _brl(x, curto=True)))
        ax.spines[["top", "right"]].set_visible(False)
        self._salvar(fig, destino)

    @staticmethod
    def _salvar(fig: plt.Figure, destino: Path) -> None:
        fig.tight_layout()
        fig.savefig(destino, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Grafico salvo: %s", destino)

    def _normalizar_perfil(self, perfil: str) -> str:
        perfil = str(perfil).strip().capitalize()
        validos = set(self.dados["Perfil_Ideal_Investidor"].unique())
        if perfil not in validos:
            raise ValueError(f"Perfil '{perfil}' invalido. Use um de: {sorted(validos)}")
        return perfil


# ==========================================================================
#  Utilidades
# ==========================================================================
def _brl(valor: float, curto: bool = False) -> str:
    if curto:
        if abs(valor) >= 1e6: return f"R$ {valor / 1e6:.1f}M"
        if abs(valor) >= 1e3: return f"R$ {valor / 1e3:.0f}k"
        return f"R$ {valor:.0f}"
    s = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


# ==========================================================================
#  Demonstracao: simula a LLM chamando a ferramenta
# ==========================================================================
def _demo() -> None:
    print(">>> A LLM extraiu da conversa: perfil=Moderado, aporte_inicial=10000, "
          "aporte_mensal=1000, anos=20")
    print(">>> Chamando a ferramenta projetar_patrimonio(...)\n")

    resposta = projetar_patrimonio(
        perfil="Moderado", aporte_inicial=10_000, aporte_mensal=1_000, anos=20,
        reinvestir_dividendos=True, gerar_graficos=True,
    )

    print("=== SAIDA JSON (o que a LLM recebe de volta) ===")
    print(json.dumps(resposta, ensure_ascii=False, indent=2))

    if resposta["ok"]:
        print("\n=== TEXTO QUE A LLM NARRARIA AO USUARIO ===")
        print(resposta["texto_para_llm"])


if __name__ == "__main__":
    _demo()
