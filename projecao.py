"""
projecao.py
===========
Módulo de Projeções Financeiras — Integrante 3 (LORENZO)
Projeto Final de IA — PROJETO_FINAL_IA

Responsabilidades cobertas:
    1. Implementar a matemática de projeção patrimonial.
    2. Calcular rendimento por classe de ativo ao longo do tempo.
    3. Gerar gráficos comparativos (carteira recomendada vs. poupança).
    4. Calcular estimativa de renda passiva mensal futura do usuário.

Integração com o projeto:
    Este módulo consome o `dataset_fiis_b3_refinado.csv` (mesmo dataset usado
    pelo `ML_perfil_invest.py` do Miguel) para montar uma carteira recomendada
    a partir do `Perfil_Ideal_Investidor` previsto pelo modelo de Machine
    Learning. O DY (Dividend Yield) real de cada FII alimenta a estimativa de
    renda passiva, tornando a projeção coerente com a classificação da IA.

Fluxo típico de uso:
    >>> from projecao import ProjetorPatrimonial, ParametrosProjecao
    >>> proj = ProjetorPatrimonial.de_csv("dataset_fiis_b3_refinado.csv")
    >>> params = ParametrosProjecao(aporte_inicial=10_000, aporte_mensal=1_000, anos=20)
    >>> resultado = proj.projetar(perfil="Moderado", params=params)
    >>> resultado.resumo()
    >>> proj.gerar_grafico_comparativo(resultado, salvar_em="comparativo.png")

Autor: Lorenzo
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

# Matplotlib em backend não interativo para rodar em qualquer ambiente (servidor/CI).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("projecao")

# ---------------------------------------------------------------------------
# Estilo visual (paleta acessível, daltônico-friendly)
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (11, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)

COR_CARTEIRA = "#4C72B0"   # azul
COR_POUPANCA = "#DD8452"   # laranja
COR_APORTES = "#55A868"    # verde
PALETA_CLASSES = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# ===========================================================================
# 1. Parâmetros de entrada
# ===========================================================================
@dataclass(frozen=True)
class ParametrosProjecao:
    """Parâmetros que descrevem o plano de investimento do usuário.

    Atributos
    ---------
    aporte_inicial : float
        Capital inicial investido (R$).
    aporte_mensal : float
        Valor aportado todo mês (R$).
    anos : int
        Horizonte da projeção em anos.
    taxa_poupanca_aa : float
        Rentabilidade anual estimada da poupança (decimal). Default ~6% a.a.
    reinvestir_dividendos : bool
        Se True, os proventos são reinvestidos na própria carteira (juros
        compostos sobre os dividendos). Se False, são tratados como renda
        retirada (a carteira cresce só por aporte + valorização de cota).
    inflacao_aa : float
        Inflação anual estimada (decimal) para cálculo do valor real (opcional).
    """

    aporte_inicial: float = 0.0
    aporte_mensal: float = 0.0
    anos: int = 10
    taxa_poupanca_aa: float = 0.0617          # ~6,17% a.a. (referência histórica)
    reinvestir_dividendos: bool = True
    inflacao_aa: float = 0.0450               # ~4,5% a.a.

    def __post_init__(self) -> None:
        if self.aporte_inicial < 0 or self.aporte_mensal < 0:
            raise ValueError("Aportes não podem ser negativos.")
        if self.anos <= 0:
            raise ValueError("O horizonte (anos) deve ser positivo.")

    @property
    def meses(self) -> int:
        return self.anos * 12


# ===========================================================================
# 2. Resultado da projeção
# ===========================================================================
@dataclass
class ResultadoProjecao:
    """Empacota a saída de uma projeção, pronta para relatório ou gráfico."""

    perfil: str
    params: ParametrosProjecao
    dy_carteira_aa: float                      # dividend yield médio anual da carteira
    valorizacao_carteira_aa: float             # valorização de cota anual estimada
    composicao: pd.DataFrame                   # carteira por classe/segmento
    evolucao: pd.DataFrame                     # série mensal: aportes, carteira, poupança
    renda_passiva_mensal_final: float          # R$/mês no fim do horizonte
    total_investido: float
    patrimonio_final_carteira: float
    patrimonio_final_poupanca: float

    @property
    def patrimonio_real_carteira(self) -> float:
        """Patrimônio final descontada a inflação (poder de compra de hoje)."""
        fator = (1 + self.params.inflacao_aa) ** self.params.anos
        return self.patrimonio_final_carteira / fator

    def resumo(self) -> str:
        """Gera (e loga) um resumo textual amigável da projeção."""
        ganho = self.patrimonio_final_carteira - self.total_investido
        rent_total = (ganho / self.total_investido) if self.total_investido else 0.0
        vantagem = self.patrimonio_final_carteira - self.patrimonio_final_poupanca

        linhas = [
            "=" * 64,
            f"  PROJEÇÃO PATRIMONIAL — Perfil: {self.perfil.upper()}",
            "=" * 64,
            f"  Horizonte ................. {self.params.anos} anos ({self.params.meses} meses)",
            f"  Aporte inicial ............ {_brl(self.params.aporte_inicial)}",
            f"  Aporte mensal ............. {_brl(self.params.aporte_mensal)}",
            f"  Reinveste dividendos ...... {'Sim' if self.params.reinvestir_dividendos else 'Não'}",
            "-" * 64,
            f"  DY médio da carteira ...... {self.dy_carteira_aa:.2%} a.a.",
            f"  Valorização de cota ....... {self.valorizacao_carteira_aa:.2%} a.a.",
            "-" * 64,
            f"  Total investido ........... {_brl(self.total_investido)}",
            f"  Patrimônio final (carteira) {_brl(self.patrimonio_final_carteira)}",
            f"  Patrimônio final (poupança) {_brl(self.patrimonio_final_poupanca)}",
            f"  Ganho sobre o investido ... {_brl(ganho)} ({rent_total:.1%})",
            f"  Vantagem vs. poupança ..... {_brl(vantagem)}",
            f"  Valor real (hoje) ......... {_brl(self.patrimonio_real_carteira)}",
            "-" * 64,
            f"  RENDA PASSIVA MENSAL no fim do período: {_brl(self.renda_passiva_mensal_final)}/mês",
            "=" * 64,
        ]
        texto = "\n".join(linhas)
        logger.info("\n%s", texto)
        return texto


# ===========================================================================
# 3. Núcleo: o projetor patrimonial
# ===========================================================================
class ProjetorPatrimonial:
    """Motor de projeção patrimonial baseado no dataset de FIIs do projeto.

    A classe monta uma carteira recomendada por perfil (a partir dos FIIs cujo
    `Perfil_Ideal_Investidor` corresponde ao perfil informado), calcula o
    dividend yield médio e projeta a evolução do patrimônio mês a mês,
    comparando com a poupança.
    """

    # Premissas de valorização de cota (capital gain) por perfil, ao ano.
    # FIIs mais arrojados tendem a ter maior potencial de valorização e volatilidade.
    VALORIZACAO_POR_PERFIL: Mapping[str, float] = {
        "Conservador": 0.020,   # 2,0% a.a.
        "Moderado": 0.035,      # 3,5% a.a.
        "Arrojado": 0.050,      # 5,0% a.a.
    }

    COLUNAS_OBRIGATORIAS = {
        "Ticker",
        "Segmento",
        "DY_Anual",
        "Perfil_Ideal_Investidor",
    }

    def __init__(self, dados: pd.DataFrame) -> None:
        faltando = self.COLUNAS_OBRIGATORIAS - set(dados.columns)
        if faltando:
            raise ValueError(f"Dataset sem as colunas obrigatórias: {sorted(faltando)}")
        self.dados = dados.copy()
        # Saneamento: DY como número, não-negativo, sem NaN.
        self.dados["DY_Anual"] = (
            pd.to_numeric(self.dados["DY_Anual"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )

    # ----------------------------------------------------------------- IO
    @classmethod
    def de_csv(cls, caminho: str | Path) -> "ProjetorPatrimonial":
        """Carrega o projetor diretamente do CSV de FIIs do projeto."""
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {caminho}")
        df = pd.read_csv(caminho)
        logger.info("Dataset carregado: %s (%d FIIs)", caminho.name, len(df))
        return cls(df)

    # -------------------------------------------------- carteira por perfil
    def montar_carteira(self, perfil: str, dy_minimo: float = 0.0) -> pd.DataFrame:
        """Monta a carteira recomendada para um perfil.

        Filtra os FIIs do perfil, descarta os de DY zerado (sem distribuição) e
        agrega por `Segmento` (classe de ativo), atribuindo peso proporcional ao
        número de ativos de cada classe — uma alocação simples e diversificada.

        Retorna um DataFrame com: Segmento, qtd_fiis, DY médio da classe e peso.
        """
        perfil = self._normalizar_perfil(perfil)
        carteira = self.dados[self.dados["Perfil_Ideal_Investidor"] == perfil].copy()
        carteira = carteira[carteira["DY_Anual"] > dy_minimo]

        if carteira.empty:
            raise ValueError(
                f"Nenhum FII com DY > {dy_minimo:.1%} para o perfil '{perfil}'."
            )

        composicao = (
            carteira.groupby("Segmento")
            .agg(qtd_fiis=("Ticker", "count"), dy_medio_classe=("DY_Anual", "mean"))
            .reset_index()
        )
        composicao["peso"] = composicao["qtd_fiis"] / composicao["qtd_fiis"].sum()
        composicao = composicao.sort_values("peso", ascending=False).reset_index(drop=True)
        return composicao

    def dy_medio_carteira(self, composicao: pd.DataFrame) -> float:
        """DY anual da carteira = média ponderada do DY das classes pelo peso."""
        return float(np.average(composicao["dy_medio_classe"], weights=composicao["peso"]))

    # --------------------------------------------------------- projeção
    def projetar(self, perfil: str, params: ParametrosProjecao) -> ResultadoProjecao:
        """Executa a projeção patrimonial completa para um perfil.

        Modelo financeiro (composição mensal):
            patrimônio_t = patrimônio_{t-1} * (1 + r_valorização_mensal)
                                            * (1 + r_dividendo_mensal_reinvestido)
                         + aporte_mensal
        onde:
            - r_valorização_mensal deriva da valorização de cota anual do perfil;
            - r_dividendo_mensal deriva do DY anual da carteira; se
              `reinvestir_dividendos` for False, ele não compõe o patrimônio
              (vira renda retirada), mas continua sendo usado para estimar a
              renda passiva mensal.
        A poupança usa só a taxa da poupança, sem dividendos.
        """
        perfil = self._normalizar_perfil(perfil)
        composicao = self.montar_carteira(perfil)
        dy_aa = self.dy_medio_carteira(composicao)
        valorizacao_aa = self.VALORIZACAO_POR_PERFIL[perfil]

        # Conversão de taxas anuais -> mensais (equivalência composta).
        r_div_m = (1 + dy_aa) ** (1 / 12) - 1
        r_val_m = (1 + valorizacao_aa) ** (1 / 12) - 1
        r_poup_m = (1 + params.taxa_poupanca_aa) ** (1 / 12) - 1

        meses = params.meses
        idx = np.arange(meses + 1)  # inclui o mês 0 (estado inicial)

        invest_acum = np.zeros(meses + 1)
        carteira = np.zeros(meses + 1)
        poupanca = np.zeros(meses + 1)

        invest_acum[0] = params.aporte_inicial
        carteira[0] = params.aporte_inicial
        poupanca[0] = params.aporte_inicial

        # Taxa de crescimento mensal da carteira.
        if params.reinvestir_dividendos:
            r_cart_m = (1 + r_val_m) * (1 + r_div_m) - 1
        else:
            r_cart_m = r_val_m  # dividendos saem como renda, não compõem

        for t in range(1, meses + 1):
            invest_acum[t] = invest_acum[t - 1] + params.aporte_mensal
            carteira[t] = carteira[t - 1] * (1 + r_cart_m) + params.aporte_mensal
            poupanca[t] = poupanca[t - 1] * (1 + r_poup_m) + params.aporte_mensal

        evolucao = pd.DataFrame(
            {
                "mes": idx,
                "ano": idx / 12,
                "investido_acumulado": invest_acum,
                "carteira": carteira,
                "poupanca": poupanca,
            }
        )

        # Renda passiva mensal estimada no fim do horizonte:
        # patrimônio final * DY anual / 12.
        renda_passiva_final = carteira[-1] * dy_aa / 12

        resultado = ResultadoProjecao(
            perfil=perfil,
            params=params,
            dy_carteira_aa=dy_aa,
            valorizacao_carteira_aa=valorizacao_aa,
            composicao=composicao,
            evolucao=evolucao,
            renda_passiva_mensal_final=float(renda_passiva_final),
            total_investido=float(invest_acum[-1]),
            patrimonio_final_carteira=float(carteira[-1]),
            patrimonio_final_poupanca=float(poupanca[-1]),
        )
        return resultado

    # ----------------------------------------- rendimento por classe
    def rendimento_por_classe(
        self, perfil: str, params: ParametrosProjecao
    ) -> pd.DataFrame:
        """Calcula o rendimento (proventos) por classe de ativo ao longo do tempo.

        Aloca o patrimônio final da carteira entre as classes conforme os pesos
        e, para cada classe, estima os proventos anuais (patrimônio_classe *
        DY_classe). Útil para enxergar de onde vem a renda passiva.
        """
        resultado = self.projetar(perfil, params)
        comp = resultado.composicao.copy()
        patrimonio_final = resultado.patrimonio_final_carteira

        comp["patrimonio_alocado"] = comp["peso"] * patrimonio_final
        comp["proventos_anuais"] = comp["patrimonio_alocado"] * comp["dy_medio_classe"]
        comp["proventos_mensais"] = comp["proventos_anuais"] / 12
        return comp[
            [
                "Segmento",
                "peso",
                "dy_medio_classe",
                "patrimonio_alocado",
                "proventos_anuais",
                "proventos_mensais",
            ]
        ]

    # ============================================================ GRÁFICOS
    def gerar_grafico_comparativo(
        self, resultado: ResultadoProjecao, salvar_em: str | Path | None = None
    ) -> Path | None:
        """Gráfico de linha: carteira recomendada vs. poupança vs. aportes."""
        ev = resultado.evolucao
        fig, ax = plt.subplots(figsize=(11, 6))

        ax.plot(ev["ano"], ev["carteira"], color=COR_CARTEIRA, linewidth=2.5,
                label="Carteira recomendada")
        ax.plot(ev["ano"], ev["poupanca"], color=COR_POUPANCA, linewidth=2.0,
                linestyle="--", label="Poupança")
        ax.plot(ev["ano"], ev["investido_acumulado"], color=COR_APORTES,
                linewidth=1.5, linestyle=":", label="Total aportado")

        # Sombreia o ganho da carteira sobre a poupança.
        ax.fill_between(ev["ano"], ev["poupanca"], ev["carteira"],
                        where=ev["carteira"] >= ev["poupanca"],
                        color=COR_CARTEIRA, alpha=0.08)

        ax.set_title(
            f"Projeção patrimonial — perfil {resultado.perfil}: "
            f"carteira supera poupança em {_brl(resultado.patrimonio_final_carteira - resultado.patrimonio_final_poupanca)}"
        )
        ax.set_xlabel("Anos")
        ax.set_ylabel("Patrimônio acumulado (R$)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _brl(x, curto=True)))
        ax.legend(loc="upper left", frameon=True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.text(0.99, 0.01, "Fonte: dataset_fiis_b3_refinado.csv",
                 ha="right", va="bottom", fontsize=8, color="gray")

        return self._finalizar_figura(fig, salvar_em)

    def gerar_grafico_composicao(
        self, resultado: ResultadoProjecao, salvar_em: str | Path | None = None
    ) -> Path | None:
        """Gráfico de barras horizontais: peso de cada classe na carteira."""
        comp = resultado.composicao.sort_values("peso")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(comp))))

        cores = [PALETA_CLASSES[i % len(PALETA_CLASSES)] for i in range(len(comp))]
        barras = ax.barh(comp["Segmento"], comp["peso"] * 100, color=cores)
        for barra in barras:
            largura = barra.get_width()
            ax.text(largura + 0.4, barra.get_y() + barra.get_height() / 2,
                    f"{largura:.1f}%", va="center", fontsize=9)

        ax.set_title(f"Composição da carteira por classe de ativo — {resultado.perfil}")
        ax.set_xlabel("Peso na carteira (%)")
        ax.spines[["top", "right"]].set_visible(False)

        return self._finalizar_figura(fig, salvar_em)

    def gerar_grafico_renda_passiva(
        self, resultado: ResultadoProjecao, salvar_em: str | Path | None = None
    ) -> Path | None:
        """Gráfico de área: evolução da renda passiva mensal estimada ao longo do tempo."""
        ev = resultado.evolucao.copy()
        ev["renda_passiva_mensal"] = ev["carteira"] * resultado.dy_carteira_aa / 12

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.fill_between(ev["ano"], ev["renda_passiva_mensal"], color=COR_CARTEIRA, alpha=0.25)
        ax.plot(ev["ano"], ev["renda_passiva_mensal"], color=COR_CARTEIRA, linewidth=2.5)

        final = resultado.renda_passiva_mensal_final
        ax.annotate(
            f"{_brl(final)}/mês",
            xy=(ev["ano"].iloc[-1], final),
            xytext=(-10, 10), textcoords="offset points",
            ha="right", fontweight="bold", color=COR_CARTEIRA,
        )

        ax.set_title(f"Evolução da renda passiva mensal estimada — {resultado.perfil}")
        ax.set_xlabel("Anos")
        ax.set_ylabel("Renda passiva mensal (R$)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _brl(x, curto=True)))
        ax.spines[["top", "right"]].set_visible(False)

        return self._finalizar_figura(fig, salvar_em)

    # ------------------------------------------------------------- helpers
    def _normalizar_perfil(self, perfil: str) -> str:
        perfil = str(perfil).strip().capitalize()
        validos = set(self.dados["Perfil_Ideal_Investidor"].unique())
        if perfil not in validos:
            raise ValueError(
                f"Perfil '{perfil}' inválido. Use um de: {sorted(validos)}"
            )
        return perfil

    @staticmethod
    def _finalizar_figura(fig: plt.Figure, salvar_em: str | Path | None) -> Path | None:
        fig.tight_layout()
        if salvar_em is not None:
            destino = Path(salvar_em)
            fig.savefig(destino, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Gráfico salvo em: %s", destino)
            return destino
        plt.close(fig)
        return None


# ===========================================================================
# Utilidades
# ===========================================================================
def _brl(valor: float, curto: bool = False) -> str:
    """Formata um número como moeda brasileira (R$)."""
    if curto:
        if abs(valor) >= 1e6:
            return f"R$ {valor / 1e6:.1f}M"
        if abs(valor) >= 1e3:
            return f"R$ {valor / 1e3:.0f}k"
        return f"R$ {valor:.0f}"
    s = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


# ===========================================================================
# Demonstração / ponto de entrada
# ===========================================================================
def _demo() -> None:
    """Executa uma demonstração ponta-a-ponta com o dataset do projeto."""
    proj = ProjetorPatrimonial.de_csv("dataset_fiis_b3_refinado.csv")

    params = ParametrosProjecao(
        aporte_inicial=10_000,
        aporte_mensal=1_000,
        anos=20,
        reinvestir_dividendos=True,
    )

    for perfil in ["Conservador", "Moderado", "Arrojado"]:
        resultado = proj.projetar(perfil, params)
        resultado.resumo()

        print("\n  Rendimento por classe de ativo (renda mensal estimada):")
        rend = proj.rendimento_por_classe(perfil, params)
        for _, linha in rend.iterrows():
            print(
                f"    - {linha['Segmento']:<22} peso {linha['peso']:.1%}  "
                f"DY {linha['dy_medio_classe']:.2%}  "
                f"=> {_brl(linha['proventos_mensais'])}/mês"
            )

        slug = perfil.lower()
        proj.gerar_grafico_comparativo(resultado, f"projecao_comparativo_{slug}.png")
        proj.gerar_grafico_composicao(resultado, f"projecao_composicao_{slug}.png")
        proj.gerar_grafico_renda_passiva(resultado, f"projecao_renda_{slug}.png")


if __name__ == "__main__":
    _demo()
