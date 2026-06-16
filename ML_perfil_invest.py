import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

# ==========================================================
# CAMINHOS
# ==========================================================

_DIR      = os.path.dirname(os.path.abspath(__file__))
_CSV      = os.path.join(_DIR, "dataset_fiis_b3_refinado.csv")
_PKL      = os.path.join(_DIR, "modelo_xgb.pkl")
_COLS_PKL = os.path.join(_DIR, "modelo_colunas.pkl")

# ==========================================================
# CARREGAMENTO
# ==========================================================

df = pd.read_csv(_CSV)

print("\nDataset carregado:")
print(df.shape)

df = df.set_index("Ticker")

# ==========================================================
# WINSORIZAÇÃO P/VP
# ==========================================================

print("\nAplicando winsorização no P/VP...")

df["P_VP"] = df["P_VP"].clip(
    lower=0,
    upper=5
)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

df["Log_Patrimonio"] = np.log1p(df["Patrimonio"])

df["Log_Liquidez"] = np.log1p(df["Liquidez_Diaria"])

df["Liquidez_Patrimonio"] = (
    df["Liquidez_Diaria"] /
    (df["Patrimonio"] + 1)
)

df["Indice_Valor"] = (
    df["DY_Anual"] /
    (df["P_VP"] + 0.01)
)

df["Desconto_VP"] = (
    1 - df["P_VP"]
)

df["DY_Ajustado"] = (
    df["DY_Anual"] *
    (1 - df["Vacancia"])
)

# ==========================================================
# PREPARAÇÃO DOS DADOS
# ==========================================================

TARGET = "Perfil_Ideal_Investidor"

X = df.drop(
    columns=[
        TARGET,
        "Perfil_Risco",
        "Score_Risco"
    ]
)

y = df[TARGET]

# ==========================================================
# TRANSFORMAR CLASSES EM NÚMEROS
# ==========================================================

mapa_classes = {
    "Conservador": 0,
    "Moderado": 1,
    "Arrojado": 2
}

mapa_reverso = {
    0: "Conservador",
    1: "Moderado",
    2: "Arrojado"
}

y_numerico = y.map(mapa_classes)

# ==========================================================
# ONE HOT ENCODING
# ==========================================================

X = pd.get_dummies(
    X,
    columns=["Segmento"],
    drop_first=True
)

print("\nQuantidade de atributos:")
print(X.shape[1])

print("\nDistribuição das classes:")
print(y.value_counts())

# ==========================================================
# TREINO / TESTE
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_numerico,
    test_size=0.20,
    random_state=42,
    stratify=y_numerico
)

# ==========================================================
# XGBOOST — treina ou carrega modelo salvo
# ==========================================================

if os.path.exists(_PKL) and os.path.exists(_COLS_PKL):

    print("\nModelo salvo encontrado. Carregando...")

    modelo  = joblib.load(_PKL)
    colunas = joblib.load(_COLS_PKL)
    X       = X.reindex(columns=colunas, fill_value=0)

    print("Modelo carregado com sucesso.")

else:

    print("\nNenhum modelo salvo. Treinando XGBoost...")

    modelo = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0
    )

    modelo.fit(X_train, y_train)

    joblib.dump(modelo, _PKL)
    joblib.dump(X.columns.tolist(), _COLS_PKL)

    print("Modelo treinado e salvo com sucesso.")

# ==========================================================
# PREVISÕES
# ==========================================================

y_pred = modelo.predict(X_test)

# ==========================================================
# MÉTRICAS
# ==========================================================

print("\n" + "="*60)
print("DESEMPENHO DO MODELO XGBOOST")
print("="*60)

acuracia = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAcurácia: {acuracia:.2%}")

print("\nRelatório:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Conservador",
            "Moderado",
            "Arrojado"
        ]
    )
)

# ==========================================================
# CROSS VALIDATION
# ==========================================================

scores = cross_val_score(
    modelo,
    X,
    y_numerico,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

print("\nValidação Cruzada (5-Fold):")
print(scores)

print(
    f"\nMédia CV: {scores.mean():.2%}"
)

# ==========================================================
# MATRIZ DE CONFUSÃO
# ==========================================================
#
#
#cm = confusion_matrix(
#    y_test,
#    y_pred
#)
#
#disp = ConfusionMatrixDisplay(
#    confusion_matrix=cm,
#    display_labels=[
#        "Conservador",
#        "Moderado",
#        "Arrojado"
#    ]
#)
#
#disp.plot()
#
#plt.title("Matriz de Confusão - XGBoost")
#plt.show()
#
# ==========================================================
# IMPORTÂNCIA DAS FEATURES
# ==========================================================

importancias = pd.DataFrame({
    "Feature": X.columns,
    "Importancia": modelo.feature_importances_
})

importancias = importancias.sort_values(
    by="Importancia",
    ascending=False
)

print("\nTOP 20 VARIÁVEIS MAIS IMPORTANTES")

print(importancias.head(20))

# ==========================================================
# DY MÉDIO POR PERFIL
# (usado pelo projecao.py como fallback quando o
#  Status Invest não responder para algum ticker)
# ==========================================================

PERFIS_DY: dict = (
    df
    .groupby("Perfil_Ideal_Investidor")["DY_Anual"]
    .mean()
    .to_dict()
)

# ==========================================================
# RECOMENDAÇÃO — função pública usada pelo app.py
# ==========================================================

def recomendar_fii(ticker):

    ticker = ticker.upper()

    if ticker not in X.index:
        return {
            "ok":   False,
            "erro": f"Ticker '{ticker}' não encontrado no dataset."
        }

    dados = X.loc[[ticker]].reindex(columns=X.columns, fill_value=0)

    previsao = modelo.predict(dados)[0]

    probabilidade = modelo.predict_proba(dados)

    confianca = np.max(probabilidade) * 100

    return {
        "ok":                 True,
        "Ticker":             ticker,
        "Perfil_Recomendado": mapa_reverso[int(previsao)],
        "Confianca":          f"{confianca:.2f}%"
    }
