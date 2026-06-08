import pandas as pd
import numpy as np

from sklearn.model_selection import ( train_test_split, cross_val_score )

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

import matplotlib.pyplot as plt
import joblib


# Carregando arquivo CSV

df = pd.read_csv("dataset_fiis_b3_refinado.csv")

print("\nDataset carregado:")
print(df.shape)


# Preparação dos dados de teste e treino
df = df.set_index("Ticker")

TARGET = "Perfil_Ideal_Investidor"

X = df.drop(columns=[TARGET, "Perfil_Risco", "Score_Risco"])

y = df[TARGET]

# One Hot Encoding do Segmento

X = pd.get_dummies(X, columns=["Segmento"], drop_first=True)

print("\nQuantidade de atributos:")
print(X.shape[1])

# Treino / Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

#Gerando o modelo com Random Forest
modelo = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=4 )

modelo.fit(X_train, y_train)

#Orevisões
y_pred = modelo.predict(X_test)

#extraindo as métricas de desempenho do modelo
print("\n" + "="*60)
print("DESEMPENHO DO MODELO")
print("="*60)

acuracia = accuracy_score(y_test, y_pred)

print(f"\nAcurácia: {acuracia:.2%}")

print("\nRelatório:")
print(classification_report(y_test, y_pred))

#Cross validation
scores = cross_val_score( modelo, X, y, cv=5 )

print("\nValidação Cruzada (5-Fold):")

print(scores)

print(
    f"\nMédia CV: {scores.mean():.2%}"
)

#importancia das features para a decisão do modelo
importancias = pd.DataFrame({"Feature": X.columns,"Importancia": modelo.feature_importances_})

importancias = (importancias.sort_values(by="Importancia",ascending=False))

print("\nTOP 15 VARIÁVEIS MAIS IMPORTANTES")

print(importancias.head(15))


#Função de recomendação de perfil de investidor para um FII específico, com base no modelo treinado

def recomendar_fii(ticker):

    ticker = ticker.upper()

    if ticker not in X.index:
        return "Ticker não encontrado."

    dados = X.loc[[ticker]]

    previsao = modelo.predict(dados)[0]

    probabilidade = modelo.predict_proba(dados)

    confianca = (np.max(probabilidade) * 100)

    return { "Ticker": ticker, "Perfil_Recomendado": previsao, "Confianca": f"{confianca:.2f}%"}

#Exemplo
print(
    recomendar_fii("HGLG11")
)
