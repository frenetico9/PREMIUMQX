import os
import yfinance as yf
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import csv
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# COLETA DE DADOS
def fetch_asset_data(symbol="EURUSD=X", interval="1m", period="7d"):
    df = yf.download(tickers=symbol, interval=interval, period=period)
    df.dropna(inplace=True)
    return df

# TREINAMENTO
def train_model(df):
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    X = df[['Open', 'High', 'Low', 'Close']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Modelo treinado. Acurácia: {acc:.2%}")
    return model

# AJUSTE DINÂMICO DE CONFIANÇA
def confianca_dinamica(csv_file="historico_trades.csv", janela=5, base=0.6):
    try:
        df = pd.read_csv(csv_file)
        ultimos = df.tail(janela)
        acertos = ultimos["resultado"].apply(lambda x: x == "win").sum()
        taxa = acertos / len(ultimos)

        print(f"📊 Taxa de acerto nos últimos {janela} trades: {taxa:.2%}")

        if taxa >= 0.8:
            return base - 0.05
        elif taxa <= 0.4:
            return base + 0.1
        else:
            return base
    except:
        return base

# PREVISÃO COM FILTRO - usa modelo treinado direto
def predict_next(df, min_confidence=0.6):
    model = train_model(df)
    latest = df.iloc[-1:][['Open', 'High', 'Low', 'Close']].values
    prob = model.predict_proba(latest)[0]
    decision = "UP" if prob[1] > 0.5 else "DOWN"
    confidence = prob[1] if decision == "UP" else 1 - prob[1]

    print(f"🔮 Previsão: {decision} (Confiança: {confidence:.2%})")

    if confidence < min_confidence:
        print("⚠️ Confiança abaixo do mínimo. Operação cancelada.")
        return None, confidence
    return decision, confidence

# GRÁFICO DE RESULTADOS
def plot_historico_trades(csv_file="historico_trades.csv"):
    if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
        print(f"⚠️ Arquivo '{csv_file}' não encontrado ou está vazio.")
        return

    try:
        df = pd.read_csv(csv_file)
        df["data_hora"] = pd.to_datetime(df["data"] + " " + df["hora"])
        df["acumulado"] = df["entrada"].where(df["resultado"] == "win", -df["entrada"]).cumsum()

        plt.figure(figsize=(10, 5))
        plt.plot(df["data_hora"], df["acumulado"], marker='o', color='blue')
        plt.xticks(rotation=45)
        plt.title("Lucro/Prejuízo Acumulado")
        plt.xlabel("Data/Hora")
        plt.ylabel("Saldo ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("grafico_resultado.png")
        print("📈 Gráfico salvo como 'grafico_resultado.png'")
    except Exception as e:
        print(f"❌ Erro ao processar o histórico: {e}")

# ENTRADAS
try:
    with open("input.txt", "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    email = lines[0]
    password = lines[1]
    stop_win = float(lines[2])
    stop_loss = float(lines[3])
    fixed_amount = float(lines[4])
except Exception as e:
    st.error(f"Erro ao ler o arquivo input.txt: {e}")
    st.stop()

# EXECUÇÃO PRINCIPAL
print("🚀 Iniciando pipeline completo com IA + ajustes + gráfico")

df = fetch_asset_data()
min_conf = confianca_dinamica()
decision, confidence = predict_next(df, min_confidence=min_conf)

if decision:
    print(f"Simulação: Direção={decision}, Confiança={confidence:.2%}")
    plot_historico_trades()
else:
    print("⏹️ Nenhuma operação realizada. Gráfico não gerado.")
