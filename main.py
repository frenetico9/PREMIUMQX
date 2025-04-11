# IMPORTS
import yfinance as yf
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import csv
import time
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# COLETA DE DADOS
def fetch_asset_data(symbol="EURUSD=X", interval="1m", period="7d"):
    df = yf.download(tickers=symbol, interval=interval, period=period)
    df.dropna(inplace=True)
    return df

# TREINAMENTO
def train_model(df, model_path="modelo_rf.joblib"):
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    X = df[['Open', 'High', 'Low', 'Close']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Modelo treinado. Acurácia: {acc:.2%}")
    joblib.dump(model, model_path)
    return model

# AJUSTE DINÂMICO DE CONFIANÇA
def confianca_dinamica(csv_file="historico_trades.csv", janela=5, base=0.6):
    try:
        df = pd.read_csv(csv_file)
        ultimos = df.tail(janela)
        acertos = ultimos["resultado"].apply(lambda x: x > 0).sum()
        taxa = acertos / len(ultimos)

        print(f"📊 Taxa de acerto nos últimos {janela} trades: {taxa:.2%}")

        if taxa >= 0.8:
            nova = base - 0.05
            print(f"⬇️ Reduzindo confiança mínima para {nova:.2f}")
            return nova
        elif taxa <= 0.4:
            nova = base + 0.1
            print(f"⬆️ Aumentando confiança mínima para {nova:.2f}")
            return nova
        else:
            print(f"➡️ Mantendo confiança mínima em {base:.2f}")
            return base
    except:
        return base

# PREVISÃO COM FILTRO
def predict_next(df, model_path="modelo_rf.joblib", min_confidence=0.6):
    model = joblib.load(model_path)
    latest = df.iloc[-1:][['Open', 'High', 'Low', 'Close']].values
    prob = model.predict_proba(latest)[0]
    decision = "UP" if prob[1] > 0.5 else "DOWN"
    confidence = prob[1] if decision == "UP" else 1 - prob[1]

    print(f"🔮 Previsão: {decision} (Confiança: {confidence:.2%})")

    if confidence < min_confidence:
        print("⚠️ Confiança abaixo do mínimo. Operação cancelada.")
        return None, confidence
    return decision, confidence

# HISTÓRICO
def registrar_trade(data):
    arquivo = "historico_trades.csv"
    cabecalho = ["data_hora", "direcao", "confianca", "resultado", "saldo_antes", "saldo_depois"]

    try:
        with open(arquivo, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(cabecalho)
            writer.writerow(data)
        print("📝 Histórico salvo.")
    except Exception as e:
        print(f"❌ Falha ao salvar histórico: {e}")

# GRÁFICO DE RESULTADOS
def plot_historico_trades(csv_file="historico_trades.csv"):
    df = pd.read_csv(csv_file)
    df["acumulado"] = df["resultado"].cumsum()

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

# BOT DE LOGIN E TRADE
def start_bot(email, password, stop_win, stop_loss, amount, decision, confidence):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    try:
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 15)
        driver.get("https://qxbroker.com/pt")

        # LOGIN
        wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/sign-in')]"))).click()
        wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='email']"))).send_keys(email)
        driver.find_element(By.XPATH, "//input[@type='password']").send_keys(password)
        driver.find_element(By.XPATH, "//form//button").click()
        time.sleep(5)

        # PLATAFORMA
        wait.until(EC.presence_of_element_located((By.XPATH, "//div[@class='platform']")))
        saldo_anterior = float(driver.find_element(By.XPATH, "//*[@id='root']//header//div[contains(@class, 'balance')]").text.replace("$", "").replace(",", "").strip())
        print(f"💰 Saldo inicial: ${saldo_anterior:.2f}")

        # VALOR
        field = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="root"]/div/div[1]/main/div[2]/div[1]/div/div[5]/div[2]/div/div/input')))
        field.click()
        field.send_keys(Keys.CONTROL + 'a')
        field.send_keys(Keys.BACK_SPACE)
        field.send_keys(str(amount))

        # CLIQUE
        if decision == "UP":
            driver.find_element(By.XPATH, "//button[contains(text(), 'Para cima')]").click()
            print("⬆️ Trade: Para cima")
        else:
            driver.find_element(By.XPATH, "//button[contains(text(), 'Para baixo')]").click()
            print("⬇️ Trade: Para baixo")

        time.sleep(70)  # Aguarda fim da operação

        saldo_atual = float(driver.find_element(By.XPATH, "//*[@id='root']//header//div[contains(@class, 'balance')]").text.replace("$", "").replace(",", "").strip())
        resultado = saldo_atual - saldo_anterior
        print(f"📈 Resultado: ${resultado:.2f}")

        registrar_trade([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            decision,
            round(confidence * 100, 2),
            round(resultado, 2),
            round(saldo_anterior, 2),
            round(saldo_atual, 2),
        ])

        if resultado >= stop_win:
            print("🏆 Stop Win atingido!")
        elif abs(resultado) >= stop_loss:
            print("🛑 Stop Loss atingido.")

        driver.quit()

    except Exception as e:
        print(f"❌ Erro no bot: {e}")

# ENTRADAS
with open("input.txt", "r") as f:
    lines = f.read().splitlines()
    email = lines[0]
    password = lines[1]
    stop_win = float(lines[2])
    stop_loss = float(lines[3])
    fixed_amount = float(lines[4])

# EXECUÇÃO
print("🚀 Iniciando pipeline completo com IA + ajustes + gráfico")

df = fetch_asset_data()
train_model(df)

# ajuste da confiança mínima
min_conf = confianca_dinamica()

decision, confidence = predict_next(df, min_confidence=min_conf)

if decision:
    start_bot(email, password, stop_win, stop_loss, fixed_amount, decision, confidence)
    plot_historico_trades()

# === DASHBOARD STREAMLIT INTEGRADO ===
try:
    if st._is_running_with_streamlit:
        st.set_page_config(page_title="IA Binárias", layout="wide")
        st.title("📊 Painel do Bot de Opções Binárias com IA")

        try:
            df_dash = pd.read_csv("historico_trades.csv")
            df_dash["data_hora"] = pd.to_datetime(df_dash["data_hora"])
            df_dash["acumulado"] = df_dash["resultado"].cumsum()

            st.subheader("📈 Lucro/Prejuízo Acumulado")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_dash["data_hora"], df_dash["acumulado"], marker="o", color="green")
            ax.set_title("Evolução do saldo")
            ax.grid(True)
            st.pyplot(fig)

            st.subheader("📋 Últimos trades")
            st.dataframe(df_dash.sort_values(by="data_hora", ascending=False).head(10))

        except FileNotFoundError:
            st.warning("📂 Execute o bot primeiro para gerar 'historico_trades.csv'")
except:
    pass
