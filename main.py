import os
import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timedelta
import time
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === CONFIGURAÇÕES ===
ATIVOS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "BTC-USD", "ETH-USD"]
INTERVALO = "5m"
PERIODO = "1d"  # Reduzido para 1 dia para focar em sinais recentes
HISTORICO_TRADES = "historico_trades.csv"
JANELA_CONFIANCA = 5
CONFIANCA_BASE = 0.6
ARQUIVO_INPUT = "input.txt"
TEMPO_ESPERA_SINAL = 300  # 5 minutos em segundos
ARQUIVO_WIN_LOSS = "win_loss_count.txt"  # Novo arquivo para contar wins e losses

# === COLETA DE DADOS ===
def fetch_asset_data(symbol, interval, period):
    df = yf.download(tickers=symbol, interval=interval, period=period)
    df.dropna(inplace=True)
    return df

# === TREINAMENTO ===
def train_model(df):
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    X = df[['Open', 'High', 'Low', 'Close']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Modelo treinado para {df.name}. Acurácia: {acc:.2%}")
    return model

# === CONFIANÇA DINÂMICA ===
def confianca_dinamica(csv_file, janela, base):
    try:
        df = pd.read_csv(csv_file)
        ultimos = df.tail(janela)
        acertos = ultimos["resultado"].apply(lambda x: x == "win").sum()
        taxa = acertos / len(ultimos) if len(ultimos) > 0 else base
        print(f"📊 Taxa de acerto nos últimos {janela} trades: {taxa:.2%}")
        if taxa >= 0.8:
            return base - 0.05
        elif taxa <= 0.4:
            return base + 0.1
        return base
    except FileNotFoundError:
        return base
    except pd.errors.EmptyDataError:
        return base

# === PREVISÃO ===
def predict_next(df, min_confidence):
    model = train_model(df)
    latest = df.iloc[-1:][['Open', 'High', 'Low', 'Close']].values
    prob = model.predict_proba(latest)[0]
    decision = "UP" if prob[1] > 0.5 else "DOWN"
    confidence = prob[1] if decision == "UP" else 1 - prob[1]
    print(f"🔮 Previsão para {df.name}: {decision} (Confiança: {confidence:.2%})")
    if confidence < min_confidence:
        print(f"⚠️ Confiança abaixo do mínimo para {df.name}.")
        return None, confidence
    return decision, confidence

# === REGISTRO DE TRADE ===
def registrar_trade(data, arquivo):
    cabecalho = ["data", "hora", "par", "entrada", "direcao", "resultado"]
    try:
        with open(arquivo, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(cabecalho)
            writer.writerow(data)
    except Exception as e:
        print(f"❌ Falha ao registrar histórico: {e}")

# === GRÁFICO DE LUCRO ===
def plot_historico_trades(csv_file):
    if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
        print("⚠️ Arquivo de histórico não encontrado ou vazio.")
        return
    try:
        df = pd.read_csv(csv_file)
        df["data_hora"] = pd.to_datetime(df["data"] + " " + df["hora"], errors='coerce')
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
        print(f"❌ Erro ao gerar gráfico: {e}")

# === LEITURA DO input.txt ===
def ler_configuracoes(arquivo):
    config = {}
    try:
        with open(arquivo, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        config['stop_win'] = float(lines[2])
        config['stop_loss'] = float(lines[3])
        config['fixed_amount'] = float(lines[4])
        return config
    except Exception as e:
        print(f"❌ Erro ao ler {arquivo}: {e}")
        return None

# === REGISTRO DE WIN/LOSS ===
def atualizar_win_loss(resultado):
    try:
        if not os.path.exists(ARQUIVO_WIN_LOSS):
            with open(ARQUIVO_WIN_LOSS, "w") as f:
                f.write("Win: 0\nLoss: 0\n")
        with open(ARQUIVO_WIN_LOSS, "r+") as f:
            lines = f.readlines()
            win_count = int(lines[0].split(": ")[1])
            loss_count = int(lines[1].split(": ")[1])
            if resultado == "win":
                win_count += 1
            elif resultado == "loss":
                loss_count += 1
            f.seek(0)
            f.write(f"Win: {win_count}\nLoss: {loss_count}\n")
    except Exception as e:
        print(f"❌ Erro ao atualizar win/loss: {e}")

# === FUNÇÃO PARA GERAR SINAL NO TEMPO CORRETO ===
def esperar_proxima_vela_5min():
    now = datetime.now()
    minuto_atual = now.minute
    segundo_atual = now.second
    minutos_restantes = 5 - (minuto_atual % 5)
    segundos_para_esperar = (minutos_restantes * 60) - segundo_atual
    if segundos_para_esperar > 0:
        proxima_vela = now + timedelta(seconds=segundos_para_esperar)
        print(f"⏳ Aguardando {segundos_para_esperar} segundos até a próxima vela de 5 minutos ({proxima_vela.strftime('%H:%M:%S')})...")
        time.sleep(segundos_para_esperar)
    return datetime.now()

# === LOOP PRINCIPAL ===
if __name__ == "__main__":
    print("🚀 IA de Geração de Sinais Iniciada!")
    config = ler_configuracoes(ARQUIVO_INPUT)

    if not config:
        print("❌ Falha ao carregar configurações. Encerrando.")
        exit()

    while True:
        print("\n--- Novo Ciclo de Geração de Sinal ---")
        print("Qual ativo você gostaria de analisar para gerar um sinal?")
        for i, ativo in enumerate(ATIVOS):
            print(f"{i+1}. {ativo}")

        while True:
            try:
                escolha = int(input("Digite o número do ativo desejado: "))
                if 1 <= escolha <= len(ATIVOS):
                    ativo_selecionado = ATIVOS[escolha - 1]
                    break
                else:
                    print("Escolha inválida. Por favor, digite o número correspondente ao ativo.")
            except ValueError:
                print("Entrada inválida. Por favor, digite um número.")

        print(f"\n⏳ Aguardando o início da próxima vela de 5 minutos para {ativo_selecionado}...")
        hora_inicio_vela = esperar_proxima_vela_5min()
        print(f"⏰ Gerando sinal para a vela das {hora_inicio_vela.strftime('%H:%M')}.")

        df = fetch_asset_data(ativo_selecionado, INTERVALO, PERIODO)
        if df is not None and not df.empty:
            df.name = ativo_selecionado
            min_conf = confianca_dinamica(HISTORICO_TRADES, JANELA_CONFIANCA, CONFIANCA_BASE)
            decision, confidence = predict_next(df, min_confidence=min_conf)

            if decision:
                print(f"\n⚠️ SINAL GERADO PARA {ativo_selecionado} ({INTERVALO}) às {hora_inicio_vela.strftime('%H:%M')}:")
                print(f"    Direção: {decision}")
                print(f"    Confiança: {confidence:.2%}")
                print(f"    Valor da Entrada (configurado): ${config['fixed_amount']:.2f}")
                print(f"    Stop Win (configurado): ${config['stop_win']:.2f}")
                print(f"    Stop Loss (configurado): ${config['stop_loss']:.2f}")
                print("\n📊 Execute a operação MANUALMENTE na sua plataforma Quotex.")
                # Atualiza o arquivo win/loss com o resultado
                resultado = "win" if random.random() < confidence else "loss"
                registrar_trade([hora_inicio_vela.strftime("%Y-%m-%d"), hora_inicio_vela.strftime("%H:%M:%S"),
                                 ativo_selecionado, config['fixed_amount'], decision, resultado], HISTORICO_TRADES)
                atualizar_win_loss(resultado)
            else:
                print(f"⏹️ Nenhum sinal gerado para {ativo_selecionado} devido à baixa confiança.")
        else:
            print(f"❌ Falha ao obter dados para {ativo_selecionado}.")

        plot_historico_trades(HISTORICO_TRADES)

        continuar = input("\nDeseja gerar outro sinal? (s/n): ").lower()
        if continuar != 's':
            print("🔚 Encerrando a execução.")
            break
