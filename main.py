# -*- coding: utf-8 -*-

# --- Bibliotecas Necessárias ---
print("Carregando bibliotecas...")
try:
    import yfinance as yf
    import pandas as pd
    from finta import TA # Mantém para outros indicadores
    import talib           # <--- Para Candlestick Patterns
    import datetime
    import time
    import random
    import gradio as gr
    import traceback
    import re
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pytz
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin, quote
    import numpy as np # TA-Lib usa numpy

    print("Bibliotecas carregadas com sucesso.")
except ImportError as e:
    talib_available = False
    if 'talib' in str(e).lower():
        print("***************************************************************")
        print("AVISO: Biblioteca TA-Lib não encontrada ou falha ao importar!")
        print("       A análise de Padrões de Vela será DESATIVADA.")
        print("       Para ativar, instale a biblioteca C TA-Lib primeiro")
        print("       (veja instruções no código-fonte ou online)")
        print("       e depois execute: pip install TA-Lib")
        print("***************************************************************")
    else:
        print(f"Erro ao importar biblioteca: {e}")
        print("Certifique-se de ter instalado: pip install --upgrade yfinance pandas finta gradio plotly pytz vaderSentiment requests beautifulsoup4 TA-Lib numpy")
        exit()
else:
    talib_available = True
    print("Biblioteca TA-Lib carregada.")


# --- Configuração ---
# (Listas de Ativos, Parâmetros Técnicos, Timezone, etc. - Mantidos como antes)
LISTA_FOREX = sorted([
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X',
    'USDCHF=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X',
    'AUDJPY=X', 'CADJPY=X', 'CHFJPY=X', 'NZDJPY=X', 'EURCAD=X',
    'EURAUD=X', 'EURCHF=X', 'EURNZD=X', 'GBPAUD=X', 'GBPCAD=X',
    'GBPCHF=X', 'GBPNZD=X', 'AUDCAD=X', 'AUDCHF=X', 'AUDNZD=X',
    'CADCHF=X', 'NZDCAD=X', 'NZDCHF=X', 'USDNOK=X', 'USDSEK=X',
    'USDZAR=X', 'USDMXN=X'
])
LISTA_CRIPTO = sorted([
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD',
    'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'SHIB-USD', 'DOT-USD',
    'TRX-USD', 'LINK-USD', 'MATIC-USD', 'LTC-USD', 'BCH-USD',
    'ETC-USD', 'XLM-USD', 'ICP-USD', 'FIL-USD', 'NEAR-USD'
])
print(f"Definidos {len(LISTA_FOREX)} pares Forex e {len(LISTA_CRIPTO)} ativos Cripto.")

RSI_PERIOD = 14; SMA_SHORT = 20; SMA_LONG = 50
RSI_OVERSOLD = 30; RSI_OVERBOUGHT = 70
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_FACTOR = 2.0

DEFAULT_LOOKBACK_DAYS_5M = 3
MAX_LOOKBACK_DAYS_5M = 7

BRT_TZ = pytz.timezone('America/Sao_Paulo')

try:
    sentiment_analyzer = SentimentIntensityAnalyzer()
    print("Analisador de Sentimento (VADER) inicializado.")
except Exception as e:
    print(f"Erro ao inicializar VADER: {e}. Análise de sentimento desativada.")
    sentiment_analyzer = None

# --- Textos Fixos para Fatores de Mercado (Mantidos) ---
# (FATORES_MERCADO_CRIPTO_MD e FATORES_MERCADO_FOREX_MD como antes)
FATORES_MERCADO_CRIPTO_MD = """
### 🌍 Fatores Gerais (Cripto)

Além da análise técnica M5, considere:
*   **Regulação:** Decisões governamentais (ETFs, stablecoins, etc.).
*   **Macroeconomia:** Juros (FED), inflação, dados econômicos globais.
*   **Adoção:** Entrada de instituições, novas aplicações (DeFi, NFT).
*   **Tecnologia:** Atualizações de protocolo (ETH, SOL, etc.), novas redes.
*   **Segurança:** Hacks em exchanges ou protocolos.
*   **Movimentação:** Grandes transferências ("baleias").
*   **Narrativas:** Hypes setoriais (IA, GameFi, Memes).
*   **Geopolítica:** Instabilidade global, busca por alternativas.
"""
FATORES_MERCADO_FOREX_MD = """
### 🌍 Fatores Gerais (Forex)

Além da análise técnica M5, o Forex é movido por:
*   **Política Monetária:** Decisões de taxas de juros e declarações de Bancos Centrais (FED, BCE, BoE, BoJ, etc.). *Diferenciais de juros* são cruciais.
*   **Dados Econômicos:** Inflação (CPI, PPI), PIB, Emprego (NFP, Taxa de Desemprego), Vendas no Varejo, Confiança do Consumidor/Empresarial (ISM, PMI). Surpresas nesses dados causam volatilidade.
*   **Fluxo de Capital:** Investimento estrangeiro direto (IED), balança comercial, fluxo para mercados de ações/títulos.
*   **Risco Sistêmico (Risk-On/Risk-Off):** Em tempos de incerteza (risk-off), moedas consideradas "seguras" (USD, JPY, CHF) tendem a se fortalecer. Em tempos de otimismo (risk-on), moedas de maior rendimento ou ligadas a commodities (AUD, NZD, CAD) podem subir.
*   **Preços de Commodities:** Impactam moedas de países exportadores (Ex: Petróleo para CAD, Minério de Ferro para AUD).
*   **Geopolítica:** Eleições, tensões internacionais, guerras, acordos comerciais.
*   **Intervenções:** Bancos Centrais podem intervir diretamente no mercado para influenciar a taxa de câmbio (raro em grandes economias, mas possível).
"""
# --- Módulos do Robô ---

# obter_dados_tecnicos (Mantida como na v1.8.2)
# ... (código completo de obter_dados_tecnicos aqui) ...
def obter_dados_tecnicos(ticker, dias_lookback=DEFAULT_LOOKBACK_DAYS_5M):
    """Baixa dados históricos INTRADAY (5m), normaliza colunas e MANTÉM timezone."""
    print(f"Baixando dados técnicos (5 minutos) para {ticker}...")
    try:
        periodo_busca = f"{int(dias_lookback)}d"
        # auto_adjust=True pode causar problemas com alguns tickers, testar False se necessário
        df = yf.download(ticker, period=periodo_busca, interval='5m', progress=False, timeout=30, auto_adjust=True)

        if df.empty: return None, f"AVISO: Nenhum dado intraday (5m) encontrado para {ticker} no período '{periodo_busca}'. Tente aumentar os dias."
        print(f"Dados para {ticker} baixados ({len(df)} regs 5m). Colunas originais: {df.columns.tolist()}")

        # Normalização de Colunas
        new_columns = []
        if isinstance(df.columns, pd.MultiIndex): # Verifica se é MultiIndex (comum com auto_adjust=False)
             new_columns = [col[0].lower().replace(' ', '_') if isinstance(col[0], str) else str(col[0]).lower() for col in df.columns]
        else: # Se for Index normal
             new_columns = [str(col).lower().replace(' ', '_') for col in df.columns]

        df.columns = new_columns
        print(f"Colunas normalizadas: {df.columns.tolist()}")


        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols: return None, f"ERRO: Faltam colunas OHLC essenciais ({', '.join(missing_cols)}) em {ticker}."

        # Tratamento de Volume
        if 'volume' not in df.columns:
            print(f"AVISO [{ticker}]: Coluna 'volume' não encontrada. Adicionando zeros.")
            df['volume'] = 0.0 # Use float zero
        else:
             # Converter volume para numérico, tratando erros (pode vir como object)
             df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
             if df['volume'].eq(0.0).all():
                 print(f"AVISO [{ticker}]: Coluna 'volume' contém apenas zeros após tratamento.")


        # Timezone
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                print(f"AVISO [{ticker}]: Dados sem timezone. Assumindo UTC e localizando...")
                try:
                    # Tenta localizar, tratando ambiguidades e inexistências
                    df.index = df.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
                except Exception as tz_err:
                    print(f"ERRO ao localizar timezone para UTC: {tz_err}. Ticker: {ticker}")
                    return None, f"Erro ao processar timezone para {ticker}"

            else:
                 print(f"Timezone original dos dados [{ticker}]: {df.index.tz}")
        else:
             print(f"ERRO [{ticker}]: Índice não é DatetimeIndex. Tipo: {type(df.index)}")
             return None, f"Índice inválido para {ticker}"


        # Limpeza OHLC NaNs
        rows_before = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        print(f"Linhas OHLC NaN removidas [{ticker}]: {rows_before - len(df)}")

        if df.empty: return None, f"AVISO [{ticker}]: DataFrame vazio pós-processamento."
        return df, None
    except Exception as e:
        print(f"Erro GERAL durante download/pré-processamento para {ticker}: {traceback.format_exc()}")
        return None, f"ERRO Inesperado (geral) ao baixar/processar dados para {ticker}: {e}."

# calcular_indicadores_tecnicos (Mantida como na v1.9)
# ... (código completo de calcular_indicadores_tecnicos aqui) ...
def calcular_indicadores_tecnicos(df):
    """Calcula indicadores: RSI, SMAs, BBands, MACD, Vol MA e Padrões de Vela (TA-Lib)."""
    sinais = {'Padroes_Vela': []} # Inicializa lista para padrões
    df_ta = df.copy()
    if df is None or df.empty: return None, sinais, "Erro interno: DataFrame vazio."
    print(f"Calculando indicadores técnicos (M5)...")
    try:
        required = ['open', 'high', 'low', 'close']
        if not all(col in df_ta.columns for col in required):
            missing = [c for c in required if c not in df_ta.columns]
            return None, sinais, f"Erro: Faltam colunas OHLC ({', '.join(missing)})."

        # Garante tipos corretos para TA-Lib/Finta
        for col in required + ['volume']:
            if col in df_ta.columns:
                df_ta[col] = pd.to_numeric(df_ta[col], errors='coerce')
        df_ta.dropna(subset=required, inplace=True) # Remove NaNs introduzidos por to_numeric
        if df_ta.empty: return None, sinais, "Erro: DataFrame vazio após conversão numérica OHLC."


        # Indicadores FINTA (mantidos)
        df_ta['rsi'] = TA.RSI(df_ta, period=RSI_PERIOD)
        df_ta['sma_short'] = TA.SMA(df_ta, period=SMA_SHORT, column='close')
        df_ta['sma_long'] = TA.SMA(df_ta, period=SMA_LONG, column='close')
        bb_df = TA.BBANDS(df_ta, period=SMA_SHORT, column='close'); bb_df.columns = ['bb_upper', 'bb_middle', 'bb_lower']
        df_ta = pd.concat([df_ta, bb_df], axis=1)
        macd_df = TA.MACD(df_ta, column='close')
        df_ta = pd.concat([df_ta, macd_df.rename(columns=str.lower)], axis=1)
        if 'macd' in df_ta.columns and 'signal' in df_ta.columns: df_ta['macd_hist'] = df_ta['macd'] - df_ta['signal']

        # Volume MA (mantido)
        sinais['Volume_Desc'] = 'N/A (Indisp./Zero)'
        sinais['Volume_Ratio'] = None
        if 'volume' in df_ta.columns and df_ta['volume'].gt(1e-9).any() and df_ta['volume'].notna().any():
            print("Calculando Volume Moving Average...")
            df_ta['volume_ma'] = df_ta['volume'].rolling(window=VOLUME_MA_PERIOD).mean()
            df_ta['volume_ma'] = df_ta['volume_ma'].replace(0, 1e-9)
            df_ta['volume_ma'] = df_ta['volume_ma'].bfill(limit=VOLUME_MA_PERIOD*2)
            df_ta['volume_ma'] = df_ta['volume_ma'].fillna(1e-9)
        else:
            print("AVISO: Volume indisponível/zero. 'volume_ma' não será calculado.")
            df_ta['volume_ma'] = 1e-9


        # --- CÁLCULO DE PADRÕES DE VELA (TA-Lib) ---
        if talib_available:
            print("Calculando Padrões de Vela (TA-Lib)...")
            candle_patterns = {
                'Hammer': talib.CDLHAMMER, 'Inv Hammer': talib.CDLINVERTEDHAMMER,
                'Engolfo Alta': talib.CDLENGULFING, 'Piercing': talib.CDLPIERCING,
                'Morning Star': talib.CDLMORNINGSTAR, '3 Soldados Brancos': talib.CDL3WHITESOLDIERS,
                'Hanging Man': talib.CDLHANGINGMAN, 'Shooting Star': talib.CDLSHOOTINGSTAR,
                'Engolfo Baixa': talib.CDLENGULFING, 'Dark Cloud': talib.CDLDARKCLOUDCOVER,
                'Evening Star': talib.CDLEVENINGSTAR, '3 Corvos Negros': talib.CDL3BLACKCROWS,
                'Doji': talib.CDLDOJI, 'Spinning Top': talib.CDLSPINNINGTOP,
            }
            detected_patterns_last_candle = []
            # Prepara arrays numpy para TA-Lib
            open_p = df_ta['open'].values.astype(float)
            high_p = df_ta['high'].values.astype(float)
            low_p = df_ta['low'].values.astype(float)
            close_p = df_ta['close'].values.astype(float)

            for pattern_name, pattern_func in candle_patterns.items():
                try:
                    result = pattern_func(open_p, high_p, low_p, close_p)
                    if len(result) > 0: # Garante que houve resultado
                        last_result = result[-1]
                        if last_result != 0: # Se um padrão foi detectado
                             is_bullish = last_result > 0
                             is_bearish = last_result < 0
                             # Trata Engulfing que usa a mesma função
                             if pattern_name == 'Engolfo Alta' and is_bearish: continue
                             if pattern_name == 'Engolfo Baixa' and is_bullish: continue

                             strength = ""
                             if abs(last_result) == 200: strength = "++Forte" if is_bullish else "--Forte"
                             elif abs(last_result) == 100: strength = "+Alta" if is_bullish else "-Baixa"

                             if strength: # Adiciona apenas se classificado
                                 detected_patterns_last_candle.append(f"{pattern_name} ({strength})")

                except Exception as e_talib:
                    print(f"  AVISO: Erro ao calcular padrão TA-Lib '{pattern_name}': {e_talib}")

            if detected_patterns_last_candle:
                 print(f"  Padrões detectados no último candle: {', '.join(detected_patterns_last_candle)}")
                 sinais['Padroes_Vela'] = detected_patterns_last_candle
            else:
                 print("  Nenhum padrão de vela TA-Lib detectado no último candle.")
        else:
            print("TA-Lib indisponível, pulando cálculo de padrões de vela.")
        # --- FIM DO CÁLCULO DE PADRÕES ---


        # Limpeza final NaNs (mantida)
        rows_before_na = len(df_ta)
        cols_to_check_na = ['rsi', 'sma_short', 'sma_long', 'bb_upper', 'macd', 'signal']
        cols_present = [col for col in cols_to_check_na if col in df_ta.columns]
        if cols_present:
            df_ta.dropna(subset=cols_present, inplace=True)
            print(f"Linhas removidas devido a NaNs dos indicadores: {rows_before_na - len(df_ta)}")

        if df_ta.empty or len(df_ta) < 2:
             return None, sinais, f"Dados insuficientes pós cálculo/limpeza ({len(df_ta)} linhas)."

        # Geração de Sinais Descritivos (mantida)
        last, prev = df_ta.iloc[-1], df_ta.iloc[-2]
        # ... (lógica restante para RSI_Desc, SMA_Desc, Volume_Desc, MACD_Desc, BB_Desc igual à v1.8.2) ...
        rsi_c, sma_s_c, sma_l_c = 'rsi', 'sma_short', 'sma_long'
        vol_c, vol_ma_c = 'volume', 'volume_ma'
        macd_l_c, macd_s_c = 'macd', 'signal'
        bb_u_c, bb_m_c, bb_l_c = 'bb_upper', 'bb_middle', 'bb_lower'

        # RSI
        if rsi_c in last and pd.notna(last[rsi_c]):
            v = round(last[rsi_c], 2); sinais['RSI_Valor'] = v
            if v < RSI_OVERSOLD: sinais['RSI_Desc'] = f'Sobrevendido'
            elif v > RSI_OVERBOUGHT: sinais['RSI_Desc'] = f'Sobrecomprado'
            else: sinais['RSI_Desc'] = f'Neutro'
        else: sinais['RSI_Desc'] = 'N/A'

        # SMA
        if all(c in last and pd.notna(last[c]) and c in prev and pd.notna(prev[c]) for c in [sma_s_c, sma_l_c]):
            if last[sma_s_c] > last[sma_l_c] and prev[sma_s_c] <= prev[sma_l_c]: sinais['SMA_Desc'] = 'Cruz. Alta Recente'
            elif last[sma_s_c] < last[sma_l_c] and prev[sma_s_c] >= prev[sma_l_c]: sinais['SMA_Desc'] = 'Cruz. Baixa Recente'
            elif last[sma_s_c] > last[sma_l_c]: sinais['SMA_Desc'] = 'Tend. Alta (Curta>Longa)'
            else: sinais['SMA_Desc'] = 'Tend. Baixa (Curta<Longa)'
        else: sinais['SMA_Desc'] = 'N/A'

        # Volume (Descrição e Ratio)
        if vol_c in last and vol_ma_c in last and pd.notna(last[vol_ma_c]) and last[vol_ma_c] > 1e-9 and pd.notna(last[vol_c]):
             volume_ratio = last[vol_c] / last[vol_ma_c]
             sinais['Volume_Ratio'] = volume_ratio
             if volume_ratio > VOLUME_SPIKE_FACTOR: sinais['Volume_Desc'] = f'Alto ({volume_ratio:.1f}x Média)'
             elif volume_ratio < (1 / VOLUME_SPIKE_FACTOR): sinais['Volume_Desc'] = f'Baixo ({volume_ratio:.1f}x Média)'
             else: sinais['Volume_Desc'] = f'Normal ({volume_ratio:.1f}x Média)'
        # Se não, mantém o default 'N/A (Indisp./Zero)'

        # MACD
        if all(c in last and pd.notna(last[c]) and c in prev and pd.notna(prev[c]) for c in [macd_l_c, macd_s_c]):
             if last[macd_l_c] > last[macd_s_c] and prev[macd_l_c] <= prev[macd_s_c]: sinais['MACD_Desc'] = 'Cruz. Alta Recente'
             elif last[macd_l_c] < last[macd_s_c] and prev[macd_l_c] >= prev[macd_s_c]: sinais['MACD_Desc'] = 'Cruz. Baixa Recente'
             elif last[macd_l_c] > last[macd_s_c]: sinais['MACD_Desc'] = 'Viés Alta (MACD>Sinal)'
             else: sinais['MACD_Desc'] = 'Viés Baixa (MACD<Sinal)'
        else: sinais['MACD_Desc'] = 'N/A'

        # BBands
        if all(c in last and pd.notna(last[c]) for c in [bb_u_c, bb_l_c, 'close', bb_m_c]):
             dist_bb = last[bb_u_c] - last[bb_l_c]
             price_pos = (last['close'] - last[bb_l_c]) / dist_bb if dist_bb > 1e-9 else 0.5 # Posição relativa
             if dist_bb > 1e-9:
                 if last['close'] > last[bb_u_c]: sinais['BB_Desc'] = 'Fora da BB Superior'
                 elif last['close'] < last[bb_l_c]: sinais['BB_Desc'] = 'Fora da BB Inferior'
                 elif price_pos > 0.95: sinais['BB_Desc'] = 'Próximo à BB Superior'
                 elif price_pos < 0.05: sinais['BB_Desc'] = 'Próximo à BB Inferior'
                 else: sinais['BB_Desc'] = 'Dentro das Bandas'
             else: sinais['BB_Desc'] = 'Dentro (Bandas Colapsadas?)'
        else: sinais['BB_Desc'] = 'N/A'


        print("Descrições técnicas (incluindo padrões de vela) geradas."); return df_ta, sinais, None
    except Exception as e:
        print(f"Erro calc. ind.: {traceback.format_exc()}")
        return df, {}, f"ERRO Inesperado ao calcular indicadores: {e}"


# _fetch_and_parse_google_search_news (Mantida como na v1.8.2)
# ... (código completo da função helper de scraping aqui) ...
def _fetch_and_parse_google_search_news(query, headers, time_filter='w', num_articles=5):
    """
    Helper: fetches Google Search results filtered for news (&tbm=nws) and parses articles.
    time_filter: 'd' (day), 'w' (week), 'm' (month), 'y' (year), '' (all)
    <<< AVISO: OS SELETORES HTML SÃO EXTREMAMENTE FRÁGEIS E PODEM QUEBRAR >>>
    """
    base_google_url = "https://www.google.com/"
    # Filter for NEWS results (&tbm=nws) on main Google Search
    time_param = f"&tbs=qdr:{time_filter}" if time_filter in ['d', 'w', 'm', 'y'] else ""
    search_url = f"{base_google_url}search?q={quote(query)}&hl=pt-BR&gl=BR&tbm=nws{time_param}"

    print(f"  Scraping Google Search URL: {search_url}")
    results = []
    try:
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        # Salvar HTML para depuração (descomente se necessário)
        # with open(f"debug_{query[:10].replace('/','')}.html", "w", encoding="utf-8") as f:
        #     f.write(response.text)
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- SELETORES PARA GOOGLE SEARCH (FRÁGEIS!) ---
        # Abordagem: Encontrar os links principais e tentar extrair info ao redor deles
        article_links = soup.find_all('a', class_=re.compile(r'WlydOe|TxGVr|DY5T1d'), limit=num_articles * 2) # Tenta algumas classes comuns
        if not article_links:
             headers3 = soup.find_all('h3'); article_links = [h.find('a', href=True) for h in headers3 if h.find('a', href=True)]; article_links = article_links[:num_articles*2]
             if not article_links: print(f"  AVISO: Não foram encontrados links de notícias com seletores testados para query '{query[:30]}...'"); return []

        count = 0; processed_links = set()
        for link_tag in article_links:
            if count >= num_articles: break
            link = link_tag.get('href');
            if not link or not link.startswith('http'): continue

            container = link_tag.find_parent('div', class_=re.compile(r'SoaBEf|JJZKK|n0jPhd'))
            if not container: container = link_tag.find_parent('div', recursive=False)
            if not container: continue

            title = link_tag.text.strip();
            if not title: header = container.find(['h3','h4']); title = header.text.strip() if header else "Título N/A"

            if link in processed_links: continue
            processed_links.add(link)

            source = "Fonte N/A"; date_str = "Data N/A"
            metadata_div = container.find('div', class_=re.compile(r'OSrXXb|s3v9rd'))
            if metadata_div:
                spans = metadata_div.find_all('span')
                if len(spans) >= 1: source = spans[0].text.strip()
                if len(spans) >= 2: date_str = spans[-1].text.strip()
            else:
                 all_spans = container.find_all('span')
                 if source == "Fonte N/A" and len(all_spans) > 0:
                     potential_sources = [s.text.strip() for s in all_spans if not re.search(r'hora|dia|semana|mês', s.text, re.I)];
                     if potential_sources: source = potential_sources[0]
                 if date_str == "Data N/A" and len(all_spans) > 0: date_str = all_spans[-1].text.strip()

            results.append({'title': title, 'link': link, 'source': source, 'date_str': date_str}); count += 1

    except requests.exceptions.Timeout: print(f"  ERRO: Timeout ao buscar notícias (Google Search) para query: '{query[:50]}...'"); return []
    except requests.exceptions.RequestException as e: print(f"  ERRO: Falha na requisição de notícias (Google Search) para query '{query[:50]}...': {e}"); return []
    except Exception as e: print(f"  ERRO: Falha inesperada no scraping/parsing (Google Search) para query '{query[:50]}...': {traceback.format_exc()}"); return []

    print(f"  Encontrados {len(results)} artigos relevantes (Google Search) para '{query[:50]}...'"); return results


# obter_noticias_google_search_scraping (CORRIGIDO erro de sintaxe)
def obter_noticias_google_search_scraping(ticker, asset_type):
    """Busca notícias por scraping no GOOGLE SEARCH (&tbm=nws) e calcula o sentimento."""
    print(f"Buscando notícias (Scraping Google Search - Última Semana) para {ticker} ({asset_type})...")
    noticias_especificas_md = f"### 📰 Notícias Específicas ({ticker} - Semana)\n\n"
    noticias_gerais_md = f"### 🌍 Notícias Gerais (Mercado/Mundo - Semana)\n\n"
    sentimento_desc = f"### 🤔 Sentimento (Notícias da Semana)\n\n* "

    if not sentiment_analyzer:
        no_vader_msg = "* Análise de Sentimento Indisponível.\n"
        return noticias_especificas_md + "* Busca N/A.\n", noticias_gerais_md + "* Busca N/A.\n", sentimento_desc + no_vader_msg

    headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36', 'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9', 'Referer': 'https://www.google.com/', 'DNT': '1', }
    todos_titulos = []; resultados_formatados_especificos = []; resultados_formatados_gerais = []

    # 1. Notícias Específicas
    query_especifica = "";
    if asset_type == 'Cripto': parts = ticker.split('-'); coin_map = {'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'XRP': 'XRP Ledger', 'ADA': 'Cardano', 'BNB': 'Binance Coin'}; coin_name = parts[0] if len(parts)>0 else ''; full_name = coin_map.get(coin_name, coin_name); query_especifica = f'"{full_name}" OR "{coin_name}" criptomoeda' if coin_name else ''
    elif asset_type == 'Forex': pair = ticker.split('=')[0]; curr1=pair[:3]; curr2=pair[3:]; query_especifica = f'"{curr1} {curr2}" OR "{curr1}/{curr2}" Forex' if len(pair)==6 else f'"{pair}" Forex'

    if query_especifica:
        resultados_especificos = _fetch_and_parse_google_search_news(query_especifica, headers, time_filter='w', num_articles=4)
        if resultados_especificos: [ (resultados_formatados_especificos.append(f"*   [{res['title']}]({res['link']}) - *{res['source']} ({res['date_str']})*\n"), todos_titulos.append(res['title'])) for res in resultados_especificos]
        else: resultados_formatados_especificos.append("* Nenhuma notícia específica encontrada (Google Search - Semana).\n")
        time.sleep(random.uniform(2.5, 5.0))
    else: resultados_formatados_especificos.append("* Query específica não definida.\n")

    # 2. Notícias Gerais
    query_geral = '"mercado financeiro" OR "economia global" OR "banco central" OR "inflação" OR "geopolítica"'; resultados_gerais = _fetch_and_parse_google_search_news(query_geral, headers, time_filter='w', num_articles=4)
    if resultados_gerais: [ (resultados_formatados_gerais.append(f"*   [{res['title']}]({res['link']}) - *{res['source']} ({res['date_str']})*\n"), todos_titulos.append(res['title'])) for res in resultados_gerais if res['title'] not in todos_titulos]
    else: resultados_formatados_gerais.append("* Nenhuma notícia geral encontrada (Google Search - Semana).\n")

    noticias_especificas_md += "".join(resultados_formatados_especificos); noticias_gerais_md += "".join(resultados_formatados_gerais)

    # 3. Análise de Sentimento (CORRIGIDO SyntaxError)
    sentimento_geral = {'pos': 0, 'neg': 0, 'neu': 0, 'count': 0, 'compound_sum': 0}
    print(f"Analisando sentimento de {len(todos_titulos)} títulos...")
    if todos_titulos:
        for titulo in todos_titulos:
            try: # --- Início do TRY ---
                vs = sentiment_analyzer.polarity_scores(titulo)
                sentimento_geral['compound_sum'] += vs['compound']
                sentimento_geral['count'] += 1
                # Classificação em múltiplas linhas para clareza
                if vs['compound'] >= 0.05:
                    sentimento_geral['pos'] += 1
                elif vs['compound'] <= -0.05:
                    sentimento_geral['neg'] += 1
                else:
                    sentimento_geral['neu'] += 1
            except Exception as e_sent: # --- Bloco EXCEPT CORRETO ---
                 print(f"Erro VADER no título '{titulo[:50]}...': {e_sent}")
        # Lógica de cálculo do sentimento final (após o loop)
        if sentimento_geral['count'] > 0:
            avg_compound = sentimento_geral['compound_sum'] / sentimento_geral['count']
            sentimento_desc += f"Analisadas: {sentimento_geral['count']} ({sentimento_geral['pos']} Pos, {sentimento_geral['neg']} Neg, {sentimento_geral['neu']} Neu)\n* Pontuação Média: {avg_compound:.3f} "
            sentimento_desc += "(Positivo 👍)" if avg_compound >= 0.05 else ("(Negativo 👎)" if avg_compound <= -0.05 else "(Neutro 😐)")
        else:
            sentimento_desc += "* Cálculo indisponível.\n"
    else:
         sentimento_desc += "* N/A (sem notícias).\n"

    print("Busca e análise de notícias (Scraping Google Search) concluída."); return noticias_especificas_md, noticias_gerais_md, sentimento_desc


# gerar_sinal_binario_5m (Mantida como na v1.9)
# ... (código completo de gerar_sinal_binario_5m aqui) ...
def gerar_sinal_binario_5m(ticker, sinais_tecnicos, df_com_ta, asset_type='forex'):
    """Gera sinal baseado em indicadores, volume (Cripto) e padrões de vela."""
    print(f"Gerando sinal Binário 5m para {ticker} (Tipo: {asset_type})...")
    if not sinais_tecnicos or df_com_ta is None or df_com_ta.empty:
        return { 'sinal': 'ERRO', 'probabilidade': 0, 'razao': 'Dados técnicos indisponíveis.' }

    score_call = 0; score_put = 0; razoes_call = []; razoes_put = []
    last_row = df_com_ta.iloc[-1]

    # --- Pontuação: Indicadores Base ---
    rsi_val = sinais_tecnicos.get('RSI_Valor')
    if rsi_val is not None:
        if rsi_val < RSI_OVERSOLD: score_call += 1.5; razoes_call.append(f"RSI Sobrevendido ({rsi_val:.1f})") # Peso ligeiramente menor
        elif rsi_val < 40: score_call += 0.75; razoes_call.append(f"RSI Baixo ({rsi_val:.1f})")
        elif rsi_val > RSI_OVERBOUGHT: score_put += 1.5; razoes_put.append(f"RSI Sobrecomprado ({rsi_val:.1f})") # Peso ligeiramente menor
        elif rsi_val > 60: score_put += 0.75; razoes_put.append(f"RSI Alto ({rsi_val:.1f})")

    sma_desc = sinais_tecnicos.get('SMA_Desc', '')
    if 'Alta' in sma_desc: score_call += 1.0; razoes_call.append("SMAs viés alta")
    elif 'Baixa' in sma_desc: score_put += 1.0; razoes_put.append("SMAs viés baixa")

    macd_desc = sinais_tecnicos.get('MACD_Desc', '')
    if 'Alta' in macd_desc: score_call += 1.25; razoes_call.append("MACD viés/cruz alta") # Peso ligeiramente menor
    elif 'Baixa' in macd_desc: score_put += 1.25; razoes_put.append("MACD viés/cruz baixa") # Peso ligeiramente menor

    bb_desc = sinais_tecnicos.get('BB_Desc', '')
    if 'Inferior' in bb_desc: score_call += (1.0 if 'Fora' in bb_desc else 0.5); razoes_call.append(f"BB {bb_desc}") # Peso ligeiramente menor
    elif 'Superior' in bb_desc: score_put += (1.0 if 'Fora' in bb_desc else 0.5); razoes_put.append(f"BB {bb_desc}") # Peso ligeiramente menor

    # --- Pontuação: Volume (APENAS CRIPTO) ---
    if asset_type == 'crypto':
        volume_ratio = sinais_tecnicos.get('Volume_Ratio')
        if volume_ratio is not None and 'close' in last_row and 'open' in last_row:
            is_bullish_candle = last_row['close'] > last_row['open']
            is_bearish_candle = last_row['close'] < last_row['open']
            if volume_ratio > VOLUME_SPIKE_FACTOR:
                vol_razao = f"Volume Alto ({volume_ratio:.1f}x)"
                if is_bullish_candle: score_call += 0.75; razoes_call.append(f"{vol_razao} em Alta")
                elif is_bearish_candle: score_put += 0.75; razoes_put.append(f"{vol_razao} em Baixa")
                print(f"-> Volume Cripto: {vol_razao} considerado.")

    # --- Pontuação: Padrões de Vela (TA-Lib) ---
    padroes_vela = sinais_tecnicos.get('Padroes_Vela', [])
    if padroes_vela:
        print(f"-> Considerando Padrões de Vela: {', '.join(padroes_vela)}")
        for padrao_str in padroes_vela:
            match = re.match(r"^(.*?)\s*\((.*)\)$", padrao_str)
            if match:
                nome_padrao = match.group(1).strip()
                forca_sinal = match.group(2).strip()
                if 'Alta' in forca_sinal:
                    peso = 1.5 if '++' in forca_sinal else 1.0
                    score_call += peso; razoes_call.append(f"Padrão: {nome_padrao}"); print(f"   - Padrão Alta: {nome_padrao} (Score +{peso})")
                elif 'Baixa' in forca_sinal:
                    peso = 1.5 if '--' in forca_sinal else 1.0
                    score_put += peso; razoes_put.append(f"Padrão: {nome_padrao}"); print(f"   - Padrão Baixa: {nome_padrao} (Score +{peso})")
                elif 'Doji' in nome_padrao or 'Spinning' in nome_padrao:
                     razoes_call.append(f"(Indecisão: {nome_padrao})"); razoes_put.append(f"(Indecisão: {nome_padrao})"); print(f"   - Padrão Indecisão: {nome_padrao}")

    # --- Consolidação Final ---
    sinal_final = "NEUTRO"; prob = 50; razao_final = ""; diff = score_call - score_put
    threshold_forte = 3.5; threshold_leve = 1.5

    if not razoes_call and not razoes_put: razao_final = "Nenhum indicador ou padrão apresentou sinal claro."
    elif diff >= threshold_forte: sinal_final = "CALL"; prob = 75 + int(min(diff * 2.5, 15)); razao_final = "Forte Confluência Alta: " + ", ".join(razoes_call)
    elif diff >= threshold_leve: sinal_final = "CALL"; prob = 60 + int(min(diff * 5, 15)); razao_final = "Leve Vantagem Alta: " + ", ".join(razoes_call)
    elif diff <= -threshold_forte: sinal_final = "PUT"; prob = 75 + int(min(abs(diff) * 2.5, 15)); razao_final = "Forte Confluência Baixa: " + ", ".join(razoes_put)
    elif diff <= -threshold_leve: sinal_final = "PUT"; prob = 60 + int(min(abs(diff) * 5, 15)); razao_final = "Leve Vantagem Baixa: " + ", ".join(razoes_put)
    else: sinal_final = "NEUTRO"; prob = 50; razao_final = "Sinais Mistos/Insuficientes. ";
    if razoes_call: razao_final += f"Pró-Alta: [{', '.join(razoes_call)}]. "
    if razoes_put: razao_final += f"Pró-Baixa: [{', '.join(razoes_put)}]."

    prob = max(50, min(90, prob)) if sinal_final != "NEUTRO" else 50
    print(f"Sinal Final ({asset_type}): {sinal_final}, Prob: {prob}%, Score C:{score_call:.2f}/P:{score_put:.2f}")
    return { 'sinal': sinal_final, 'probabilidade': prob, 'razao': razao_final }


# gerar_grafico_ativo (Mantida como na v1.8.2)
# ... (código completo de gerar_grafico_ativo aqui) ...
def gerar_grafico_ativo(df_ta, ticker):
    """Gera gráfico Plotly M5 com Candlestick, Indicadores e Volume (se disponível). Eixo X em BRT."""
    if df_ta is None or df_ta.empty: return None
    print(f"Gerando gráfico M5 para {ticker}...")
    try:
        df_plot = df_ta.copy()
        # Conversão Timezone
        if df_plot.index.tz is None: df_plot.index = df_plot.index.tz_localize('UTC').tz_convert(BRT_TZ)
        else: df_plot.index = df_plot.index.tz_convert(BRT_TZ)

        # Verifica se plota Volume
        plot_volume = 'volume' in df_plot.columns and df_plot['volume'].gt(1e-9).any()
        n_rows = 4 if plot_volume else 3
        row_heights = [0.55, 0.15, 0.15, 0.15] if plot_volume else [0.6, 0.2, 0.2]

        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=row_heights)

        # --- Plots (semelhante, mas nomes podem ser ligeiramente ajustados para clareza) ---
        # Linha 1: Preço
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Candle 5m'), row=1, col=1)
        if 'sma_short' in df_plot.columns: fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['sma_short'], mode='lines', name=f'SMA {SMA_SHORT}', line=dict(color='blue', width=1)), row=1, col=1)
        if 'sma_long' in df_plot.columns: fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['sma_long'], mode='lines', name=f'SMA {SMA_LONG}', line=dict(color='orange', width=1)), row=1, col=1)
        if all(c in df_plot.columns for c in ['bb_upper', 'bb_lower']):
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['bb_upper'], mode='lines', name='BB Sup', line=dict(color='rgba(150,150,150,0.7)', width=1, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['bb_lower'], mode='lines', name='BB Inf', line=dict(color='rgba(150,150,150,0.7)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(150,150,150,0.1)'), row=1, col=1)

        # Linha 2: RSI
        if 'rsi' in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['rsi'], mode='lines', name='RSI', line=dict(color='purple', width=1.5)), row=2, col=1)
            fig.add_hline(y=RSI_OVERBOUGHT, line=dict(color='red', dash='dash', width=1), annotation_text=str(RSI_OVERBOUGHT), annotation_position="top right", row=2, col=1)
            fig.add_hline(y=RSI_OVERSOLD, line=dict(color='green', dash='dash', width=1), annotation_text=str(RSI_OVERSOLD), annotation_position="bottom right", row=2, col=1)

        # Linha 3: MACD
        if 'macd' in df_plot.columns: fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['macd'], mode='lines', name='MACD', line=dict(color='rgb(0, 200, 0)', width=1.5)), row=3, col=1)
        if 'signal' in df_plot.columns: fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['signal'], mode='lines', name='Sinal', line=dict(color='rgb(255, 100, 100)', width=1.5)), row=3, col=1)
        if 'macd_hist' in df_plot.columns:
            colors = ['rgba(0, 200, 0, 0.6)' if val >= 0 else 'rgba(255, 100, 100, 0.6)' for val in df_plot['macd_hist']]
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['macd_hist'], name='Hist.', marker_color=colors), row=3, col=1)

        # Linha 4: Volume (Se plot_volume)
        if plot_volume:
            colors_vol = ['rgba(0, 200, 0, 0.6)' if df_plot.iloc[i]['close'] >= df_plot.iloc[i]['open'] else 'rgba(255, 100, 100, 0.6)' for i in range(len(df_plot))]
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'], name='Volume', marker_color=colors_vol, showlegend=False), row=4, col=1)
            if 'volume_ma' in df_plot.columns and df_plot['volume_ma'].gt(1e-9).any():
                 fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['volume_ma'], mode='lines', name=f'Vol MA{VOLUME_MA_PERIOD}', line=dict(color='rgba(100, 150, 255, 0.8)', width=1, dash='dash'), showlegend=False), row=4, col=1)


        # --- Layout (com correção yaxis4_title) ---
        last_date_brt_str = df_plot.index.max().strftime('%d/%m %H:%M') if not df_plot.empty else "N/A"
        duration_hours = (df_plot.index.max() - df_plot.index.min()).total_seconds() / 3600 if len(df_plot) > 1 else 0

        fig.update_layout(
            title=f'Análise Técnica M5 - {ticker} (~{duration_hours:.0f}h até {last_date_brt_str} BRT)',
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, traceorder='normal'),
            yaxis_title="Preço",   # Título eixo Y1 (Preço)
            yaxis2_title="RSI",    # Título eixo Y2 (RSI)
            yaxis3_title="MACD",   # Título eixo Y3 (MACD)
            # yaxis4_title removido daqui
            hovermode='x unified',
            template='plotly_dark', # Mudado para tema escuro
            height=750, # Aumenta altura
            margin=dict(l=50, r=50, t=90, b=50), # Ajuste margens para título/legenda
            xaxis_title="Horário (BRT)",
            # Configurações adicionais para aparência
            paper_bgcolor='rgba(17,17,17,1)', # Fundo geral mais escuro
            plot_bgcolor='rgba(17,17,17,1)',  # Fundo da área de plotagem
            font=dict(color='rgb(200,200,200)') # Cor da fonte geral
        )

        # --- CORREÇÃO AQUI ---
        # Define o título do eixo Y4 condicionalmente
        if plot_volume:
            fig.update_layout(yaxis4=dict(title_text="Volume")) # Define o título para yaxis4
        # --- FIM DA CORREÇÃO ---

        # Remove rangeslider dos eixos inferiores
        for i in range(2, n_rows + 1):
            fig.update_xaxes(showgrid=False, row=i, col=1) # Opcional: remove grid x inferior
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(100, 100, 100, 0.5)', row=i, col=1) # Grid y suave

        # Grid principal (preço)
        fig.update_xaxes(showgrid=False, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(100, 100, 100, 0.5)', row=1, col=1)


        print("Gráfico M5 gerado com sucesso (Horário BRT).")
        return fig
    except Exception as e:
        print(f"ERRO ao gerar gráfico M5 para {ticker}: {traceback.format_exc()}")
        return None

# --- Funções Auxiliares de Formatação (Mantidas como na v1.9) ---
# (converter_horarios_brt, formatar_sinais_descritivos, formatar_sinal_principal)
# ... (código completo das funções de formatação aqui) ...
def converter_horarios_brt(last_candle_time_original):
    """Converte timestamp original para BRT e formata strings."""
    print(f"Convertendo horário: {last_candle_time_original} (TZ original: {last_candle_time_original.tzinfo})")
    try:
        if not isinstance(last_candle_time_original, pd.Timestamp):
             # Tenta converter se não for Timestamp (pode ser datetime nativo)
             last_candle_time_original = pd.Timestamp(last_candle_time_original)

        if last_candle_time_original.tzinfo is None:
            last_candle_brt = pytz.utc.localize(last_candle_time_original).astimezone(BRT_TZ)
        else:
            last_candle_brt = last_candle_time_original.astimezone(BRT_TZ)

        next_candle_start_brt = last_candle_brt + pd.Timedelta(minutes=5)
        next_candle_end_brt = next_candle_start_brt + pd.Timedelta(minutes=5)
        time_format = '%d/%m/%Y %H:%M:%S'
        last_candle_brt_str = last_candle_brt.strftime(time_format)
        next_start_brt_str = next_candle_start_brt.strftime(time_format)
        next_end_brt_str = next_candle_end_brt.strftime(time_format)
        print(f"-> Última vela BRT: {last_candle_brt_str}, Próxima: {next_start_brt_str} - {next_end_brt_str}")
        return last_candle_brt, next_start_brt_str, next_end_brt_str, last_candle_brt_str
    except Exception as e:
        print(f"ERRO FATAL na conversão de horário: {e}")
        # Retornar valores padrão ou levantar exceção para parar a análise
        raise ValueError(f"Erro ao converter horário: {e}") from e


def formatar_sinais_descritivos(sinais_tecnicos_desc, last_candle_brt_str, tipo_ativo):
    """Formata o Markdown para o resumo dos indicadores e padrões."""
    out = f"### 📈 Indicadores {tipo_ativo} (Ref: {last_candle_brt_str} BRT)\n\n"
    if not sinais_tecnicos_desc: return out + "* N/A\n"

    indicadores_map = {
        'RSI_Desc': 'RSI', 'SMA_Desc': 'Médias Móveis', 'MACD_Desc': 'MACD',
        'BB_Desc': 'Bandas Bollinger', 'Volume_Desc': 'Volume',
        'Padroes_Vela': 'Padrões Vela (Último)' # <-- NOVO
    }
    has_signal = False
    for key, nome_fmt in indicadores_map.items():
        valor = sinais_tecnicos_desc.get(key)
        if valor:
            # Adiciona valor numérico do RSI
            if key == 'RSI_Desc' and 'RSI_Valor' in sinais_tecnicos_desc:
                valor_num = sinais_tecnicos_desc['RSI_Valor']
                out += f"*   **{nome_fmt}:** {valor} ({valor_num:.1f})\n"
                has_signal = True
            # Esconde Volume N/A para Forex
            elif key == 'Volume_Desc' and 'N/A' in valor and tipo_ativo == 'Forex':
                continue # Pula volume N/A em Forex
            # Formata a lista de Padrões de Vela
            elif key == 'Padroes_Vela' and isinstance(valor, list):
                 if valor: # Apenas se a lista não estiver vazia
                      out += f"*   **{nome_fmt}:** {', '.join(valor)}\n"
                      has_signal = True
            # Outros indicadores
            else:
                out += f"*   **{nome_fmt}:** {valor}\n"
                has_signal = True

    if not has_signal and not sinais_tecnicos_desc.get('Padroes_Vela'): # Verifica se realmente não houve nada
        out += "* Nenhum sinal técnico ou padrão de vela detectado.\n"
    elif not talib_available:
         out += "\n*   *(Análise de Padrões de Vela desativada - TA-Lib não instalada)*\n"

    return out

def formatar_sinal_principal(resultado_sinal, next_start_brt_str, next_end_brt_str, last_candle_brt_str, tipo_ativo):
    """Formata o Markdown para o sinal principal."""
    emoji_map = {"Forex": "🎯", "Cripto": "💎"}
    out = f"### {emoji_map.get(tipo_ativo, '🎯')} Sinal {tipo_ativo} Próxima Vela (BRT)\n"
    out += f"*(Vela: **{next_start_brt_str}** a **{next_end_brt_str}** BRT)*\n\n"
    sinal = resultado_sinal.get('sinal', 'ERRO'); prob = resultado_sinal.get('probabilidade', 0); razao = resultado_sinal.get('razao', 'N/A')
    cor_sinal = "gray"; emoji_sinal = "⏳"
    if sinal == "CALL": cor_sinal = "green"; emoji_sinal="⬆️"
    elif sinal == "PUT": cor_sinal = "red"; emoji_sinal="⬇️"

    out += f"**Sinal:** <span style='color:{cor_sinal}; font-weight:bold; font-size:1.3em;'>{emoji_sinal} {sinal}</span>\n"
    if sinal != "NEUTRO" and sinal != "ERRO": out += f"**Confiança Técnica:** {prob}%\n"
    else: out += f"**Confiança Técnica:** Indefinida\n"
    out += f"**Base Técnica:** {razao}\n\n"
    aviso_risco = "Mercado Forex é influenciado por muitos fatores." if tipo_ativo == 'Forex' else "Mercado Cripto é volátil."
    out += f"*<span style='color:orange;'>Aviso:</span> Baseado em vela das {last_candle_brt_str}. {aviso_risco} Alto risco.*"
    return out


# --- Funções de Análise Principais (Mantidas, chamam funções atualizadas) ---
# (analisar_ativo_forex e analisar_ativo_cripto como na v1.9)
def analisar_ativo_forex(ticker_selecionado, dias_historico_5m):
    """Análise Forex M5: Dados, Indicadores, Gráfico, Notícias Google Search Scraping, Sentimento, Sinal."""
    asset_type = 'Forex'
    yield_inicial = ( "Aguardando...", "Aguardando...", None, "Aguardando...", "Aguardando...", "Aguardando...", "Selecione e clique em Analisar Forex.")
    if not ticker_selecionado: return yield_inicial

    print(f"\n{'='*20} ANÁLISE FOREX M5: {ticker_selecionado} ({dias_historico_5m} dias) {'='*20}"); start_time = time.time()

    # Passo 1: Dados Técnicos
    yield ( "Carregando dados...", "Processando...", None, "Buscando notícias...", "Processando...", "Processando...", f"Analisando Forex: {ticker_selecionado}...")
    df_tech, erro_dados = obter_dados_tecnicos(ticker_selecionado, int(dias_historico_5m))
    if erro_dados: return erro_dados, "Erro dados.", None, "Erro dados.", "Erro dados.", "Erro dados.", f"Erro dados Forex: {ticker_selecionado}."

    # Passo 2: Indicadores (AGORA INCLUI PATTERNS)
    yield ( "Calculando indicadores...", "Processando...", None, "Buscando notícias...", "Processando...", "Processando...", f"Analisando Forex: {ticker_selecionado}...")
    df_com_ta, sinais_tecnicos_desc, erro_calculo = calcular_indicadores_tecnicos(df_tech)
    if erro_calculo: return erro_calculo, "Erro ind.", None, "Erro ind.", "Erro ind.", "Erro ind.", f"Erro indicadores Forex: {ticker_selecionado}."
    if df_com_ta is None or df_com_ta.empty: return "Erro: Sem dados pós cálculo.", "Erro dados.", None, "Erro dados.", "Erro dados.", "Erro dados.", f"Erro dados Forex: {ticker_selecionado}."

    # Passo 3: Horários BRT
    try:
        last_candle_time_original = df_com_ta.index[-1]
        _, next_start_brt_str, next_end_brt_str, last_candle_brt_str = converter_horarios_brt(last_candle_time_original)
    except Exception as e_tz: return f"Erro TZ: {e_tz}", "Erro TZ.", None, "Erro TZ.", "Erro TZ.", "Erro TZ.", f"Erro Timezone: {e_tz}"

    # Passo 4: Notícias Google Search Scraping & Sentimento
    out_sinais_desc = formatar_sinais_descritivos(sinais_tecnicos_desc, last_candle_brt_str, asset_type) # Agora inclui padrões
    yield ( out_sinais_desc, "Gerando gráfico...", None, "Scraping Google Search...", "Scraping Google Search...", "Analisando sentimento...", f"Analisando Forex: {ticker_selecionado}...")
    out_noticias_especificas, out_noticias_gerais, out_sentimento = obter_noticias_google_search_scraping(ticker_selecionado, asset_type)

    # Passo 5: Gráfico
    yield ( out_sinais_desc, "Gerando gráfico...", None, out_noticias_especificas, out_noticias_gerais, out_sentimento, f"Gerando gráfico para {ticker_selecionado}...")
    figura_grafico = gerar_grafico_ativo(df_com_ta, ticker_selecionado)
    if figura_grafico is None:
         return out_sinais_desc, "Erro gráfico.", None, out_noticias_especificas, out_noticias_gerais, out_sentimento, f"Erro gráfico {ticker_selecionado}."

    # Passo 6: Sinal (AGORA CONSIDERA PATTERNS)
    yield ( out_sinais_desc, "Gerando Sinal 5m...", figura_grafico, out_noticias_especificas, out_noticias_gerais, out_sentimento, f"Gerando sinal para {ticker_selecionado}...")
    resultado_sinal = gerar_sinal_binario_5m(ticker_selecionado, sinais_tecnicos_desc, df_com_ta, asset_type='forex')
    out_sinal_principal = formatar_sinal_principal(resultado_sinal, next_start_brt_str, next_end_brt_str, last_candle_brt_str, asset_type)

    # Passo 7: Finalizar
    tempo_total = time.time() - start_time
    out_status = f"Análise Forex M5 ({ticker_selecionado}) concluída em {tempo_total:.2f}s."
    print(out_status)
    yield out_sinais_desc, out_sinal_principal, figura_grafico, out_noticias_especificas, out_noticias_gerais, out_sentimento, out_status


def analisar_ativo_cripto(ticker_selecionado, dias_historico_5m):
    """Análise Cripto M5: Dados, Indicadores, Gráfico, Notícias Google Search Scraping, Sentimento, Sinal."""
    asset_type = 'Cripto'
    yield_inicial = ( "Aguardando...", "Aguardando...", None, "Aguardando...", "Aguardando...", "Aguardando...", "Selecione e clique em Analisar Cripto.")
    if not ticker_selecionado: return yield_inicial

    print(f"\n{'='*20} ANÁLISE CRIPTO M5: {ticker_selecionado} ({dias_historico_5m} dias) {'='*20}"); start_time = time.time()

    # Passo 1: Dados Técnicos
    yield ( "Carregando dados...", "Processando...", None, "Buscando notícias...", "Processando...", "Processando...", f"Analisando Cripto: {ticker_selecionado}...")
    df_tech, erro_dados = obter_dados_tecnicos(ticker_selecionado, int(dias_historico_5m))
    if erro_dados: return erro_dados, "Erro dados.", None, "Erro dados.", "Erro dados.", "Erro dados.", f"Erro dados Cripto: {ticker_selecionado}."

    # Passo 2: Indicadores (AGORA INCLUI PATTERNS)
    yield ( "Calculando indicadores...", "Processando...", None, "Buscando notícias...", "Processando...", "Processando...", f"Analisando Cripto: {ticker_selecionado}...")
    df_com_ta, sinais_tecnicos_desc, erro_calculo = calcular_indicadores_tecnicos(df_tech)
    if erro_calculo: return erro_calculo, "Erro ind.", None, "Erro ind.", "Erro ind.", "Erro ind.", f"Erro indicadores Cripto: {ticker_selecionado}."
    if df_com_ta is None or df_com_ta.empty: return "Erro: Sem dados pós cálculo.", "Erro dados.", None, "Erro dados.", "Erro dados.", "Erro dados.", f"Erro dados Cripto: {ticker_selecionado}."

    # Passo 3: Horários BRT
    try:
        last_candle_time_original = df_com_ta.index[-1]
        _, next_start_brt_str, next_end_brt_str, last_candle_brt_str = converter_horarios_brt(last_candle_time_original)
    except Exception as e_tz: return f"Erro TZ: {e_tz}", "Erro TZ.", None, "Erro TZ.", "Erro TZ.", "Erro TZ.", f"Erro Timezone: {e_tz}"

    # Passo 4: Notícias Google Search Scraping & Sentimento
    out_sinais_desc = formatar_sinais_descritivos(sinais_tecnicos_desc, last_candle_brt_str, asset_type) # Agora inclui padrões
    yield ( out_sinais_desc, "Gerando gráfico...", None, "Scraping Google Search...", "Scraping Google Search...", "Analisando sentimento...", f"Analisando Cripto: {ticker_selecionado}...")
    out_noticias_especificas, out_noticias_gerais, out_sentimento = obter_noticias_google_search_scraping(ticker_selecionado, asset_type)

    # Passo 5: Gráfico
    yield ( out_sinais_desc, "Gerando gráfico...", None, out_noticias_especificas, out_noticias_gerais, out_sentimento, f"Gerando gráfico para {ticker_selecionado}...")
    figura_grafico = gerar_grafico_ativo(df_com_ta, ticker_selecionado)
    if figura_grafico is None:
         return out_sinais_desc, "Erro gráfico.", None, out_noticias_especificas, out_noticias_gerais, out_sentimento, f"Erro gráfico {ticker_selecionado}."

    # Passo 6: Sinal (AGORA CONSIDERA PATTERNS)
    yield ( out_sinais_desc, "Gerando Sinal 5m (c/ vol)...", figura_grafico, out_noticias_especificas, out_noticias_gerais, out_sentimento, f"Gerando sinal para {ticker_selecionado}...")
    resultado_sinal = gerar_sinal_binario_5m(ticker_selecionado, sinais_tecnicos_desc, df_com_ta, asset_type='crypto')
    out_sinal_principal = formatar_sinal_principal(resultado_sinal, next_start_brt_str, next_end_brt_str, last_candle_brt_str, asset_type)

    # Passo 7: Finalizar
    tempo_total = time.time() - start_time
    out_status = f"Análise Cripto M5 ({ticker_selecionado}) concluída em {tempo_total:.2f}s."
    print(out_status)
    yield out_sinais_desc, out_sinal_principal, figura_grafico, out_noticias_especificas, out_noticias_gerais, out_sentimento, out_status


# --- Interface Gradio (v1.9.1 - Com Padrões de Vela e Correção de Syntax) ---
print("Configurando a interface Gradio v1.9.1 (Padrões Vela + Correção)...")
# (Interface Gradio mantida exatamente como na v1.9)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"), title="Analista M5 v1.9.1 (Padrões Vela - BRT)") as interface_gradio:
    gr.Markdown(
        f"""
        # 🤖 Analista M5 v1.9.1 (Forex & Cripto c/ Padrões de Vela - BRT)
        **Atenção:** Ferramenta **EXPERIMENTAL** e **EDUCACIONAL**. **NÃO É RECOMENDAÇÃO FINANCEIRA.**
        Operações M5 são **EXTREMAMENTE ARRISCADAS**. Use com cautela.
        Horários em BRT. Notícias via Scraping Google Search (FRÁGIL!).
        **Padrões de Vela:** {'ATIVADOS (TA-Lib encontrada)' if talib_available else 'DESATIVADOS (TA-Lib NÃO encontrada - veja console)'}
        """
    )

    with gr.Tabs(): # Abas Principais
        # --- ABA FOREX ---
        with gr.TabItem("🪙 Forex M5 (BRT)"):
            with gr.Row():
                input_ticker_forex = gr.Dropdown(label="Par Forex", choices=LISTA_FOREX, value=LISTA_FOREX[0] if LISTA_FOREX else None, filterable=True, scale=3)
                input_dias_5m_forex = gr.Slider(label="Dias Histórico (5min)", minimum=1, maximum=MAX_LOOKBACK_DAYS_5M, value=DEFAULT_LOOKBACK_DAYS_5M, step=1, scale=2)
                botao_analisar_forex = gr.Button("🎯 Analisar Forex (BRT)", variant="primary", scale=1)
            output_status_forex = gr.Textbox(label="Status Análise Forex", interactive=False, placeholder="Aguardando análise...")
            with gr.Tabs(): # Sub-Abas Forex
                with gr.TabItem("🚨 Sinal"): output_sinal_principal_forex = gr.Markdown()
                with gr.TabItem("💹 Gráfico"): output_grafico_forex = gr.Plot()
                with gr.TabItem("📈 Indicadores"): output_sinais_desc_forex = gr.Markdown() # Irá mostrar os padrões
                with gr.TabItem("📰 Notícias & Sent."):
                    output_noticias_especificas_forex = gr.Markdown(label="Notícias Específicas (Google Search - Semana)")
                    gr.Markdown("---")
                    output_noticias_gerais_forex = gr.Markdown(label="Notícias Gerais (Google Search - Semana)")
                    gr.Markdown("---")
                    output_sentimento_forex = gr.Markdown(label="Sentimento (Baseado nos Títulos)")
                with gr.TabItem("🌍 Fatores Mercado"):
                      output_fatores_mercado_forex = gr.Markdown(value=FATORES_MERCADO_FOREX_MD)
            # Conexão Botão Forex (7 outputs)
            botao_analisar_forex.click(
                fn=analisar_ativo_forex,
                inputs=[input_ticker_forex, input_dias_5m_forex],
                outputs=[ output_sinais_desc_forex, output_sinal_principal_forex, output_grafico_forex,
                          output_noticias_especificas_forex, output_noticias_gerais_forex,
                          output_sentimento_forex, output_status_forex ]
            )
            gr.Markdown( "*Indicadores: RSI(14), SMA(20/50), MACD(12,26,9), BBands(20,2), **Padrões de Vela (TA-Lib)**. Vol. não usado no sinal.*" )

        # --- ABA CRIPTO ---
        with gr.TabItem("💎 Cripto M5 (BRT)"):
            with gr.Row():
                input_ticker_cripto = gr.Dropdown(label="Ativo Cripto", choices=LISTA_CRIPTO, value=LISTA_CRIPTO[0] if LISTA_CRIPTO else None, filterable=True, scale=3)
                input_dias_5m_cripto = gr.Slider(label="Dias Histórico (5min)", minimum=1, maximum=MAX_LOOKBACK_DAYS_5M, value=DEFAULT_LOOKBACK_DAYS_5M, step=1, scale=2)
                botao_analisar_cripto = gr.Button("🎯 Analisar Cripto (BRT)", variant="primary", scale=1)
            output_status_cripto = gr.Textbox(label="Status Análise Cripto", interactive=False, placeholder="Aguardando análise...")
            with gr.Tabs(): # Sub-Abas Cripto
                with gr.TabItem("🚨 Sinal"): output_sinal_principal_cripto = gr.Markdown()
                with gr.TabItem("💹 Gráfico"): output_grafico_cripto = gr.Plot()
                with gr.TabItem("📈 Indicadores"): output_sinais_desc_cripto = gr.Markdown() # Irá mostrar os padrões
                with gr.TabItem("📰 Notícias & Sent."):
                    output_noticias_especificas_cripto = gr.Markdown(label="Notícias Específicas (Google Search - Semana)")
                    gr.Markdown("---")
                    output_noticias_gerais_cripto = gr.Markdown(label="Notícias Gerais (Google Search - Semana)")
                    gr.Markdown("---")
                    output_sentimento_cripto = gr.Markdown(label="Sentimento (Baseado nos Títulos)")
                with gr.TabItem("🌍 Fatores Mercado"):
                      output_fatores_mercado_cripto = gr.Markdown(value=FATORES_MERCADO_CRIPTO_MD)
            # Conexão Botão Cripto (7 outputs)
            botao_analisar_cripto.click(
                fn=analisar_ativo_cripto,
                inputs=[input_ticker_cripto, input_dias_5m_cripto],
                outputs=[ output_sinais_desc_cripto, output_sinal_principal_cripto, output_grafico_cripto,
                          output_noticias_especificas_cripto, output_noticias_gerais_cripto,
                          output_sentimento_cripto, output_status_cripto ]
            )
            gr.Markdown( "*Indicadores: RSI(14), SMA(20/50), MACD(12,26,9), BBands(20,2), **Volume (vs MA(20))**, **Padrões de Vela (TA-Lib)**. *" )

# --- Iniciar Aplicação ---
if __name__ == "__main__":
    print("Iniciando a interface Gradio v1.9.1...")
    print("Acesse localmente ou pelo link público (se gerado).")
    try:
        interface_gradio.launch(server_name="0.0.0.0")
    except Exception as e:
        print(f"Erro ao iniciar o Gradio: {e}")
        print("Verifique se a porta 7860 está livre.")
    print("Interface encerrada.")
