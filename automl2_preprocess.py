import os
import pandas as pd
from tqdm import tqdm
import ta  # technical analysis library
import numpy as np

INPUT_DIR = "data_min10_262800"
OUTPUT_DIR = "G:/hacking2_data_min10_262800"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_time_features_infer_dates(df: pd.DataFrame, last_date_str="2025-08-21 23:50") -> pd.DataFrame:
    """
    CSV 마지막 날짜를 기준으로 날짜 역추적 후 정수형 시간 feature 추가
    - last_date_str: CSV 마지막 행 날짜 (YYYY-MM-DD HH:MM)
    - 데이터 간격: 4시간
    """
    n = len(df)
    last_date = pd.to_datetime(last_date_str)
    
    # 마지막 행부터 4시간 간격으로 역순 날짜 생성
    dates = pd.date_range(end=last_date, periods=n, freq="10min")
    df["date"] = dates

    # 요일 / 월일 / 연중일 / 하루 시간(시)
    df["day_of_week"] = df["date"].dt.weekday
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["hour_of_day"] = df["date"].dt.hour
    df["minute_of_hour"] = df["date"].dt.minute
    return df

def convert_to_day(df: pd.DataFrame) -> pd.DataFrame:
    open_lst = []
    high_lst = []
    low_lst = []
    close_lst = []
    volume_lst = []
    value_lst = []
    for row_idx in range(len(df)):
        open_lst.append(df.loc[row_idx, "open"])
        high_lst.append(df.loc[row_idx:min(row_idx+24*6, len(df)), "high"].max())
        low_lst.append(df.loc[row_idx:min(row_idx+24*6, len(df)), "low"].min())
        close_lst.append(df.loc[min(row_idx+24*6, len(df))-1, "close"])
        volume_lst.append(df.loc[row_idx:min(row_idx+24*6, len(df)), "volume"].sum())
        value_lst.append(df.loc[row_idx:min(row_idx+24*6, len(df)), "value"].mean())
    df["open"] = open_lst
    df["high"] = high_lst
    df["low"] = low_lst
    df["close"] = close_lst
    df["volume"] = volume_lst
    df["value"] = value_lst
    return df

def decomposition(df: pd.DataFrame):
    # 1. '시:분' 단위로 그룹핑
    grouped = df.groupby(['hour_of_day', 'minute_of_hour'])
    # 2. 그룹별 DataFrame을 리스트로 변환
    df_list = [group for _, group in grouped]
    return df_list

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 이동평균 (윈도우 ×6)
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma100"] = df["close"].rolling(100).mean()
    
    # RSI (window 14 → 84)
    df["rsi14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    
    # MACD (EMA 12/26/9 → 72/156/54)
    macd = ta.trend.MACD(df["close"], window_slow=16, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    
    # Bollinger Bands (window 20 → 120)
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()
    
    # Stochastic Oscillator (window 14 → 84, smooth_window 3 → 18)
    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"], window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    
    # ATR (window 14 → 84)
    df["atr84"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    # 결측치 보간
    df.interpolate(method="linear", inplace=True)
    df = df.ffill().bfill()
    return df


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    labels1 = []
    labels2 = []
    n = len(df)
    for i in range(n):
        if i + 6*24+1 <= n:
            base_open = df.loc[i+1, "open"]
            label1 = 0
            label2 = 0
            for j in range(i+1, i+6*24+1):
                if df.loc[j, "low"] <= base_open * 0.99:
                    break
                if df.loc[j, "high"] >= base_open * 1.02:
                    label1 = 1
                if df.loc[j, "high"] >= base_open * 1.04:
                    label2 = 1
                    break
            labels1.append(label1)
            labels2.append(label2)
        else:
            labels1.append(np.nan)
            labels2.append(np.nan)
    df["label1"] = labels1
    df["label2"] = labels2
    return df


# 전체 파일 처리
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
for file in tqdm(files, desc="Processing CSV files"):
    path = os.path.join(INPUT_DIR, file)
    df = pd.read_csv(path)
    if len(df) < 600 * 24:  # 최소 window 길이에 맞게 조건 강화
        continue

    # 라벨 생성
    df = create_label(df)
    
    # 컬럼명 통일 (소문자 강제)
    df.columns = [c.lower() for c in df.columns]
    
    # 시간 feature 추가
    df = add_time_features_infer_dates(df)

    df = convert_to_day(df)

    df_lst = decomposition(df)

    new_df_lst = []
    for df in df_lst:
        df = add_indicators(df)
        new_df_lst.append(df)
    df = pd.concat(new_df_lst, ignore_index=True)
    df = df.sort_values(by="date")
    # 저장
    save_path = os.path.join(OUTPUT_DIR, file)
    df.to_csv(save_path, index=False)
