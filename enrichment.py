import pandas as pd
import numpy as np
import yaml

# --- 1. Helper Functions ---
def _get_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

# --- 2. Feature Implementations (Templates) ---

# 📈 Trend Indicators
def feat_sma(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    return g[on].rolling(window, min_periods=window).mean()

def feat_ema(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    return g[on].ewm(span=window, adjust=False, min_periods=window).mean()

def feat_macd(g: pd.DataFrame, on: str, fast_period: int, slow_period: int, signal_period: int) -> pd.DataFrame:
    ema_fast = g[on].ewm(span=fast_period, adjust=False).mean()
    ema_slow = g[on].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    # Return a DataFrame instead of a dict
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "hist": macd_line - signal_line
    })

# 🏃 Momentum Indicators
def feat_rsi(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    delta = g[on].diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def feat_stoch(g: pd.DataFrame, high: str, low: str, close: str, k_window: int, d_window: int) -> pd.DataFrame:
    low_k = g[low].rolling(k_window).min()
    high_k = g[high].rolling(k_window).max()
    k_line = 100 * ((g[close] - low_k) / (high_k - low_k).replace(0, np.nan))
    d_line = k_line.rolling(d_window).mean()
    # Return a DataFrame instead of a dict
    return pd.DataFrame({"stoch_k": k_line, "stoch_d": d_line})

# 🌊 Volatility Indicators
def feat_rolling_vol(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    return g[on].rolling(window, min_periods=window).std()

def feat_atr(g: pd.DataFrame, high: str, low: str, close: str, window: int) -> pd.Series:
    true_range = _get_true_range(g[high], g[low], g[close])
    return true_range.ewm(span=window, adjust=False).mean()

def feat_bbands(g: pd.DataFrame, on: str, window: int, std_dev: int) -> pd.DataFrame:
    middle_band = g[on].rolling(window).mean()
    std = g[on].rolling(window).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    # Return a DataFrame instead of a dict
    return pd.DataFrame({
        "bband_upper": upper_band,
        "bband_middle": middle_band,
        "bband_lower": lower_band
    })

#  Volume Indicators
def feat_obv(g: pd.DataFrame, close: str, volume: str) -> pd.Series:
    signed_vol = g[volume] * np.sign(g[close].diff()).fillna(0)
    return signed_vol.cumsum()

#  Basic Transformations & Statistics
def feat_ret(g: pd.DataFrame, on: str, periods: int, method: str) -> pd.Series:
    if method == "log":
        return np.log(g[on] / g[on].shift(periods))
    return g[on].pct_change(periods)

def feat_lag(g: pd.DataFrame, on: str, periods: int) -> pd.Series:
    return g[on].shift(periods)

def feat_diff(g: pd.DataFrame, on: str, periods: int) -> pd.Series:
    return g[on].diff(periods)

def feat_rolling_max(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    return g[on].rolling(window).max()

def feat_rolling_min(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    return g[on].rolling(window).min()

def feat_zscore(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    rolling_mean = g[on].rolling(window).mean()
    rolling_std = g[on].rolling(window).std()
    return (g[on] - rolling_mean) / rolling_std.replace(0, np.nan)

#  Calendar Features
def feat_session_flags(g: pd.DataFrame) -> pd.DataFrame:
    ts = g['ts']
    # Return a DataFrame instead of a dict
    return pd.DataFrame({
        "dow": ts.dt.dayofweek,
        "month": ts.dt.month,
        "week": ts.dt.isocalendar().week,
        "hour": ts.dt.hour,
        "is_month_start": ts.dt.is_month_start.astype("int8"),
        "is_month_end": ts.dt.is_month_end.astype("int8"),
    })

# dispatcher
FEATURE_IMPLEMENTATIONS = {
    "sma": feat_sma, "ema": feat_ema, "macd": feat_macd,
    "rsi": feat_rsi, "stoch": feat_stoch,
    "rolling_vol": feat_rolling_vol, "atr": feat_atr, "bbands": feat_bbands,
    "obv": feat_obv,
    "ret": feat_ret, "lag": feat_lag, "diff": feat_diff,
    "rolling_max": feat_rolling_max, "rolling_min": feat_rolling_min, "zscore": feat_zscore,
    "session_flags": feat_session_flags,
}

#  The Main Executor 
def apply_features(df: pd.DataFrame, dsl: dict, registry: dict) -> pd.DataFrame:

    #Applies features to a DataFrame based on a DSL recipe.

    if not {"ticker", "ts"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'ticker' and 'ts' columns.")

    df_enriched = df.sort_values(["ticker", "ts"]).copy()
    all_new_cols = []
    
    for request in dsl.get("features", []):
        name = request["name"]
        user_params = request.get("params", {})
        impl_func = FEATURE_IMPLEMENTATIONS.get(name)

        final_params = {}
        registry_params = registry["features"][name].get("params", {})
        for p_name, p_rules in registry_params.items():
            if "default" in p_rules:
                final_params[p_name] = p_rules["default"]
        final_params.update(user_params)

        # Direct Calculation per Group
        result_list = [impl_func(group_df, **final_params) for _, group_df in df_enriched.groupby("ticker")]
        full_result = pd.concat(result_list)

        # Assign Results
        if isinstance(full_result, pd.DataFrame):
            for col in full_result.columns:
                output_col_name = f"{name}_{col}"
                all_new_cols.append(full_result[[col]].rename(columns={col: output_col_name}))
        else:
            output_col_name = request.get("as", f"{name}_{final_params.get('on', '')}_{final_params.get('window', '')}".rstrip('_'))
            all_new_cols.append(full_result.rename(output_col_name))
    
    # Combine original df with all new feature columns at once
    if all_new_cols:
        df_final = pd.concat([df_enriched] + all_new_cols, axis=1)
        return df_final.copy()
    
    return df_enriched.copy()
