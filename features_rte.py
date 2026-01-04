import pandas as pd
import numpy as np


def _parse_hour_range_to_end_hour(hour_range: str) -> int:
    # example: "00:00-01:00" -> end hour 1
    try:
        end = hour_range.split("-")[1]
        hh = int(end.split(":")[0])
        if hh == 24:
            return 0
        return hh
    except Exception:
        return 0

def build_hourly_prod_timeseries(df_prod: pd.DataFrame) -> pd.DataFrame:
    df = df_prod.copy()
    if "date" not in df.columns or "Heures" not in df.columns:
        raise ValueError("Production df must contain columns: date, Heures")

    # Create a timestamp: use date + start hour from range
    start_hour = df["Heures"].str.slice(0, 2).astype(int, errors="ignore")
    start_hour = pd.to_numeric(df["Heures"].str.slice(0, 2), errors="coerce").fillna(0).astype(int)
    df["ts"] = pd.to_datetime(df["date"]) + pd.to_timedelta(start_hour, unit="h")
    df = df.sort_values("ts").reset_index(drop=True)
    return df

def build_15min_demand_timeseries(df_cons: pd.DataFrame) -> pd.DataFrame:
    df = df_cons.copy()
    if "date" not in df.columns or "Heures" not in df.columns:
        raise ValueError("Consumption df must contain columns: date, Heures")

    # Heures like "17:15"
    df["ts"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["Heures"] + ":00")
    df = df.sort_values("ts").reset_index(drop=True)
    return df

def compute_renewable_share(df_hourly_prod: pd.DataFrame) -> pd.DataFrame:
    df = df_hourly_prod.copy()

    # Try to detect renewable columns based on your headers
    possible_renewables = [
        "Solaire", "Éolien terrestre", "Éolien en mer",
        "Hydraulique STEP", "Hydraulique fil de l'eau / éclusée", "Hydraulique lacs",
        "Biomasse", "Déchets"
    ]
    renew_cols = [c for c in possible_renewables if c in df.columns]
    if "Total" not in df.columns:
        raise ValueError("Production df must contain 'Total'")

    df["renewable_mw"] = df[renew_cols].sum(axis=1, skipna=True) if renew_cols else np.nan
    df["renewable_share"] = np.where(df["Total"] > 0, df["renewable_mw"] / df["Total"], np.nan)

    # Fossil proxy
    fossil_cols = [c for c in ["Gaz", "Charbon", "Fioul"] if c in df.columns]
    df["fossil_mw"] = df[fossil_cols].sum(axis=1, skipna=True) if fossil_cols else np.nan
    df["fossil_share"] = np.where(df["Total"] > 0, df["fossil_mw"] / df["Total"], np.nan)

    return df

def join_prod_demand(df_prod_hourly: pd.DataFrame, df_demand_15m: pd.DataFrame) -> pd.DataFrame:
    # resample production to 15-min by forward fill
    prod_15 = df_prod_hourly.set_index("ts").resample("15min").ffill().reset_index()
    dem = df_demand_15m.copy()
    out = dem.merge(prod_15[["ts", "renewable_share", "fossil_share", "Total"]], on="ts", how="left")
    out = out.sort_values("ts").reset_index(drop=True)

    return out

def surplus_proxy(df_joined_15m: pd.DataFrame, top_green_pct: float = 0.15) -> pd.DataFrame:
    """
    Surplus proxy:
    - green windows = top X% renewable_share
    - surplus score higher if green AND demand low (below median)
    """
    df = df_joined_15m.copy()
    if "Consommation" not in df.columns:
        raise ValueError("Joined df must have Consommation")

    thr = df["renewable_share"].quantile(1 - top_green_pct)
    demand_med = df["Consommation"].median()

    df["is_green_window"] = df["renewable_share"] >= thr
    df["is_low_demand"] = df["Consommation"] <= demand_med

    # 0..2 simple score
    df["surplus_score"] = df["is_green_window"].astype(int) + df["is_low_demand"].astype(int)
    return df

EMISSION_FACTORS_GCO2_PER_KWH = {
    "Charbon": 900,
    "Gaz": 400,
    "Fioul": 700,
    "Déchets": 300,
    "Biomasse": 100,
    "Nucléaire": 6,
    "Solaire": 40,
    "Éolien terrestre": 12,
    "Éolien en mer": 12,
    "Hydraulique STEP": 10,
    "Hydraulique fil de l'eau / éclusée": 10,
    "Hydraulique lacs": 10,
}

def compute_modelled_carbon_intensity(df_prod_hourly):
    """
    Returns df with gco2_per_kwh_model computed from production mix shares.
    Uses df['Total'] as denominator and EMISSION_FACTORS_GCO2_PER_KWH mapping.
    """
    df = df_prod_hourly.copy()
    if "Total" not in df.columns:
        raise ValueError("Production dataframe must contain column 'Total'.")

    total = df["Total"].replace(0, np.nan)

    cols = [c for c in EMISSION_FACTORS_GCO2_PER_KWH.keys() if c in df.columns]
    if not cols:
        df["gco2_per_kwh_model"] = np.nan
        return df

    intensity = np.zeros(len(df), dtype=float)
    for c in cols:
        intensity += (df[c] / total) * EMISSION_FACTORS_GCO2_PER_KWH[c]

    df["gco2_per_kwh_model"] = intensity
    return df
