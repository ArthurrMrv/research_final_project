import numpy as np
import pandas as pd

def make_synthetic_sncf_load(ts_index: pd.DatetimeIndex, annual_mwh: float = 1_000_000.0) -> pd.Series:
    """
    Synthetic but realistic-ish: weekday higher, night lower, smooth day shape.
    Scales to annual_mwh.
    """
    ts = pd.DatetimeIndex(ts_index)
    hour = ts.hour.values
    dow = ts.dayofweek.values  # 0=Mon

    # day shape: peak midday/early evening, low night
    day_shape = (
        0.35
        + 0.25 * np.exp(-((hour - 12) / 4.5) ** 2)
        + 0.20 * np.exp(-((hour - 18) / 3.5) ** 2)
    )

    # weekends slightly lower
    weekend = (dow >= 5).astype(float)
    shape = day_shape * (1.0 - 0.12 * weekend)

    # normalize to sum = 1
    shape = shape / shape.sum()
    mwh = annual_mwh * shape
    return pd.Series(mwh, index=ts, name="sncf_mwh")

def compute_consumption_weighted_ci(load_mwh: pd.Series, gco2_per_kwh: pd.Series) -> float:
    """
    Returns average carbon intensity (gCO2/kWh) weighted by consumption.
    """
    load = load_mwh.to_numpy(dtype=float)
    ci = gco2_per_kwh.fillna(gco2_per_kwh.median()).to_numpy(dtype=float)
    denom = load.sum()
    if denom <= 0:
        return float("nan")
    return float((load * ci).sum() / denom)

def compute_total_emissions_tco2(load_mwh: pd.Series, gco2_per_kwh: pd.Series) -> float:
    """
    tCO2 = sum( MWh * 1000 kWh/MWh * gCO2/kWh ) / 1e6
    """
    load = load_mwh.to_numpy(dtype=float)
    ci = gco2_per_kwh.fillna(gco2_per_kwh.median()).to_numpy(dtype=float)
    gco2 = (load * 1000.0) * ci
    return float(gco2.sum() / 1e6)

def build_objective_score(df: pd.DataFrame, mode: str = "co2") -> np.ndarray:
    """
    Higher score = better to consume at that timestep.
    mode:
      - "co2": prefer lower carbon intensity
      - "green": prefer higher renewable share (if present)
    """
    if mode == "co2" and "gco2_per_kwh_model" in df.columns:
        ci = df["gco2_per_kwh_model"].to_numpy(dtype=float)
        # invert so lower CI => higher score
        return -np.nan_to_num(ci, nan=np.nanmedian(ci))
    elif mode == "green" and "renewable_share" in df.columns:
        r = df["renewable_share"].to_numpy(dtype=float)
        return np.nan_to_num(r, nan=np.nanmedian(r))
    else:
        # fallback: prefer low fossil share if present
        if "fossil_share" in df.columns:
            f = df["fossil_share"].to_numpy(dtype=float)
            return -np.nan_to_num(f, nan=np.nanmedian(f))
        return np.zeros(len(df), dtype=float)

def shift_load_constrained(
    df: pd.DataFrame,
    load_col: str,
    score: np.ndarray,
    shiftable_share: float,
    window_steps: int,
    freq_minutes: int,
    cap_multiplier: float = 1.20,
    no_shift_hours: tuple[int, int] | None = None,
    ramp_limit_multiplier: float | None = None,
    distance_penalty: float = 0.0,
) -> pd.DataFrame:
    """
    Client-grade constrained shifter:

    - Only a fraction of each timestep load can move: shiftable_share
    - Can move within +/- window_steps
    - Destination cannot exceed cap_multiplier * baseline_load[t]
    - Optionally forbid shifting INTO certain hours (e.g. peak hours)
    - Optionally limit ramp: |opt[t]-opt[t-1]| <= ramp_limit_multiplier * baseline[t]
    - Optionally penalize long shifts (distance_penalty > 0)

    Returns df with:
      - load_opt
      - shifted_mwh_total
      - avg_shift_minutes (approx)
      - cap_violations (should be 0)
      - shift_matrix_hour (hour->hour table components)
    """
    out = df.copy().sort_values("ts").reset_index(drop=True)

    load = out[load_col].to_numpy(dtype=float)
    n = len(load)

    # movable energy and fixed energy
    movable = load * shiftable_share
    fixed = load - movable

    # destination headroom: cap - current fixed - remaining movable (initially 0 assigned)
    cap = cap_multiplier * load  # timestep capacity relative to baseline at that timestep
    assigned = np.zeros(n, dtype=float)  # movable energy assigned to timesteps

    # feasibility mask for destination hours
    dest_allowed = np.ones(n, dtype=bool)
    if no_shift_hours is not None:
        start_h, end_h = no_shift_hours  # e.g. (7, 9) blocks hours 7-8
        hours = pd.to_datetime(out["ts"]).dt.hour.to_numpy()
        block = (hours >= start_h) & (hours < end_h)
        dest_allowed = ~block

    # Sort sources from "worst" to "best" by score (lowest score are worst)
    src_order = np.argsort(score)

    # For reporting: approximate shift distances and from/to hours
    shift_minutes_accum = 0.0
    shift_energy_accum = 0.0

    from_hour = pd.to_datetime(out["ts"]).dt.hour.to_numpy()
    to_hour_accum = np.zeros((24, 24), dtype=float)

    # Precompute candidate lists for each index (within window) sorted by best adjusted score
    # adjusted_score = score - distance_penalty*|dt|
    idx = np.arange(n)
    candidates_sorted = []
    for i in range(n):
        lo = max(0, i - window_steps)
        hi = min(n - 1, i + window_steps)
        cand = idx[lo:hi+1]
        if distance_penalty > 0:
            dist = np.abs(cand - i).astype(float)
            adj = score[cand] - distance_penalty * dist
        else:
            adj = score[cand]
        order = cand[np.argsort(adj)[::-1]]  # best first
        candidates_sorted.append(order)

    # Allocate movable energy from worst slots into best feasible slots
    remaining = movable.copy()
    for i in src_order:
        e = remaining[i]
        if e <= 0:
            continue

        # if source hour is not allowed to move OUT? we still allow moving out; constraint is for destination by default
        for j in candidates_sorted[i]:
            if j == i:
                continue
            if not dest_allowed[j]:
                continue

            # only move if destination is better than source
            if score[j] <= score[i]:
                break

            # headroom at destination
            headroom = cap[j] - (fixed[j] + assigned[j])
            if headroom <= 0:
                continue

            move = min(e, headroom)
            if move <= 0:
                continue

            assigned[j] += move
            e -= move

            # reporting shift distance
            shift_minutes = abs(j - i) * freq_minutes
            shift_minutes_accum += move * shift_minutes
            shift_energy_accum += move

            to_hour_accum[from_hour[i], from_hour[j]] += move

            if e <= 1e-12:
                break

        remaining[i] = e  # unshifted part stays at source

    # Construct optimized load
    opt = fixed + remaining + assigned

    # Optional ramp limit pass (simple smoothing, greedy)
    cap_violations = int(np.sum(opt > cap + 1e-9))
    if ramp_limit_multiplier is not None:
        # enforce |opt[t]-opt[t-1]| <= ramp_limit_multiplier*baseline[t]
        lim = ramp_limit_multiplier * load
        opt2 = opt.copy()
        for t in range(1, n):
            max_up = opt2[t-1] + lim[t]
            max_dn = max(0.0, opt2[t-1] - lim[t])
            if opt2[t] > max_up:
                opt2[t] = max_up
            elif opt2[t] < max_dn:
                opt2[t] = max_dn
        opt = opt2
        cap_violations = int(np.sum(opt > cap + 1e-9))

    out[load_col + "_opt"] = opt

    # shifted energy (MWh) = half L1 difference
    shifted_mwh_total = float(np.abs(out[load_col + "_opt"] - out[load_col]).sum() / 2.0)

    avg_shift_minutes = float(shift_minutes_accum / shift_energy_accum) if shift_energy_accum > 0 else 0.0

    out.attrs["shifted_mwh_total"] = shifted_mwh_total
    out.attrs["avg_shift_minutes"] = avg_shift_minutes
    out.attrs["cap_violations"] = cap_violations
    out.attrs["to_hour_matrix"] = to_hour_accum

    return out
