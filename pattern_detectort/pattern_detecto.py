from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


class PatternDirector:
    """
    One-stop 'present-focused' pattern detector for stocks.
    - Candlesticks: Hammer, Shooting Star, Doji, Bullish/Bearish Engulfing
    - Chart patterns: Cup & Handle, Double Bottom, Head & Shoulders
    - Prints actionable plan (entry/stop/target) + 'cancel now' logic
    """

    @dataclass
    class Plan:
        pattern: str
        side: str           # 'bull' | 'bear' | 'neutral'
        state: str          # 'CANDLE' | 'PRE_BREAKOUT' | 'BREAKOUT'
        date: pd.Timestamp
        entry: Optional[float]
        stop: Optional[float]
        target: Optional[float]
        cancel_now: bool
        status: str         # 'VALID' | 'PENDING' | 'CANCELED NOW'
        notes: str = ""

    def __init__(self,
                 config: Optional[Dict] = None,
                 config_path: Optional[str | Path] = "pattern_detectort\stocks.json"):
        """
        config can be a dict like:
          {
            "tickers": ["AAPL", ...],
            "period": "1y",
            "interval": "1d",
            "risk": {"atr_mult": 1.0, "percent_buffer": 0.0025}
          }
        If config is None, reads config_path JSON.
        """
        if config is None:
            cfg = self._load_config(config_path)
        else:
            cfg = self._normalize_config(config)

        self.tickers: List[str] = cfg["tickers"]
        self.period: str = cfg["period"]
        self.interval: str = cfg["interval"]
        self.atr_mult: float = float(cfg["risk"].get("atr_mult", 1.0))
        self.pct_buf: float = float(cfg["risk"].get("percent_buffer", 0.0025))

    # --------------- Public API ---------------

    def run(self):
        for t in self.tickers:
            try:
                df = self._get_history(t, self.period, self.interval)
            except Exception as e:
                print(f"[{t}] ERROR: {e}")
                continue

            # Chart patterns near present
            chart_plans: List[PatternDirector.Plan] = []
            for detector in (self._detect_cup_and_handle,
                             self._detect_double_bottom,
                             self._detect_head_and_shoulders):
                try:
                    plan = detector(df)
                    if plan:
                        chart_plans.append(plan)
                except Exception:
                    # Avoid a single detector killing the run
                    pass

            # Candlesticks for last two bars w/ present cancel
            candle_plans = self._latest_candle_plan(df)

            best = self._choose_best_plan(chart_plans, candle_plans)

            print(f"\n=== {t} — Present Pattern Director ({self.period}, {self.interval}) ===")
            if best:
                self._print_best(df, best)
            else:
                print("No actionable setup now. (Everything either stale or canceled.)")

            # Context
            for p in chart_plans:
                print(f"  • {p.pattern:16} {p.state:12} {p.status:12} "
                      f"entry={p.entry and round(p.entry,2)} stop={p.stop and round(p.stop,2)}")
            for c in candle_plans:
                # only show the last two bars (already the case), keep line for clarity
                print(f"  • Candle {c.pattern:16} {c.state:12} {c.status:12} ({c.notes})")

    # --------------- Config / Data ---------------

    def _load_config(self, json_path: str | Path) -> Dict:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return self._normalize_config(raw)

    def _normalize_config(self, raw: Dict) -> Dict:
        if "tickers" not in raw or not isinstance(raw["tickers"], list) or not raw["tickers"]:
            raise ValueError("Config must contain non-empty 'tickers' list.")
        raw.setdefault("period", "1y")
        raw.setdefault("interval", "1d")
        raw.setdefault("risk", {"atr_mult": 1.0, "percent_buffer": 0.0025})
        return raw

    def _get_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data for {ticker}.")
        df = df.rename(columns=str.lower).dropna()
        df["atr14"] = self._atr(df, 14)
        return df

    def _atr(self, df: pd.DataFrame, n: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return tr.rolling(n, min_periods=n).mean()

    # --------------- Utilities ---------------

    @staticmethod
    def _real_body(o: float, c: float) -> float:
        return abs(c - o)

    @staticmethod
    def _upper_shadow(o: float, h: float, c: float) -> float:
        return h - max(o, c)

    @staticmethod
    def _lower_shadow(o: float, l: float, c: float) -> float:
        return min(o, c) - l

    @staticmethod
    def _find_peaks(arr: np.ndarray, window: int = 3) -> List[int]:
        idxs = []
        for i in range(window, len(arr) - window):
            if arr[i] == max(arr[i - window:i + window + 1]) and arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                idxs.append(i)
        return idxs

    @staticmethod
    def _find_troughs(arr: np.ndarray, window: int = 3) -> List[int]:
        idxs = []
        for i in range(window, len(arr) - window):
            if arr[i] == min(arr[i - window:i + window + 1]) and arr[i] < arr[i-1] and arr[i] < arr[i+1]:
                idxs.append(i)
        return idxs

    # --------------- Candlestick detectors ---------------

    def _is_doji(self, o, h, l, c, rb_ratio: float = 0.1) -> bool:
        rng = max(h - l, 1e-9)
        return self._real_body(o, c) <= rb_ratio * rng

    def _is_shooting_star(self, o, h, l, c) -> bool:
        rng = max(h - l, 1e-9)
        rb = self._real_body(o, c)
        return (self._upper_shadow(o, h, c) >= 0.6 * rng and
                self._lower_shadow(o, l, c) <= 0.1 * rng and
                rb <= 0.3 * rng)

    def _is_hammer(self, o, h, l, c) -> bool:
        rng = max(h - l, 1e-9)
        rb = self._real_body(o, c)
        return (self._lower_shadow(o, l, c) >= 0.6 * rng and
                self._upper_shadow(o, h, c) <= 0.1 * rng and
                rb <= 0.3 * rng)

    @staticmethod
    def _is_bullish_engulfing(prev: pd.Series, cur: pd.Series) -> bool:
        p_o = float(prev["open"]);  p_c = float(prev["close"])
        c_o = float(cur["open"]);   c_c = float(cur["close"])
        # prev down, current up, and current body engulfs prev body
        return (p_c < p_o) and (c_c > c_o) and (c_o <= p_c) and (c_c >= p_o)

    @staticmethod
    def _is_bearish_engulfing(prev: pd.Series, cur: pd.Series) -> bool:
        p_o = float(prev["open"]);  p_c = float(prev["close"])
        c_o = float(cur["open"]);   c_c = float(cur["close"])
        # prev up, current down, and current body engulfs prev body
        return (p_c > p_o) and (c_c < c_o) and (c_o >= p_c) and (c_c <= p_o)

    def _latest_candle_plan(self, df: pd.DataFrame) -> List[Plan]:
        """Candlestick signals on yesterday & today with present cancel rules."""
        plans: List[PatternDirector.Plan] = []
        if len(df) < 3:
            return plans

        last_close = float(df["close"].iloc[-1])

        for i in [-2, -1]:  # yesterday and today
            row, prev = df.iloc[i], df.iloc[i - 1]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            date = df.index[i]

            def add(name: str, side: str, hi: float, lo: float):
                if side == "bear":
                    cancel_now = last_close > hi
                elif side == "bull":
                    cancel_now = last_close < lo
                else:
                    cancel_now = False
                status = "CANCELED NOW" if cancel_now else ("PENDING" if i == -1 else "VALID")
                stop = lo if side == "bull" else (hi if side == "bear" else None)
                plans.append(self.Plan(
                    pattern=name, side=side, state="CANDLE", date=date,
                    entry=None, stop=stop, target=None,
                    cancel_now=cancel_now, status=status,
                    notes=f"signal_candle={'today' if i==-1 else 'yesterday'}"
                ))

            if self._is_shooting_star(o, h, l, c):
                add("Shooting Star", "bear", h, l)
            if self._is_hammer(o, h, l, c):
                add("Hammer", "bull", h, l)
            if self._is_doji(o, h, l, c):
                add("Doji", "neutral", h, l)
            if self._is_bullish_engulfing(prev, row):
                add("Bullish Engulfing", "bull", h, l)
            if self._is_bearish_engulfing(prev, row):
                add("Bearish Engulfing", "bear", h, l)

        return plans

    # --------------- Chart pattern detectors ---------------

    def _detect_cup_and_handle(self,
                               df: pd.DataFrame,
                               lookback: int = 100,
                               min_depth: float = 0.12,
                               max_handle_depth: float = 0.10) -> Optional[Plan]:
        section = df.iloc[-lookback:].copy()
        highs = section["high"].to_numpy()
        lows = section["low"].to_numpy()
        closes = section["close"].to_numpy()
        last_close = closes[-1]
        atr14 = float(section["atr14"].iloc[-1])

        rim_price = float(highs.max())
        bottom_price = float(lows.min())
        depth = (rim_price - bottom_price) / max(rim_price, 1e-9)
        if depth < min_depth:
            return None

        rim_idx = int(highs.argmax())
        bottom_idx = int(lows.argmin())
        if not (rim_idx < bottom_idx < len(section) - 6):
            return None

        # Handle = last 5–20 bars staying near rim
        handle_window = min(20, len(section) - (bottom_idx + 1))
        if handle_window < 5:
            return None
        handle_lows = lows[-handle_window:]
        handle_highs = highs[-handle_window:]

        handle_low = float(handle_lows.min())
        handle_high = float(handle_highs.max())
        handle_depth = (rim_price - handle_low) / max(rim_price, 1e-9)
        if handle_depth > max_handle_depth or handle_low <= bottom_price:
            return None

        entry = max(handle_high, rim_price)  # breakout level
        stop = handle_low
        target = entry + (rim_price - bottom_price)

        # Present cancel/valid state
        if last_close > entry:
            # Breakout already; fail if immediate slump below entry (small buffer)
            buff = max(atr14, entry * 0.0025)
            cancel_now = last_close < (entry - 0.25 * buff)
            status, state = ("CANCELED NOW" if cancel_now else "VALID", "BREAKOUT")
        else:
            cancel_now = last_close < stop
            status, state = ("CANCELED NOW" if cancel_now else "PENDING", "PRE_BREAKOUT")

        notes = f"depth={depth:.1%}, handle_depth={handle_depth:.1%}"
        return self.Plan("Cup & Handle", "bull", state, section.index[-1], float(entry), float(stop),
                         float(target), cancel_now, status, notes)

    def _detect_double_bottom(self,
                              df: pd.DataFrame,
                              lookback: int = 120,
                              max_gap_pct: float = 0.03) -> Optional[Plan]:
        section = df.iloc[-lookback:].copy()
        lows = section["low"].to_numpy()
        highs = section["high"].to_numpy()
        closes = section["close"].to_numpy()
        atr14 = float(section["atr14"].iloc[-1])
        last_close = float(closes[-1])

        troughs = self._find_troughs(lows, window=3)
        if len(troughs) < 2:
            return None
        i1, i2 = troughs[-2], troughs[-1]
        low1, low2 = float(lows[i1]), float(lows[i2])

        if abs(low2 - low1) / max((low1 + low2) / 2, 1e-9) > max_gap_pct:
            return None

        neckline = float(highs[i1:i2+1].max())
        entry = neckline
        stop = min(low1, low2)
        target = entry + (entry - stop)

        if last_close > entry:
            buff = max(atr14, entry * 0.0025)
            cancel_now = last_close < (entry - 0.25 * buff)
            status, state = ("CANCELED NOW" if cancel_now else "VALID", "BREAKOUT")
        else:
            cancel_now = last_close < stop
            status, state = ("CANCELED NOW" if cancel_now else "PENDING", "PRE_BREAKOUT")

        notes = f"lows≈equal ({low1:.2f}/{low2:.2f})"
        return self.Plan("Double Bottom", "bull", state, section.index[-1], float(entry), float(stop),
                         float(target), cancel_now, status, notes)

    def _detect_head_and_shoulders(self,
                                   df: pd.DataFrame,
                                   lookback: int = 150,
                                   shoulder_tol: float = 0.10) -> Optional[Plan]:
        section = df.iloc[-lookback:].copy()
        highs = section["high"].to_numpy()
        lows = section["low"].to_numpy()
        closes = section["close"].to_numpy()
        atr14 = float(section["atr14"].iloc[-1])
        last_close = float(closes[-1])

        peaks = self._find_peaks(highs, window=3)
        troughs = self._find_troughs(lows, window=3)
        if len(peaks) < 3 or len(troughs) < 2:
            return None

        p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
        LS, HEAD, RS = float(highs[p1]), float(highs[p2]), float(highs[p3])
        if not (HEAD > LS and HEAD > RS):
            return None
        if abs(LS - RS) / max((LS + RS) / 2, 1e-9) > shoulder_tol:
            return None

        lows_between = [t for t in troughs if p1 < t < p3]
        if len(lows_between) < 2:
            return None
        t1, t2 = lows_between[0], lows_between[-1]
        neck1, neck2 = float(lows[t1]), float(lows[t2])
        neckline = (neck1 + neck2) / 2.0  # simple avg neckline

        entry = neckline
        stop = max(LS, RS)
        target = entry - (HEAD - neckline)  # measured move downward

        if last_close < entry:
            buff = max(atr14, entry * 0.0025)
            cancel_now = last_close > (entry + 0.25 * buff)
            status, state = ("CANCELED NOW" if cancel_now else "VALID", "BREAKOUT")
        else:
            cancel_now = last_close > stop
            status, state = ("CANCELED NOW" if cancel_now else "PENDING", "PRE_BREAKOUT")

        notes = f"LS={LS:.2f} HEAD={HEAD:.2f} RS={RS:.2f}"
        return self.Plan("Head & Shoulders", "bear", state, section.index[-1], float(entry), float(stop),
                         float(target), cancel_now, status, notes)

    # --------------- Planner ---------------

    def _choose_best_plan(self, chart_plans: List[Plan], candle_plans: List[Plan]) -> Optional[Plan]:
        """Priority: Chart BREAKOUT > Chart PRE > Candle VALID > Candle PENDING. Drop CANCELED NOW."""
        def pri(p: PatternDirector.Plan) -> Tuple[int, int]:
            state_rank = {"BREAKOUT": 3, "PRE_BREAKOUT": 2, "CANDLE": 1}[p.state]
            ok_rank = {"VALID": 2, "PENDING": 1, "CANCELED NOW": 0}[p.status]
            return (state_rank, ok_rank)

        pool = [p for p in chart_plans + candle_plans if p.status != "CANCELED NOW"]
        if not pool:
            return None
        pool.sort(key=pri, reverse=True)
        return pool[0]

    def _print_best(self, df: pd.DataFrame, best: Plan):
        last_atr = float(df["atr14"].iloc[-1])
        price = float(df["close"].iloc[-1])
        buff = max(last_atr * self.atr_mult, price * self.pct_buf)

        e = f"{best.entry:.2f}" if best.entry else "-"
        s = f"{best.stop:.2f}" if best.stop else "-"
        t = f"{best.target:.2f}" if best.target else "-"

        print(f"[{best.date.date()}] {best.pattern} ({best.state}) -> {best.status} | {best.notes}")
        print(f"  Entry : {e}")
        print(f"  Stop  : {s}")
        print(f"  Target: {t}")
        if best.entry and best.stop:
            print(f"  Buffers -> ATR14={last_atr:.2f}, min_pct_buffer={self.pct_buf*100:.2f}% (use >= {buff:.2f})")

        if best.status == "PENDING":
            if best.state == "PRE_BREAKOUT":
                print("  Watch: Breakout above entry; CANCEL NOW if price closes below stop.")
            elif best.state == "CANDLE":
                if best.side == "bull":
                    print("  Watch: Confirm up next bar; CANCEL NOW if present close < signal low.")
                elif best.side == "bear":
                    print("  Watch: Confirm down next bar; CANCEL NOW if present close > signal high.")
                else:
                    print("  Note : Doji is neutral; wait for direction.")



if __name__ == "__main__":
    PatternDirector().run()