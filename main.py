
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

QUOTE_CURRENCIES = {"EUR"}
STABLECOINS = {
    "USDT", "USDC", "DAI", "TUSD", "USDP", "USDD", "BUSD", "FDUSD",
    "PYUSD", "GUSD", "LUSD", "FRAX", "SUSD", "USDS", "USDE", "SDAI",
    "EURT", "EURS", "EUROC", "EURA"
}
FIAT_ASSETS = {"USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD", "NZD", "SGD"}
LEVERAGED_WORDS = {
    "UP", "DOWN", "BULL", "BEAR", "LONG", "SHORT",
    "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S"
}


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("KrakenPortfolioBot")


LOGGER = setup_logging()


def start_health_server() -> None:
    port = int(os.getenv("PORT", "8080"))

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = b"OK"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    def run_server() -> None:
        try:
            server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
            LOGGER.info("Health server attivo su porta %s", port)
            server.serve_forever()
        except Exception as exc:
            LOGGER.exception("Errore health server: %s", exc)

    threading.Thread(target=run_server, daemon=True).start()


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# ─────────────────────────────────────────────
# Fear & Greed index (alternative.me public API)
# ─────────────────────────────────────────────
_fear_greed_cache: Dict[str, Any] = {"value": None, "ts": 0.0}
_FEAR_GREED_TTL = 3600


def fetch_fear_greed() -> Optional[int]:
    now = time.time()
    if _fear_greed_cache["value"] is not None and now - _fear_greed_cache["ts"] < _FEAR_GREED_TTL:
        return _fear_greed_cache["value"]
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        val = int(r.json()["data"][0]["value"])
        _fear_greed_cache["value"] = val
        _fear_greed_cache["ts"] = now
        return val
    except Exception:
        return _fear_greed_cache.get("value")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
@dataclass
class Config:
    kraken_api_key: str = os.getenv("KRAKEN_API_KEY", "").strip()
    kraken_secret:  str = os.getenv("KRAKEN_SECRET",  "").strip()
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "").strip()
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    # Timeframes
    timeframe:              str = os.getenv("TIMEFRAME", "15m").strip()
    timeframe_confirmation: str = os.getenv("TIMEFRAME_CONFIRMATION", "1h").strip()
    timeframe_macro:        str = os.getenv("TIMEFRAME_MACRO", "4h").strip()

    ohlcv_limit:               int   = env_int("OHLCV_LIMIT", 250)
    scan_interval_seconds:     int   = env_int("SCAN_INTERVAL_SECONDS", 60)
    per_symbol_delay_seconds:  float = env_float("PER_SYMBOL_DELAY_SECONDS", 1.2)
    market_refresh_seconds:    int   = env_int("MARKET_REFRESH_SECONDS", 3600)
    retry_attempts:            int   = env_int("RETRY_ATTEMPTS", 3)
    retry_sleep_seconds:       float = env_float("RETRY_SLEEP_SECONDS", 2.0)

    # Liquidity / market filters
    min_24h_quote_volume_eur:     float = env_float("MIN_24H_QUOTE_VOLUME_EUR", 20_000_000.0)
    max_spread_percent:           float = env_float("MAX_SPREAD_PERCENT", 0.35)
    min_orderbook_bid_ask_ratio:  float = env_float("MIN_ORDERBOOK_BID_ASK_RATIO", 1.2)

    # Portfolio / risk
    max_open_trades:                  int   = min(env_int("MAX_OPEN_TRADES", 3), 3)
    min_trade_amount:                 float = env_float("MIN_TRADE_AMOUNT", 5.0)
    daily_max_loss:                   float = min(env_float("DAILY_MAX_LOSS", 0.05), 0.05)
    max_total_risk:                   float = env_float("MAX_TOTAL_RISK", 0.15)

    # Portfolio allocation fractions (of available / open_slots)
    alloc_extreme:  float = env_float("ALLOC_EXTREME",  0.35)   # score >= 110
    alloc_high:     float = env_float("ALLOC_HIGH",     0.28)   # score >= 100
    alloc_normal:   float = env_float("ALLOC_NORMAL",   0.20)   # default

    # Replacement thresholds
    replace_score_diff:    float = env_float("REPLACE_SCORE_DIFF",    15.0)
    replace_pnl_threshold: float = env_float("REPLACE_PNL_THRESHOLD", -3.0)   # %

    # Indicators
    ema_fast:   int = env_int("EMA_FAST",   20)
    ema_slow:   int = env_int("EMA_SLOW",   50)
    ema_long:   int = env_int("EMA_LONG",  200)
    rsi_period: int = env_int("RSI_PERIOD", 14)
    atr_period: int = env_int("ATR_PERIOD", 14)

    # Buy conditions
    rsi_buy_min:              float = env_float("RSI_BUY_MIN",              55.0)
    rsi_buy_max:              float = env_float("RSI_BUY_MAX",              70.0)
    rsi_exit:                 float = env_float("RSI_EXIT",                 82.0)
    max_rsi_allowed:          float = env_float("MAX_RSI_ALLOWED",          72.0)
    momentum_min:             float = env_float("MOMENTUM_MIN",             0.012)
    momentum_max:             float = env_float("MOMENTUM_MAX",             0.055)
    sell_momentum_min:        float = env_float("SELL_MOMENTUM_MIN",        0.005)
    volume_breakout_multiplier: float = env_float("VOLUME_BREAKOUT_MULTIPLIER", 1.8)
    volume_collapse_threshold: float = env_float("VOLUME_COLLAPSE_THRESHOLD", 0.6)
    volume_window:            int   = env_int("VOLUME_WINDOW",              20)
    breakout_lookback:        int   = env_int("BREAKOUT_LOOKBACK",          20)
    breakout_buffer:          float = env_float("BREAKOUT_BUFFER",          1.0025)
    momentum_lookback:        int   = env_int("MOMENTUM_LOOKBACK",          5)
    min_atr_percent:          float = env_float("MIN_ATR_PERCENT",          0.01)
    max_atr_percent:          float = env_float("MAX_ATR_PERCENT",          0.05)
    buy_score_threshold:      float = env_float("BUY_SCORE_THRESHOLD",      84.0)

    # Stop loss / trailing / TP
    stop_loss_atr_multiplier:    float = env_float("STOP_LOSS_ATR_MULTIPLIER",    1.5)
    stop_loss_percent:           float = env_float("STOP_LOSS_PERCENT",           0.04)
    trailing_atr_multiplier:     float = env_float("TRAILING_ATR_MULTIPLIER",     2.0)
    trailing_activation_percent: float = env_float("TRAILING_ACTIVATION_PERCENT", 0.05)
    trailing_distance_percent:   float = env_float("TRAILING_DISTANCE_PERCENT",   0.02)
    partial_tp_1_percent:        float = env_float("PARTIAL_TP_1_PERCENT",        0.05)
    partial_tp_1_size:           float = env_float("PARTIAL_TP_1_SIZE",           0.25)
    partial_tp_2_percent:        float = env_float("PARTIAL_TP_2_PERCENT",        0.10)
    partial_tp_2_size:           float = env_float("PARTIAL_TP_2_SIZE",           0.25)
    runner_position_size:        float = env_float("RUNNER_POSITION_SIZE",        0.50)

    # Cooldowns / protection
    max_trade_hours:             int   = env_int("MAX_TRADE_HOURS",              48)
    max_consecutive_losses:      int   = env_int("MAX_CONSECUTIVE_LOSSES",        3)
    loss_cooldown_seconds:       int   = env_int("LOSS_COOLDOWN_SECONDS",     10800)
    min_seconds_between_trades:  int   = env_int("MIN_SECONDS_BETWEEN_TRADES",  900)
    symbol_cooldown_seconds:     int   = env_int("SYMBOL_COOLDOWN_SECONDS",    5400)

    # Telegram
    telegram_signal_interval_seconds: int = env_int("TELEGRAM_SIGNAL_INTERVAL_SECONDS", 1800)
    top_signals_limit:                int = env_int("TOP_SIGNALS_LIMIT", 10)

    # Fear & Greed
    fear_greed_min: int = env_int("FEAR_GREED_MIN", 30)
    fear_greed_max: int = env_int("FEAR_GREED_MAX", 72)

    # Execution
    dry_run:              bool  = env_bool("DRY_RUN", False)
    max_slippage_percent: float = env_float("MAX_SLIPPAGE_PERCENT", 0.5)
    state_file:           str   = os.getenv("STATE_FILE", "bot_state.json").strip()

    def telegram_enabled(self) -> bool:
        return bool(
            self.telegram_token
            and self.telegram_chat_id
            and self.telegram_token.upper()   != "DISABLED"
            and self.telegram_chat_id.upper() != "DISABLED"
        )


# ─────────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────────
class TelegramPanel:
    def __init__(self, cfg: Config):
        self.cfg      = cfg
        self.enabled  = cfg.telegram_enabled()
        self.base_url = f"https://api.telegram.org/bot{cfg.telegram_token}"
        self.chat_id  = str(cfg.telegram_chat_id)
        self.session  = requests.Session()
        self.handlers: Dict[str, Callable[[str], str]] = {}
        self.offset:   Optional[int]     = None
        self.stop_event = threading.Event()
        self.thread:   Optional[threading.Thread] = None

    def register(self, command: str, handler: Callable[[str], str]) -> None:
        self.handlers[command.strip().lower().replace("/", "")] = handler

    def send(self, text: str, silent: bool = False) -> bool:
        if not self.enabled:
            LOGGER.info("Telegram disattivato: %s", text.replace("\n", " | ")[:300])
            return False

        try:
            ok = True

            for start in range(0, len(text), 3900):
                resp = self.session.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text[start:start + 3900],
                        "disable_notification": silent,
                        "parse_mode": "HTML",
                    },
                    timeout=20,
                )

                if resp.status_code >= 400:
                    ok = False
                    LOGGER.error(
                        "Errore Telegram %s: %s",
                        resp.status_code,
                        resp.text[:500],
                    )

                time.sleep(0.2)

            return ok

        except Exception as exc:
            LOGGER.exception("Errore invio Telegram: %s", exc)
            return False

    def start_polling(self) -> None:
        if not self.enabled:
            LOGGER.warning("Telegram non configurato: polling disattivato")
            return
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self.poll_loop, daemon=True)
        self.thread.start()

    def poll_loop(self) -> None:
        LOGGER.info("Polling Telegram avviato")
        while not self.stop_event.is_set():
            try:
                params: Dict[str, Any] = {"timeout": 25, "allowed_updates": ["message"]}
                if self.offset is not None:
                    params["offset"] = self.offset
                resp = self.session.get(f"{self.base_url}/getUpdates", params=params, timeout=35)
                if resp.status_code == 409:
                    LOGGER.warning("Telegram 409: altra istanza attiva, attendo 30s")
                    time.sleep(30)
                    continue
                if resp.status_code >= 400:
                    LOGGER.error("Telegram getUpdates %s: %s", resp.status_code, resp.text[:500])
                    time.sleep(5)
                    continue
                for update in resp.json().get("result", []):
                    self.offset = int(update.get("update_id", 0)) + 1
                    self.handle_update(update)
            except Exception as exc:
                LOGGER.warning("Errore polling Telegram: %s", exc)
                time.sleep(5)

    def handle_update(self, update: Dict[str, Any]) -> None:
        try:
            message = update.get("message") or {}
            chat    = message.get("chat") or {}
            chat_id = str(chat.get("id", ""))
            text    = str(message.get("text", "")).strip()
            if chat_id != self.chat_id or not text.startswith("/"):
                return
            parts   = text.split()
            command = parts[0].split("@")[0].replace("/", "").lower()
            handler = self.handlers.get(command)
            if not handler:
                self.send("Comando non riconosciuto. Usa /help.")
                return
            reply = handler(text)
            if reply:
                self.send(reply)
        except Exception as exc:
            LOGGER.exception("Errore comando Telegram: %s", exc)
            self.send(f"Errore comando Telegram: {exc}")


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class Signal:
    symbol:     str
    price:      float
    score:      float
    buy:        bool
    reasons:    List[str]
    blocked_by: List[str]
    metrics:    Dict[str, float]
    timestamp:  str


@dataclass
class Position:
    symbol:        str
    base:          str
    quote:         str
    amount:        float
    entry_price:   float
    entry_time:    str
    stop_loss:     float
    take_profit:   float
    trailing_stop: float
    highest_price: float
    order_id:      str
    quote_cost:    float
    fees:          float
    score:         float
    tp1_done:      bool = False
    tp2_done:      bool = False


# ─────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────
class Strategy:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def dataframe(self, ohlcv: List[List[float]]) -> pd.DataFrame:
        try:
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            if df.empty:
                return df
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.dropna().reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            needed = max(self.cfg.ema_long, self.cfg.breakout_lookback, self.cfg.volume_window, 210)
            if len(df) < needed:
                return pd.DataFrame()
            df = df.copy()
            close = df["close"]
            df["ema20"]  = EMAIndicator(close=close, window=self.cfg.ema_fast).ema_indicator()
            df["ema50"]  = EMAIndicator(close=close, window=self.cfg.ema_slow).ema_indicator()
            df["ema200"] = EMAIndicator(close=close, window=self.cfg.ema_long).ema_indicator()
            df["rsi"]    = RSIIndicator(close=close, window=self.cfg.rsi_period).rsi()
            df["atr"]    = AverageTrueRange(
                high=df["high"], low=df["low"], close=df["close"], window=self.cfg.atr_period,
            ).average_true_range()
            df["volume_avg"]    = df["volume"].rolling(self.cfg.volume_window).mean()
            df["breakout_high"] = df["high"].shift(1).rolling(self.cfg.breakout_lookback).max()
            df["momentum"]      = df["close"].pct_change(self.cfg.momentum_lookback)
            df["atr_percent"]   = df["atr"] / df["close"]
            df["quote_volume"]  = df["close"] * df["volume"]
            df["ema50_slope"]   = df["ema50"].diff()
            return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def btc_regime(self, ohlcv_15m: List[List[float]]) -> Tuple[bool, str]:
        """
        Verifica il regime BTC sul timeframe 15m disponibile.
        Bullish se: EMA20 > EMA50 > EMA200 e close > EMA200.
        Ritorna (bullish: bool, descrizione: str).
        """
        try:
            df = self.indicators(self.dataframe(ohlcv_15m))
            if df.empty:
                return True, "BTC regime n/d (skip)"
            row   = df.iloc[-1]
            ema20 = float(row["ema20"])
            ema50 = float(row["ema50"])
            ema200 = float(row["ema200"])
            close  = float(row["close"])
            bullish = ema20 > ema50 and ema50 > ema200
            desc = (
                f"BTC EMA20={ema20:.2f} EMA50={ema50:.2f} EMA200={ema200:.2f} close={close:.2f}"
            )
            return bullish, desc
        except Exception:
            return True, "BTC regime errore (skip)"

    def analyze(self, symbol: str, ohlcv: List[List[float]]) -> Optional[Signal]:
        try:
            df = self.indicators(self.dataframe(ohlcv))
            if df.empty:
                return None

            row  = df.iloc[-1]

            price         = float(row["close"])
            ema20         = float(row["ema20"])
            ema50         = float(row["ema50"])
            ema200        = float(row["ema200"])
            rsi           = float(row["rsi"])
            atr           = float(row["atr"])
            atr_percent   = float(row["atr_percent"])
            volume        = float(row["volume"])
            volume_avg    = float(row["volume_avg"])
            breakout_high = float(row["breakout_high"])
            momentum      = float(row["momentum"])
            ema50_slope   = float(row["ema50_slope"])
            quote_volume_24h = float(df["quote_volume"].tail(96).sum())

            vol_ratio = volume / max(volume_avg, 1e-12)

            # ── Conditions ───────────────────────────────────────────────
            liquid          = quote_volume_24h >= self.cfg.min_24h_quote_volume_eur
            trend_up        = ema20 > ema50 and ema50 > ema200
            ema50_slope_pos = ema50_slope > 0
            rsi_ok          = self.cfg.rsi_buy_min <= rsi <= self.cfg.rsi_buy_max
            rsi_not_extreme = rsi <= self.cfg.max_rsi_allowed
            volume_breakout = vol_ratio >= self.cfg.volume_breakout_multiplier
            breakout        = price > breakout_high * self.cfg.breakout_buffer
            momentum_ok     = self.cfg.momentum_min <= momentum <= self.cfg.momentum_max
            volatility_ok   = self.cfg.min_atr_percent <= atr_percent <= self.cfg.max_atr_percent
            not_parabolic   = rsi <= self.cfg.max_rsi_allowed and atr_percent <= self.cfg.max_atr_percent

            # ── Score ─────────────────────────────────────────────────────
            score:      float     = 0.0
            reasons:    List[str] = []
            blocked_by: List[str] = []

            if liquid:
                score += 15
                reasons.append("liquido")
            else:
                blocked_by.append(f"vol24h basso ({quote_volume_24h:,.0f} EUR)")

            if trend_up:
                ema_str = min(max((ema20 - ema50) / max(price, 1e-12), 0.0) * 100.0, 8.0)
                score += 25 + ema_str
                reasons.append("EMA20>EMA50>EMA200")
            else:
                blocked_by.append("trend non allineato")

            if ema50_slope_pos:
                score += 5
                reasons.append("EMA50 slope+")
            else:
                blocked_by.append("EMA50 slope piatta/negativa")

            if rsi_ok:
                score += 15
                reasons.append(f"RSI {rsi:.1f}")
            else:
                blocked_by.append(f"RSI fuori range ({rsi:.1f})")

            if rsi_not_extreme:
                score += 5
                reasons.append("RSI non estremo")
            else:
                blocked_by.append(f"RSI estremo ({rsi:.1f} > {self.cfg.max_rsi_allowed})")

            if volume_breakout:
                vol_str = min((vol_ratio - self.cfg.volume_breakout_multiplier) * 5.0, 10.0)
                score += 20 + vol_str
                reasons.append(f"vol x{vol_ratio:.1f}")
            else:
                blocked_by.append(f"vol insufficiente ({vol_ratio:.2f}x < {self.cfg.volume_breakout_multiplier}x)")

            if breakout:
                brk_str = min(max(price / max(breakout_high, 1e-12) - 1.0, 0.0) * 100.0, 7.0)
                score += 15 + brk_str
                reasons.append("breakout confermato")
            else:
                blocked_by.append("no breakout")

            if momentum_ok:
                score += 10
                reasons.append(f"mom {momentum * 100:+.2f}%")
            else:
                blocked_by.append(f"momentum fuori range ({momentum * 100:.2f}%)")

            if volatility_ok and not_parabolic:
                score += 5
                reasons.append(f"ATR {atr_percent * 100:.2f}%")
            else:
                blocked_by.append(f"volatilita fuori range ({atr_percent * 100:.2f}%)")

            mandatory_ok = (
                liquid and trend_up and ema50_slope_pos
                and rsi_ok and rsi_not_extreme
                and volume_breakout and breakout
                and momentum_ok and volatility_ok and not_parabolic
            )

            buy = score >= self.cfg.buy_score_threshold and mandatory_ok

            return Signal(
                symbol=symbol,
                price=price,
                score=round(score, 2),
                buy=buy,
                reasons=reasons,
                blocked_by=blocked_by,
                metrics={
                    "ema20": ema20,
                    "ema50": ema50,
                    "ema200": ema200,
                    "rsi": rsi,
                    "atr": atr,
                    "atr_percent": atr_percent,
                    "volume_ratio": vol_ratio,
                    "momentum": momentum,
                    "quote_volume_24h": quote_volume_24h,
                },
                timestamp=str(row["timestamp"]),
            )
        except Exception:
            return None

    def exit_signal(self, ohlcv: List[List[float]]) -> Dict[str, Any]:
        try:
            df = self.indicators(self.dataframe(ohlcv))
            if df.empty:
                return {"exit": False, "reason": "", "metrics": {}}

            row      = df.iloc[-1]
            price    = float(row["close"])
            ema20    = float(row["ema20"])
            ema50    = float(row["ema50"])
            ema200   = float(row["ema200"])
            rsi      = float(row["rsi"])
            momentum = float(row["momentum"])
            atr      = float(row["atr"])
            volume   = float(row["volume"])
            vol_avg  = float(row["volume_avg"])
            vol_ratio = volume / max(vol_avg, 1e-12)

            metrics = {
                "price": price, "rsi": rsi, "momentum": momentum,
                "atr": atr, "volume_ratio": vol_ratio,
            }

            if ema20 < ema50 and momentum < 0:
                return {"exit": True, "reason": "inversione trend (EMA20<EMA50)", "metrics": metrics}

            if ema50 < ema200:
                return {"exit": True, "reason": "EMA50<EMA200 (trend macro ribassista)", "metrics": metrics}

            if rsi >= self.cfg.rsi_exit and momentum <= self.cfg.sell_momentum_min:
                return {"exit": True, "reason": f"RSI overbought ({rsi:.1f})", "metrics": metrics}

            if momentum < self.cfg.sell_momentum_min:
                return {"exit": True, "reason": f"momentum collassato ({momentum * 100:.2f}%)", "metrics": metrics}

            if vol_ratio < self.cfg.volume_collapse_threshold:
                return {"exit": True, "reason": f"volume collassato ({vol_ratio:.2f}x)", "metrics": metrics}

            return {"exit": False, "reason": "", "metrics": metrics}
        except Exception:
            return {"exit": False, "reason": "", "metrics": {}}


# ─────────────────────────────────────────────
# RiskManager
# ─────────────────────────────────────────────
class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.current_day          = self.today()
        self.daily_start_equity   = 0.0
        self.daily_realized_pnl   = 0.0
        self.current_drawdown     = 0.0
        self.consecutive_losses   = 0
        self.pause_until          = 0.0
        self.last_trade_at        = 0.0
        self.symbol_last_trade_at: Dict[str, float] = {}
        self.load()

    def today(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def load(self) -> None:
        try:
            path = Path(self.cfg.state_file)
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            self.positions = {}
            for symbol, raw in data.get("positions", {}).items():
                raw.setdefault("tp1_done", False)
                raw.setdefault("tp2_done", False)
                self.positions[symbol] = Position(**raw)
            self.closed_trades      = list(data.get("closed_trades", []))[-500:]
            self.current_day        = str(data.get("current_day",        self.today()))
            self.daily_start_equity = float(data.get("daily_start_equity", 0.0))
            self.daily_realized_pnl = float(data.get("daily_realized_pnl", 0.0))
            self.current_drawdown   = float(data.get("current_drawdown",   0.0))
            self.consecutive_losses = int(data.get("consecutive_losses",   0))
            self.pause_until        = float(data.get("pause_until",        0.0))
            self.last_trade_at      = float(data.get("last_trade_at",      0.0))
            self.symbol_last_trade_at = {
                str(k): float(v)
                for k, v in data.get("symbol_last_trade_at", {}).items()
            }
        except Exception:
            pass

    def save(self) -> None:
        try:
            data = {
                "positions":           {s: asdict(p) for s, p in self.positions.items()},
                "closed_trades":       self.closed_trades[-500:],
                "current_day":         self.current_day,
                "daily_start_equity":  self.daily_start_equity,
                "daily_realized_pnl":  self.daily_realized_pnl,
                "current_drawdown":    self.current_drawdown,
                "consecutive_losses":  self.consecutive_losses,
                "pause_until":         self.pause_until,
                "last_trade_at":       self.last_trade_at,
                "symbol_last_trade_at": self.symbol_last_trade_at,
            }
            Path(self.cfg.state_file).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def update_equity(self, equity: float) -> None:
        try:
            today = self.today()
            if today != self.current_day:
                self.current_day        = today
                self.daily_start_equity = max(0.0, equity)
                self.daily_realized_pnl = 0.0
                self.current_drawdown   = 0.0
                self.save()
            if self.daily_start_equity <= 0 and equity > 0:
                self.daily_start_equity = equity
                self.save()
            if self.daily_start_equity > 0:
                self.current_drawdown = max(
                    0.0,
                    (self.daily_start_equity - equity) / self.daily_start_equity,
                )
        except Exception:
            pass

    def daily_stop_hit(self) -> bool:
        if self.daily_start_equity <= 0:
            return False
        return (
            self.daily_realized_pnl <= -self.daily_start_equity * self.cfg.daily_max_loss
            or self.current_drawdown >= self.cfg.daily_max_loss
        )

    def pause_active(self) -> bool:
        return time.time() < self.pause_until

    def pause_minutes(self) -> int:
        return int(max(0.0, self.pause_until - time.time()) // 60)

    def can_open(self, symbol: str, quote_free: float) -> Tuple[bool, str]:
        now = time.time()
        if symbol in self.positions:
            return False, "posizione gia aperta"
        if quote_free < self.cfg.min_trade_amount:
            return False, f"saldo insufficiente ({quote_free:.2f} EUR)"
        if self.daily_stop_hit():
            return False, "stop giornaliero perdita attivo"
        if self.pause_active():
            return False, f"pausa rischio attiva {self.pause_minutes()} min"
        if now - self.last_trade_at < self.cfg.min_seconds_between_trades:
            remaining = int(self.cfg.min_seconds_between_trades - (now - self.last_trade_at))
            return False, f"anti-overtrading attivo ({remaining}s)"
        if now - self.symbol_last_trade_at.get(symbol, 0.0) < self.cfg.symbol_cooldown_seconds:
            return False, "cooldown simbolo attivo"
        return True, "ok"

    def trade_capital(self, quote_free: float, signal: "Signal") -> float:
        """
        Portfolio dinamico: divide il capitale disponibile per gli slot aperti.
        Alloca una frazione basata sullo score del segnale.
        """
        available  = max(0.0, quote_free * 0.95)
        open_slots = max(1, self.cfg.max_open_trades - len(self.positions))
        slot_budget = available / open_slots

        score = signal.score if signal else 0.0
        if score >= 110:
            alloc = self.cfg.alloc_extreme   # 40%
        elif score >= 100:
            alloc = self.cfg.alloc_high      # 33%
        else:
            alloc = self.cfg.alloc_normal    # 25%

        desired  = available * alloc
        capital  = min(desired, slot_budget)
        capital  = max(self.cfg.min_trade_amount, min(capital, available))
        return capital

    def weakest_position(self, current_prices: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """
        Trova la posizione più debole combinando score iniziale e PnL corrente.
        Ritorna (symbol, composite_score) oppure None se nessuna posizione aperta.
        """
        if not self.positions:
            return None
        worst_symbol: Optional[str] = None
        worst_score = float("inf")
        for symbol, pos in self.positions.items():
            price   = current_prices.get(symbol, pos.entry_price)
            pnl_pct = (price - pos.entry_price) / pos.entry_price * 100.0
            composite = pos.score + pnl_pct   # perdite abbassano il composite
            if composite < worst_score:
                worst_score  = composite
                worst_symbol = symbol
        return (worst_symbol, worst_score) if worst_symbol else None

    def levels(self, entry: float, atr: float) -> Dict[str, float]:
        stop_loss    = min(
            entry - atr * self.cfg.stop_loss_atr_multiplier,
            entry * (1.0 - self.cfg.stop_loss_percent),
        )
        take_profit  = entry * (1.0 + self.cfg.partial_tp_1_percent)
        trailing_stop = stop_loss
        return {"stop_loss": stop_loss, "take_profit": take_profit, "trailing_stop": trailing_stop}

    def add_position(self, pos: Position) -> None:
        now = time.time()
        self.positions[pos.symbol] = pos
        self.last_trade_at = now
        self.symbol_last_trade_at[pos.symbol] = now
        self.save()

    def update_trailing(self, symbol: str, price: float, atr: float) -> Optional[Position]:
        pos = self.positions.get(symbol)
        if not pos:
            return None
        if price > pos.highest_price:
            pos.highest_price = price
        activation_price = pos.entry_price * (1.0 + self.cfg.trailing_activation_percent)
        if pos.highest_price >= activation_price:
            atr_trail  = pos.highest_price - atr * self.cfg.trailing_atr_multiplier
            pct_trail  = pos.highest_price * (1.0 - self.cfg.trailing_distance_percent)
            candidate  = max(atr_trail, pct_trail)
            if candidate > pos.trailing_stop:
                pos.trailing_stop = candidate
        self.positions[symbol] = pos
        self.save()
        return pos

    def close(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
        fees: float,
        order_id: str,
    ) -> Optional[Dict[str, Any]]:
        pos = self.positions.pop(symbol, None)
        if not pos:
            return None
        gross       = (exit_price - pos.entry_price) * pos.amount
        net         = gross - pos.fees - fees
        pnl_percent = net / max(pos.entry_price * pos.amount, 1e-12) * 100.0
        trade = {
            "symbol": symbol, "base": pos.base, "quote": pos.quote,
            "amount": pos.amount, "entry_price": pos.entry_price,
            "exit_price": exit_price, "entry_time": pos.entry_time,
            "exit_time": self.now_iso(), "reason": reason,
            "net_pnl": net, "pnl_percent": pnl_percent,
            "fees": pos.fees + fees,
            "entry_order_id": pos.order_id, "exit_order_id": order_id,
        }
        self.closed_trades.append(trade)
        self.closed_trades = self.closed_trades[-500:]
        self.daily_realized_pnl += net
        now = time.time()
        self.last_trade_at = now
        self.symbol_last_trade_at[symbol] = now
        if net < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.pause_until = now + self.cfg.loss_cooldown_seconds
        else:
            self.consecutive_losses = 0
        self.save()
        return trade

    def unrealized_pnl(self, prices: Dict[str, float]) -> float:
        return sum(
            (prices.get(s, p.entry_price) - p.entry_price) * p.amount
            for s, p in self.positions.items()
        )

    def total_closed_pnl(self) -> float:
        return sum(float(t.get("net_pnl", 0.0)) for t in self.closed_trades)


# ─────────────────────────────────────────────
# KrakenTradingBot
# ─────────────────────────────────────────────
class KrakenTradingBot:
    def __init__(self):
        self.cfg      = Config()
        self.telegram = TelegramPanel(self.cfg)
        self.exchange: Optional[Any] = None
        self.strategy = Strategy(self.cfg)
        self.risk     = RiskManager(self.cfg)

        self.markets:      Dict[str, Any]   = {}
        self.symbols:      List[str]         = []
        self.best_signals: List[Signal]      = []
        self.last_prices:  Dict[str, float]  = {}
        self.last_balance: Dict[str, Any]    = {}

        self.current_equity   = 0.0
        self.trading_enabled  = True
        self.shutdown         = False
        self.last_error       = "nessuno"
        self.btc_bullish      = True   # regime BTC corrente

        self.scan_count          = 0
        self.liquid_count        = 0
        self.buy_signals_count   = 0
        self.last_scan_start     = "n/d"
        self.last_scan_end       = "n/d"
        self.last_market_reload  = 0.0
        self.last_signal_report  = 0.0

        self.register_commands()

    # ── commands ─────────────────────────────────────────────────────────
    def register_commands(self) -> None:
        self.telegram.register("saldo",       lambda _: self.cmd_balance())
        self.telegram.register("status",      lambda _: self.cmd_status())
        self.telegram.register("trades",      lambda _: self.cmd_trades())
        self.telegram.register("profitto",    lambda _: self.cmd_profit())
        self.telegram.register("mercato",     lambda _: self.cmd_market())
        self.telegram.register("portfolio",   lambda _: self.cmd_portfolio())
        self.telegram.register("segnali",     lambda _: self.format_signals())
        self.telegram.register("start",       lambda _: self.cmd_start())
        self.telegram.register("stop",        lambda _: self.cmd_stop())
        self.telegram.register("chiudi",      lambda t: self.cmd_chiudi(t))
        self.telegram.register("diagnostica", lambda _: self.cmd_diagnostica())
        self.telegram.register("help",        lambda _: self.cmd_help())

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def emoji_pnl(self, pnl: float) -> str:
        if pnl > 0:
            return "🟢"
        if pnl < 0:
            return "🔴"
        return "⚪"

    def emoji_market(self) -> str:
        return "🟢 BULLISH" if self.btc_bullish else "🔴 BEARISH"

    def format_money(self, value: float) -> str:
        return f"€{value:,.2f}"

    def kraken_env_ok(self) -> bool:
        missing = []
        if not self.cfg.kraken_api_key:
            missing.append("KRAKEN_API_KEY")
        if not self.cfg.kraken_secret:
            missing.append("KRAKEN_SECRET")
        if missing:
            msg = "Variabili ambiente Kraken mancanti: " + ", ".join(missing)
            self.last_error = msg
            LOGGER.error(msg)
            self.telegram.send(msg)
            return False
        return True

    def init_exchange(self) -> None:
        self.exchange = ccxt.kraken({
            "apiKey": os.getenv("KRAKEN_API_KEY"),
            "secret": os.getenv("KRAKEN_SECRET"),
            "enableRateLimit": True,
        })
        self.exchange.options["adjustForTimeDifference"] = True

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.cfg.retry_attempts + 1):
            try:
                return func(*args, **kwargs)
            except ccxt.RateLimitExceeded as exc:
                last_exc = exc
                time.sleep(self.cfg.retry_sleep_seconds * attempt * 2)
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.RequestException) as exc:
                last_exc = exc
                LOGGER.warning("Rete Kraken retry %s: %s", attempt, exc)
                time.sleep(self.cfg.retry_sleep_seconds * attempt)
            except ccxt.ExchangeError as exc:
                last_exc = exc
                LOGGER.warning("Exchange Kraken retry %s: %s", attempt, exc)
                time.sleep(self.cfg.retry_sleep_seconds * attempt)
            except Exception as exc:
                last_exc = exc
                LOGGER.warning("Errore generico retry %s: %s", attempt, exc)
                time.sleep(self.cfg.retry_sleep_seconds * attempt)
        raise RuntimeError(f"Errore Kraken dopo {self.cfg.retry_attempts} retry: {last_exc}")

    # ── market ───────────────────────────────────────────────────────────
    def valid_market(self, symbol: str, market: Dict[str, Any]) -> bool:
        try:
            if market.get("active") is False:
                return False
            if market.get("spot") is False:
                return False
            base      = str(market.get("base", "")).upper()
            quote     = str(market.get("quote", "")).upper()
            market_id = str(market.get("id", "")).upper()
            raw = (base + symbol + market_id).upper().replace("/", "").replace("-", "").replace("_", "")
            if quote not in QUOTE_CURRENCIES:
                return False
            if base in STABLECOINS or base in FIAT_ASSETS:
                return False
            for word in LEVERAGED_WORDS:
                if raw.endswith(word) or raw.startswith(word):
                    return False
            return True
        except Exception:
            return False

    def load_markets(self) -> None:
        LOGGER.info("Caricamento mercati Kraken")
        self.markets = self.call(lambda: self.exchange.load_markets(reload=True))
        self.symbols = sorted(
            s for s, m in self.markets.items() if self.valid_market(s, m)
        )
        self.last_market_reload = time.time()
        LOGGER.info("Mercati EUR monitorati: %s", len(self.symbols))
        self.telegram.send(
            f"Connessione Kraken riuscita\nCoin monitorate EUR: {len(self.symbols)}"
        )

    def connect(self) -> None:
        LOGGER.info("Inizializzazione Kraken")
        self.init_exchange()
        self.telegram.start_polling()
        fg = fetch_fear_greed()
        self.telegram.send(
            "Bot Portfolio avviato\n"
            "Exchange: Kraken\n"
            f"Timeframe entry: {self.cfg.timeframe}\n"
            f"Dry run: {self.cfg.dry_run}\n"
            f"Max posizioni: {self.cfg.max_open_trades}\n"
            f"Score minimo buy: {self.cfg.buy_score_threshold:.0f}\n"
            f"Volume minimo 24h: {self.cfg.min_24h_quote_volume_eur:,.0f} EUR\n"
            f"Allocazione: normal={self.cfg.alloc_normal*100:.0f}% | "
            f"high={self.cfg.alloc_high*100:.0f}% | extreme={self.cfg.alloc_extreme*100:.0f}%\n"
            f"Sostituzione: score_diff&gt;={self.cfg.replace_score_diff:.0f} | pnl&lt;{self.cfg.replace_pnl_threshold:.0f}%\n"
            f"Fear &amp; Greed: {fg if fg is not None else 'n/d'} "
            f"(range ok: {self.cfg.fear_greed_min}-{self.cfg.fear_greed_max})"
        )
        self.load_markets()
        self.refresh_balance(send=True)

    # ── balance ──────────────────────────────────────────────────────────
    def refresh_balance(self, send: bool = False) -> None:
        try:
            self.last_balance = self.call(self.exchange.fetch_balance)
            self.current_equity = self.estimate_equity(self.last_balance)
            self.risk.update_equity(self.current_equity)
            if send:
                self.telegram.send(self.format_balance())
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.exception("Errore fetch_balance(): %s", exc)
            self.telegram.send(f"Errore saldo Kraken: {exc}")

    def estimate_equity(self, balance: Dict[str, Any]) -> float:
        total  = balance.get("total", {}) or {}
        equity = float(total.get("EUR", 0.0) or 0.0)
        for asset, amount_raw in total.items():
            try:
                asset  = str(asset).upper()
                amount = float(amount_raw or 0.0)
                if amount <= 0 or asset == "EUR":
                    continue
                symbol = f"{asset}/EUR"
                if symbol not in self.markets:
                    continue
                ticker = self.call(self.exchange.fetch_ticker, symbol)
                price  = float(ticker.get("last") or ticker.get("close") or 0.0)
                if price > 0:
                    equity += amount * price
                    self.last_prices[symbol] = price
                time.sleep(self.cfg.per_symbol_delay_seconds)
            except Exception:
                continue
        return equity

    def get_quote_free(self) -> float:
        free       = self.last_balance.get("free", {}) or {}
        quote_free = float(free.get("EUR", 0.0) or 0.0)
        if quote_free <= 0:
            quote_free = float(free.get("ZEUR", 0.0) or 0.0)
        return quote_free

    # ── Fear & Greed guard ───────────────────────────────────────────────
    def _fear_greed_ok(self) -> Tuple[bool, str]:
        fg = fetch_fear_greed()
        if fg is None:
            return True, "F&amp;G n/d (skip)"
        if fg < self.cfg.fear_greed_min:
            return False, f"Fear &amp; Greed troppo basso ({fg} &lt; {self.cfg.fear_greed_min})"
        if fg > self.cfg.fear_greed_max:
            return False, f"Fear &amp; Greed troppo alto ({fg} &gt; {self.cfg.fear_greed_max})"
        return True, f"F&amp;G ok ({fg})"

    # ── BTC regime guard ─────────────────────────────────────────────────
    def _update_btc_regime(self) -> None:
        """
        Aggiorna self.btc_bullish analizzando BTC/EUR sul timeframe entry.
        Se BTC/EUR non è disponibile, regime = True (skip).
        """
        btc_symbol = "BTC/EUR"
        if btc_symbol not in self.markets:
            self.btc_bullish = True
            return
        try:
            ohlcv = self.call(
                self.exchange.fetch_ohlcv,
                btc_symbol,
                self.cfg.timeframe,
                limit=self.cfg.ohlcv_limit,
            )
            bullish, desc = self.strategy.btc_regime(ohlcv)
            self.btc_bullish = bullish
            LOGGER.info("BTC regime: %s | %s", "BULLISH" if bullish else "BEARISH", desc)
            if not bullish:
                self.telegram.send(
                    f"⚠️ BTC BEARISH — nessun nuovo trade\n{desc}",
                    silent=True,
                )
        except Exception as exc:
            LOGGER.warning("Errore BTC regime: %s", exc)
            self.btc_bullish = True   # safe default

    # ── Portfolio replacement logic ───────────────────────────────────────
    def maybe_replace_position(self, signal: Signal) -> bool:
        """
        Se il portfolio è pieno, valuta se sostituire la posizione più debole
        con il nuovo segnale.
        Ritorna True se ha sostituito (il chiamante può quindi aprire il trade).
        """
        if len(self.risk.positions) < self.cfg.max_open_trades:
            return False   # slot libero, nessuna sostituzione necessaria

        result = self.risk.weakest_position(self.last_prices)
        if result is None:
            return False

        weak_symbol, weak_composite = result
        weak_pos = self.risk.positions[weak_symbol]
        weak_price = self.last_prices.get(weak_symbol, weak_pos.entry_price)
        weak_pnl_pct = (weak_price - weak_pos.entry_price) / weak_pos.entry_price * 100.0

        score_diff = signal.score - weak_composite

        should_replace = (
            score_diff >= self.cfg.replace_score_diff
            or weak_pnl_pct <= self.cfg.replace_pnl_threshold
        )

        if not should_replace:
            LOGGER.debug(
                "Nessuna sostituzione: %s (composite=%.1f pnl=%.2f%%) vs %s (score=%.1f diff=%.1f)",
                weak_symbol, weak_composite, weak_pnl_pct,
                signal.symbol, signal.score, score_diff,
            )
            return False

        LOGGER.info(
            "Sostituzione: chiudo %s (composite=%.1f pnl=%.2f%%) per aprire %s (score=%.1f diff=%.1f)",
            weak_symbol, weak_composite, weak_pnl_pct,
            signal.symbol, signal.score, score_diff,
        )

        self.telegram.send(
            f"🔄 SOSTITUZIONE PORTFOLIO\n"
            f"Chiudo: {weak_symbol} | PnL {weak_pnl_pct:.2f}% | composite {weak_composite:.1f}\n"
            f"Apro: {signal.symbol} | score {signal.score:.1f} | diff {score_diff:+.1f}"
        )

        self.close_trade(weak_symbol, f"sostituzione portfolio (nuovo segnale {signal.symbol})", weak_price)
        return True

    # ── Main scan ────────────────────────────────────────────────────────
    def scan_market(self) -> None:
        # Global guards
        fg_ok, fg_reason = self._fear_greed_ok()
        if not fg_ok:
            LOGGER.info("Scan saltato: %s", fg_reason)
            return

        if not self.btc_bullish:
            LOGGER.info("Scan saltato: BTC in regime bearish")
            return

        self.scan_count    += 1
        self.last_scan_start = self.now_iso()

        signals:   List[Signal] = []
        liquid     = 0
        buy_count  = 0
        errors     = 0

        LOGGER.info("Inizio scan portfolio: %s simboli | posizioni aperte: %s/%s",
                    len(self.symbols), len(self.risk.positions), self.cfg.max_open_trades)

        for symbol in list(self.symbols):
            if self.shutdown:
                break
            try:
                ohlcv = self.call(
                    self.exchange.fetch_ohlcv,
                    symbol,
                    self.cfg.timeframe,
                    limit=self.cfg.ohlcv_limit,
                )
                sig = self.strategy.analyze(symbol, ohlcv)
                if not sig:
                    time.sleep(self.cfg.per_symbol_delay_seconds)
                    continue

                signals.append(sig)
                self.last_prices[symbol] = sig.price

                if sig.metrics.get("quote_volume_24h", 0.0) >= self.cfg.min_24h_quote_volume_eur:
                    liquid += 1

                if sig.buy:
                    buy_count += 1
                    LOGGER.info("BUY %s | score %.1f | %s", symbol, sig.score, ", ".join(sig.reasons))
                    self.open_trade(sig)

                time.sleep(self.cfg.per_symbol_delay_seconds)

            except Exception as exc:
                errors += 1
                self.last_error = str(exc)
                LOGGER.warning("Errore scan %s: %s", symbol, exc)
                time.sleep(self.cfg.per_symbol_delay_seconds)

        self.best_signals      = sorted(signals, key=lambda s: s.score, reverse=True)[:self.cfg.top_signals_limit]
        self.liquid_count      = liquid
        self.buy_signals_count = buy_count
        self.last_scan_end     = self.now_iso()

        LOGGER.info("Scan completato: mercati=%s liquidi=%s buy=%s errori=%s",
                    len(self.symbols), liquid, buy_count, errors)

        if time.time() - self.last_signal_report >= self.cfg.telegram_signal_interval_seconds:
            self.last_signal_report = time.time()
            self.telegram.send(self.format_signals(), silent=True)

    # ── Position management ───────────────────────────────────────────────
    def manage_positions(self) -> None:
        for symbol in list(self.risk.positions.keys()):
            if self.shutdown:
                break
            try:
                pos = self.risk.positions.get(symbol)
                if not pos:
                    continue

                ohlcv = self.call(
                    self.exchange.fetch_ohlcv,
                    symbol,
                    self.cfg.timeframe,
                    limit=self.cfg.ohlcv_limit,
                )
                exit_data = self.strategy.exit_signal(ohlcv)
                metrics   = exit_data.get("metrics", {})

                price = float(metrics.get("price") or self.fetch_price(symbol))
                atr   = float(metrics.get("atr")   or pos.entry_price * 0.01)

                updated = self.risk.update_trailing(symbol, price, atr)
                if not updated:
                    continue

                self.last_prices[symbol] = price
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100.0

                # ── Partial TP1 ───────────────────────────────────────────
                if not pos.tp1_done:
                    tp1_price = pos.entry_price * (1.0 + self.cfg.partial_tp_1_percent)
                    if price >= tp1_price:
                        tp1_amount = pos.amount * self.cfg.partial_tp_1_size
                        tp1_amount = float(self.exchange.amount_to_precision(symbol, tp1_amount))
                        if tp1_amount > 0:
                            self._partial_close(symbol, tp1_amount, price, "TP1 parziale")
                            pos.tp1_done = True
                            if pos.stop_loss < pos.entry_price:
                                pos.stop_loss = pos.entry_price   # breakeven
                            self.risk.positions[symbol] = pos
                            self.risk.save()

                # ── Partial TP2 ───────────────────────────────────────────
                if pos.tp1_done and not pos.tp2_done:
                    tp2_price = pos.entry_price * (1.0 + self.cfg.partial_tp_2_percent)
                    if price >= tp2_price:
                        tp2_amount = pos.amount * self.cfg.partial_tp_2_size
                        tp2_amount = float(self.exchange.amount_to_precision(symbol, tp2_amount))
                        if tp2_amount > 0:
                            self._partial_close(symbol, tp2_amount, price, "TP2 parziale")
                            pos.tp2_done = True
                            self.risk.positions[symbol] = pos
                            self.risk.save()

                # ── BTC bearish exit ──────────────────────────────────────
                if not self.btc_bullish:
                    self.close_trade(symbol, "BTC bearish — uscita forzata", price)
                    continue

                # ── Full exit conditions ──────────────────────────────────
                reason = ""
                if price <= updated.stop_loss:
                    reason = f"stop loss ({pnl_pct:.2f}%)"
                elif price <= updated.trailing_stop and updated.trailing_stop > pos.stop_loss:
                    reason = f"trailing stop ({pnl_pct:.2f}%)"
                elif exit_data.get("exit"):
                    reason = str(exit_data.get("reason") or "uscita strategia")
                else:
                    try:
                        entry_dt  = datetime.fromisoformat(pos.entry_time.replace("Z", "+00:00"))
                        age_hours = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600.0
                        if age_hours >= self.cfg.max_trade_hours:
                            reason = f"timeout {self.cfg.max_trade_hours}h ({pnl_pct:.2f}%)"
                    except Exception:
                        pass

                if reason:
                    self.close_trade(symbol, reason, price)

                time.sleep(self.cfg.per_symbol_delay_seconds)

            except Exception as exc:
                self.last_error = str(exc)
                LOGGER.exception("Errore posizione %s: %s", symbol, exc)
                self.telegram.send(f"Errore gestione posizione {symbol}: {exc}")

    # ── Open trade ────────────────────────────────────────────────────────
    def open_trade(self, sig: Signal) -> None:
        try:
            LOGGER.info(">>> Tentativo trade %s | score=%.1f | prezzo=%.6g",
                        sig.symbol, sig.score, sig.price)

            if not self.trading_enabled:
                self.telegram.send(f"Trade bloccato {sig.symbol}: trading disattivato")
                return

            fg_ok, fg_reason = self._fear_greed_ok()
            if not fg_ok:
                self.telegram.send(f"Trade bloccato {sig.symbol}: {fg_reason}")
                return

            if not self.btc_bullish:
                self.telegram.send(f"Trade bloccato {sig.symbol}: BTC bearish")
                return

            # Portfolio pieno? Prova sostituzione
            if len(self.risk.positions) >= self.cfg.max_open_trades:
                replaced = self.maybe_replace_position(sig)
                if not replaced:
                    LOGGER.info("Portfolio pieno, segnale non abbastanza forte per sostituire: %s", sig.symbol)
                    return
                # Attendi che il balance venga aggiornato dopo la chiusura
                time.sleep(2)

            market = self.markets.get(sig.symbol)
            if not market:
                self.telegram.send(f"Trade bloccato {sig.symbol}: mercato non trovato")
                return

            quote = str(market.get("quote", "")).upper()
            base  = str(market.get("base",  "")).upper()

            self.refresh_balance(send=False)
            quote_free = self.get_quote_free()

            LOGGER.info("Saldo disponibile %s: %.4f %s", sig.symbol, quote_free, quote)

            allowed, reason = self.risk.can_open(sig.symbol, quote_free)
            if not allowed:
                self.telegram.send(f"Trade bloccato {sig.symbol}: {reason}")
                return

            capital = self.risk.trade_capital(quote_free, sig)
            capital = self.adjust_capital_for_market_limits(sig.symbol, capital, quote_free)

            if capital <= 0:
                limits   = self.markets.get(sig.symbol, {}).get("limits", {}) or {}
                cost_min = (limits.get("cost", {}) or {}).get("min", "n/d")
                self.telegram.send(
                    f"Trade bloccato {sig.symbol}: capitale insufficiente\n"
                    f"Capitale: {capital:.4f} | Minimo: {cost_min} {quote} | Free: {quote_free:.4f}"
                )
                return

            amount     = capital / sig.price
            amount_raw = amount
            amount     = float(self.exchange.amount_to_precision(sig.symbol, amount))

            if amount <= 0:
                self.telegram.send(f"Trade bloccato {sig.symbol}: quantita arrotondata a 0")
                return

            if not self.check_market_limits(sig.symbol, amount, capital):
                self.telegram.send(f"Trade bloccato {sig.symbol}: limiti mercato non rispettati")
                return

            if self.cfg.dry_run:
                order = {
                    "id": f"dry-buy-{int(time.time())}",
                    "average": sig.price, "filled": amount,
                    "cost": amount * sig.price, "fee": {"cost": 0.0},
                }
            else:
                order = self.call(self.exchange.create_market_buy_order, sig.symbol, amount)

            entry  = float(order.get("average") or order.get("price") or sig.price)
            filled = float(order.get("filled") or amount)
            cost   = float(order.get("cost") or filled * entry)
            fees   = self.extract_fees(order)
            levels = self.risk.levels(entry, float(sig.metrics.get("atr", entry * 0.01)))

            pos = Position(
                symbol=sig.symbol, base=base, quote=quote,
                amount=filled, entry_price=entry,
                entry_time=self.now_iso(),
                stop_loss=levels["stop_loss"],
                take_profit=levels["take_profit"],
                trailing_stop=levels["trailing_stop"],
                highest_price=entry,
                order_id=str(order.get("id", "")),
                quote_cost=cost, fees=fees, score=sig.score,
                tp1_done=False, tp2_done=False,
            )
            self.risk.add_position(pos)

            tp1 = entry * (1.0 + self.cfg.partial_tp_1_percent)
            tp2 = entry * (1.0 + self.cfg.partial_tp_2_percent)

            self.telegram.send(
                f"🟢 <b>TRADE APERTO</b>\n\n"
                f"🪙 Coin: <b>{sig.symbol}</b>\n"
                f"💰 Entry: <code>{entry:.10g}</code>\n"
                f"📦 Qty: <code>{filled:.10g}</code>\n"
                f"💵 Capitale: <b>{cost:.2f} {quote}</b>\n\n"
                f"🛡️ Stop Loss: <code>{pos.stop_loss:.10g}</code>\n"
                f"📈 TP1: <code>{tp1:.10g}</code>\n"
                f"🚀 TP2: <code>{tp2:.10g}</code>\n\n"
                f"⭐ Score: <b>{sig.score:.1f}</b>\n"
                f"📊 Motivi: <i>{', '.join(sig.reasons)}</i>"
            )

        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.exception("Errore apertura trade %s: %s", sig.symbol, exc)
            self.telegram.send(f"Errore apertura trade {sig.symbol}: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────
    def adjust_capital_for_market_limits(self, symbol: str, capital: float, quote_free: float) -> float:
        try:
            available      = max(0.0, quote_free * 0.95)
            if available <= 0:
                return 0.0
            limits         = self.markets.get(symbol, {}).get("limits", {}) or {}
            cost_min_raw   = (limits.get("cost",   {}) or {}).get("min")
            amount_min_raw = (limits.get("amount", {}) or {}).get("min")
            pair_min       = max(self.cfg.min_trade_amount, 0.0)
            if cost_min_raw is not None:
                try:
                    pair_min = max(pair_min, float(cost_min_raw) * 1.01)
                except Exception:
                    pass
            if amount_min_raw is not None:
                price = self.last_prices.get(symbol, 0.0)
                if price > 0:
                    try:
                        pair_min = max(pair_min, float(amount_min_raw) * price * 1.01)
                    except Exception:
                        pass
            if pair_min > available:
                return 0.0
            return max(pair_min, min(capital, available))
        except Exception:
            return min(capital, max(0.0, quote_free * 0.95))

    def _partial_close(self, symbol: str, amount: float, price: float, label: str) -> None:
        try:
            if self.cfg.dry_run:
                order = {
                    "id": f"dry-partial-{int(time.time())}",
                    "average": price, "filled": amount,
                    "cost": amount * price, "fee": {"cost": 0.0},
                }
            else:
                order = self.call(self.exchange.create_market_sell_order, symbol, amount)
            exit_price = float(order.get("average") or order.get("price") or price)
            pos        = self.risk.positions.get(symbol)
            entry_p    = pos.entry_price if pos else price
            pnl_pct    = (exit_price - entry_p) / entry_p * 100.0
            self.telegram.send(
                f"📤 {label} — {symbol}\n"
                f"Qty venduta: {amount:.6g}\n"
                f"Prezzo: {exit_price:.6g}\n"
                f"PnL parziale: {pnl_pct:.2f}%"
            )
        except Exception as exc:
            LOGGER.exception("Errore chiusura parziale %s: %s", symbol, exc)
            self.telegram.send(f"Errore {label} {symbol}: {exc}")

    def close_trade(self, symbol: str, reason: str, fallback_price: float) -> None:
        try:
            pos = self.risk.positions.get(symbol)
            if not pos:
                return
            self.refresh_balance(send=False)
            free      = self.last_balance.get("free", {}) or {}
            available = float(free.get(pos.base, pos.amount) or 0.0)
            amount    = min(pos.amount, available if available > 0 else pos.amount)
            amount    = float(self.exchange.amount_to_precision(symbol, amount))
            if amount <= 0:
                self.telegram.send(f"Impossibile chiudere {symbol}: saldo non disponibile")
                return
            if self.cfg.dry_run:
                order = {
                    "id": f"dry-sell-{int(time.time())}",
                    "average": fallback_price, "filled": amount,
                    "cost": amount * fallback_price, "fee": {"cost": 0.0},
                }
            else:
                order = self.call(self.exchange.create_market_sell_order, symbol, amount)
            exit_price = float(order.get("average") or order.get("price") or fallback_price)
            fees       = self.extract_fees(order)
            closed     = self.risk.close(symbol, exit_price, reason, fees, str(order.get("id", "")))
            if not closed:
                return

            emoji = "🟢" if closed['net_pnl'] >= 0 else "🔴"

            self.telegram.send(
                f"{emoji} <b>TRADE CHIUSO</b>\n\n"
                f"🪙 Coin: <b>{symbol}</b>\n"
                f"📌 Motivo: <b>{reason}</b>\n\n"
                f"💰 Entry: <code>{closed['entry_price']:.10g}</code>\n"
                f"💵 Exit: <code>{closed['exit_price']:.10g}</code>\n\n"
                f"📈 PnL: <b>{closed['net_pnl']:.2f} {closed['quote']}</b>\n"
                f"📊 Return: <b>{closed['pnl_percent']:.2f}%</b>\n\n"
                f"⚠️ Loss streak: <b>{self.risk.consecutive_losses}</b>"
            )
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.exception("Errore chiusura trade %s: %s", symbol, exc)
            self.telegram.send(f"Errore chiusura trade {symbol}: {exc}")

    def fetch_price(self, symbol: str) -> float:
        try:
            ticker = self.call(self.exchange.fetch_ticker, symbol)
            price  = float(ticker.get("last") or ticker.get("close") or 0.0)
            if price > 0:
                self.last_prices[symbol] = price
            return price
        except Exception:
            return self.last_prices.get(symbol, 0.0)

    def check_market_limits(self, symbol: str, amount: float, cost: float) -> bool:
        try:
            limits     = self.markets.get(symbol, {}).get("limits", {}) or {}
            amount_min = (limits.get("amount", {}) or {}).get("min")
            cost_min   = (limits.get("cost",   {}) or {}).get("min")
            if amount_min is not None and amount < float(amount_min):
                return False
            if cost_min is not None and cost < float(cost_min):
                return False
            return True
        except Exception:
            return False

    def extract_fees(self, order: Dict[str, Any]) -> float:
        try:
            total = 0.0
            fee   = order.get("fee")
            if isinstance(fee, dict):
                total += float(fee.get("cost") or 0.0)
            for item in order.get("fees") or []:
                if isinstance(item, dict):
                    total += float(item.get("cost") or 0.0)
            return total
        except Exception:
            return 0.0

    # ── Formatters ────────────────────────────────────────────────────────
    def format_balance(self) -> str:
        free = self.last_balance.get("free", {}) or {}
        total = self.last_balance.get("total", {}) or {}

        eur_free = float(free.get("EUR", 0) or 0)
        eur_total = float(total.get("EUR", 0) or 0)

        return (
            "💰 <b>ACCOUNT OVERVIEW</b>\n\n"
            f"🏦 Equity stimata: <b>{self.format_money(self.current_equity)}</b>\n"
            f"💵 EUR Free: <b>{self.format_money(eur_free)}</b>\n"
            f"📦 EUR Totale: <b>{self.format_money(eur_total)}</b>\n\n"
            f"📈 PnL giornaliero: <b>{self.format_money(self.risk.daily_realized_pnl)}</b>\n"
            f"⚠️ Drawdown: <b>{self.risk.current_drawdown * 100:.2f}%</b>"
        )

    def format_signals(self) -> str:
        if not self.best_signals:
            return "📡 <b>SEGNALI</b>\n\nNessun segnale disponibile"

        fg = fetch_fear_greed()

        lines = [
            "📡 <b>TOP SIGNALS</b>\n",
            f"BTC: <b>{self.emoji_market()}</b>",
            f"Fear &amp; Greed: <b>{fg if fg is not None else 'n/d'}</b>",
            f"Portfolio: <b>{len(self.risk.positions)}/{self.cfg.max_open_trades}</b>",
            "━━━━━━━━━━━━━━━━━━",
        ]

        for i, sig in enumerate(self.best_signals, 1):
            m = sig.metrics

            icon = "🟢" if sig.buy else "⚪"
            status = "BUY" if sig.buy else "WATCH"

            lines.append(
                f"{icon} <b>{i}. {sig.symbol}</b> [{status}]\n"
                f"💰 Price: <code>{sig.price:.6g}</code>\n"
                f"⭐ Score: <b>{sig.score:.1f}</b>\n"
                f"📈 RSI: <b>{m.get('rsi',0):.1f}</b>\n"
                f"🚀 Momentum: <b>{m.get('momentum',0)*100:+.2f}%</b>\n"
                f"📊 Volume: <b>x{m.get('volume_ratio',0):.1f}</b>\n"
                f"🌊 ATR: <b>{m.get('atr_percent',0)*100:.2f}%</b>\n"
            )

        return "\n".join(lines)

    # ── Telegram commands ─────────────────────────────────────────────────
    def cmd_balance(self) -> str:
        self.refresh_balance(send=False)
        return self.format_balance()

    def cmd_portfolio(self) -> str:
        if not self.risk.positions:
            return (
                "💼 <b>PORTFOLIO</b>\n\n"
                "⚪ Nessuna posizione aperta"
            )

        lines = [
            f"💼 <b>PORTFOLIO</b> [{len(self.risk.positions)}/{self.cfg.max_open_trades}]\n"
        ]

        total_pnl = 0.0

        for pos in self.risk.positions.values():
            price = self.last_prices.get(pos.symbol, pos.entry_price)

            pnl = (price - pos.entry_price) * pos.amount
            pnl_pct = (price - pos.entry_price) / pos.entry_price * 100.0

            total_pnl += pnl

            emoji = self.emoji_pnl(pnl)

            lines.append(
                f"{emoji} <b>{pos.symbol}</b>\n"
                f"Entry: <code>{pos.entry_price:.6g}</code>\n"
                f"Last: <code>{price:.6g}</code>\n"
                f"PnL: <b>{pnl:+.2f} EUR ({pnl_pct:+.2f}%)</b>\n"
                f"Qty: <code>{pos.amount:.6g}</code>\n"
                f"SL: <code>{pos.stop_loss:.6g}</code>\n"
                f"TS: <code>{pos.trailing_stop:.6g}</code>\n"
                f"TP1 {'✅' if pos.tp1_done else '⬜'} | TP2 {'✅' if pos.tp2_done else '⬜'}\n"
                f"Score: <b>{pos.score:.1f}</b>\n"
            )

        lines.append(
            f"\n━━━━━━━━━━━━━━━━━━\n"
            f"📈 Totale aperto: <b>{total_pnl:+.2f} EUR</b>"
        )

        return "\n".join(lines)

    def cmd_status(self) -> str:
        fg = fetch_fear_greed()

        return (
            "🧠 <b>CRYPTO NO AUTO CHECK BOT</b>\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📊 <b>MARKET</b>\n"
            f"BTC Regime: <b>{self.emoji_market()}</b>\n"
            f"Fear &amp; Greed: <b>{fg if fg is not None else 'n/d'}</b>\n"
            f"Trading: <b>{'🟢 ATTIVO' if self.trading_enabled else '🔴 PAUSA'}</b>\n\n"
            "💼 <b>PORTFOLIO</b>\n"
            f"Posizioni: <b>{len(self.risk.positions)}/{self.cfg.max_open_trades}</b>\n"
            f"Equity: <b>{self.format_money(self.current_equity)}</b>\n"
            f"PnL Day: <b>{self.format_money(self.risk.daily_realized_pnl)}</b>\n"
            f"Drawdown: <b>{self.risk.current_drawdown * 100:.2f}%</b>\n\n"
            "⚙️ <b>STRATEGY</b>\n"
            f"Buy Score: <b>{self.cfg.buy_score_threshold:.0f}+</b>\n"
            f"RSI Buy: <b>{self.cfg.rsi_buy_min}-{self.cfg.rsi_buy_max}</b>\n"
            f"Momentum: <b>{self.cfg.momentum_min*100:.1f}% → {self.cfg.momentum_max*100:.1f}%</b>\n"
            f"Volume Spike: <b>{self.cfg.volume_breakout_multiplier}x</b>\n\n"
            "🛡️ <b>RISK</b>\n"
            f"Loss Streak: <b>{self.risk.consecutive_losses}</b>\n"
            f"Risk Pause: <b>{self.risk.pause_minutes()} min</b>\n"
            f"Dry Run: <b>{self.cfg.dry_run}</b>\n\n"
            "🖥️ <b>SYSTEM</b>\n"
            f"Scan: <b>{self.scan_count}</b>\n"
            f"Last Scan: <b>{self.last_scan_end}</b>\n"
            f"Last Error: <code>{self.last_error}</code>"
        )

    def cmd_trades(self) -> str:
        return self.cmd_portfolio() + "\n\nULTIMI CHIUSI\n" + "\n".join(
            f"{t['symbol']} | {float(t['net_pnl']):.2f} {t['quote']} "
            f"| {float(t['pnl_percent']):.2f}% | {t['reason']}"
            for t in reversed(self.risk.closed_trades[-5:])
        ) if self.risk.closed_trades else self.cmd_portfolio() + "\n\nNessun trade chiuso."

    def cmd_profit(self) -> str:
        unrealized = self.risk.unrealized_pnl(self.last_prices)
        return (
            "PROFITTO\n"
            f"PnL giornaliero realizzato: {self.risk.daily_realized_pnl:.2f} EUR\n"
            f"PnL aperto stimato: {unrealized:.2f} EUR\n"
            f"PnL totale chiuso: {self.risk.total_closed_pnl():.2f} EUR\n"
            f"Drawdown: {self.risk.current_drawdown * 100:.2f}%\n"
            f"Trade chiusi totali: {len(self.risk.closed_trades)}"
        )

    def cmd_market(self) -> str:
        fg = fetch_fear_greed()
        return (
            "MERCATO\n"
            "Exchange: Kraken\n"
            f"BTC: {'🟢 BULLISH' if self.btc_bullish else '🔴 BEARISH'}\n"
            f"Coppie EUR filtrate: {len(self.symbols)}\n"
            f"Coin liquide ultimo scan: {self.liquid_count}\n"
            f"Fear &amp; Greed: {fg if fg is not None else 'n/d'}\n"
            f"Scan completati: {self.scan_count}"
        )

    def cmd_start(self) -> str:
        self.trading_enabled = True
        return "Trading riattivato."

    def cmd_stop(self) -> str:
        self.trading_enabled = False
        return "Trading sospeso. Le posizioni aperte restano gestite."

    def cmd_chiudi(self, text: str) -> str:
        parts = text.strip().split()
        if len(parts) < 2:
            if not self.risk.positions:
                return "Nessun trade aperto."
            return f"Uso: /chiudi SIMBOLO\nAperti: {', '.join(self.risk.positions.keys())}"
        symbol = parts[1].upper()
        if symbol not in self.risk.positions:
            return (f"Nessuna posizione su {symbol}.\n"
                    f"Aperti: {', '.join(self.risk.positions.keys()) or 'nessuno'}")
        price = self.fetch_price(symbol)
        if price <= 0:
            price = self.risk.positions[symbol].entry_price
        self.close_trade(symbol, "chiusura manuale", price)
        return f"Chiusura manuale {symbol} avviata al prezzo ~{price:.6g}"

    def cmd_diagnostica(self) -> str:
        lines = ["DIAGNOSTICA BOT"]
        try:
            self.refresh_balance(send=False)
            free  = self.last_balance.get("free",  {}) or {}
            total = self.last_balance.get("total", {}) or {}
            nzf   = {k: v for k, v in free.items()  if v and float(v) > 0}
            nzt   = {k: v for k, v in total.items() if v and float(v) > 0}
            lines.append("\nSALDO (valori > 0):")
            for k, v in nzf.items():
                lines.append(f"  free[{k}] = {float(v):.6f}")
            for k, v in nzt.items():
                lines.append(f"  total[{k}] = {float(v):.6f}")
            lines.append(f"Equity stimata: {self.current_equity:.4f} EUR")
        except Exception as exc:
            lines.append(f"Errore fetch saldo: {exc}")

        fg = fetch_fear_greed()
        lines.append(
            f"\nFEAR &amp; GREED: {fg if fg is not None else 'n/d'} "
            f"(range: {self.cfg.fear_greed_min}-{self.cfg.fear_greed_max})"
        )
        lines.append(f"BTC regime: {'BULLISH' if self.btc_bullish else 'BEARISH'}")

        lines.append("\nPORTFOLIO:")
        if self.risk.positions:
            for sym, pos in self.risk.positions.items():
                price   = self.last_prices.get(sym, pos.entry_price)
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100.0
                composite = pos.score + pnl_pct
                lines.append(
                    f"  {sym}: entry={pos.entry_price:.6g} last={price:.6g} "
                    f"pnl={pnl_pct:+.2f}% score={pos.score:.1f} composite={composite:.1f}"
                )
        else:
            lines.append("  (nessuna posizione)")

        lines.append("\nCONFIG:")
        lines.append(f"  alloc: normal={self.cfg.alloc_normal*100:.0f}% "
                     f"high={self.cfg.alloc_high*100:.0f}% extreme={self.cfg.alloc_extreme*100:.0f}%")
        lines.append(f"  replace_score_diff: {self.cfg.replace_score_diff}")
        lines.append(f"  replace_pnl_threshold: {self.cfg.replace_pnl_threshold}%")
        lines.append(f"  buy_score_threshold: {self.cfg.buy_score_threshold}")
        lines.append(f"  rsi_buy: {self.cfg.rsi_buy_min}-{self.cfg.rsi_buy_max}")
        lines.append(f"  momentum: {self.cfg.momentum_min*100:.1f}%-{self.cfg.momentum_max*100:.1f}%")
        lines.append(f"  volume_spike: {self.cfg.volume_breakout_multiplier}x")

        since_last = int(time.time() - self.risk.last_trade_at) if self.risk.last_trade_at > 0 else -1
        lines.append(f"\nRISK: consecutive_losses={self.risk.consecutive_losses} "
                     f"pause={self.risk.pause_minutes()}min "
                     f"since_last_trade={since_last}s")

        if self.best_signals:
            lines.append("\nLIMITI MERCATO (top 5):")
            for sig in self.best_signals[:5]:
                try:
                    mkt      = self.markets.get(sig.symbol, {})
                    limits   = mkt.get("limits", {}) or {}
                    cost_min = (limits.get("cost",   {}) or {}).get("min", "n/d")
                    amt_min  = (limits.get("amount", {}) or {}).get("min", "n/d")
                    lines.append(f"  {sig.symbol}: cost_min={cost_min} amt_min={amt_min}")
                except Exception:
                    lines.append(f"  {sig.symbol}: errore limiti")

        return "\n".join(lines)

    def cmd_help(self) -> str:
        return (
            "COMANDI\n"
            "/saldo — saldo account\n"
            "/status — stato bot\n"
            "/portfolio — posizioni aperte dettagliate\n"
            "/trades — portfolio + ultimi chiusi\n"
            "/profitto — PnL\n"
            "/mercato — info mercato e BTC regime\n"
            "/segnali — top segnali\n"
            "/diagnostica — debug completo\n"
            "/start — riattiva trading\n"
            "/stop — sospendi trading\n"
            "/chiudi SIMBOLO — chiusura manuale\n"
            "/help — questo messaggio"
        )

    # ── Shutdown / run ────────────────────────────────────────────────────
    def request_shutdown(self, signum: int, frame: Any) -> None:
        self.shutdown = True

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.request_shutdown)
        signal.signal(signal.SIGINT,  self.request_shutdown)
        LOGGER.info("Bot Portfolio process avviato")

        while not self.shutdown:
            try:
                if not self.kraken_env_ok():
                    time.sleep(60)
                    continue

                self.connect()
                btc_regime_check_counter = 0

                while not self.shutdown:
                    if time.time() - self.last_market_reload > self.cfg.market_refresh_seconds:
                        self.load_markets()

                    # Aggiorna regime BTC ogni 5 iterazioni (~5 min con scan_interval=60s)
                    btc_regime_check_counter += 1
                    if btc_regime_check_counter >= 5:
                        self._update_btc_regime()
                        btc_regime_check_counter = 0

                    self.refresh_balance(send=False)
                    self.manage_positions()

                    if self.trading_enabled and not self.risk.daily_stop_hit():
                        self.scan_market()
                    else:
                        LOGGER.info("Trading in pausa")

                    time.sleep(self.cfg.scan_interval_seconds)

            except Exception as exc:
                self.last_error = str(exc)
                LOGGER.exception("Errore loop principale: %s", exc)
                self.telegram.send(f"Errore loop principale: {exc}")
                time.sleep(self.cfg.scan_interval_seconds)

        self.telegram.send("Bot Portfolio arrestato")


if __name__ == "__main__":
    print("Avvio container Railway...", flush=True)
    start_health_server()
    bot = KrakenTradingBot()
    bot.run()
