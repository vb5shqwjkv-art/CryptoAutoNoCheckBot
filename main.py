import os
import time
import json
import math
import signal
import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

import ccxt
import numpy as np
import pandas as pd
import requests
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


# =========================
# CONFIG
# =========================

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = str(os.getenv(name, str(default))).lower().strip()
    return value in ("1", "true", "yes", "y", "on")


@dataclass
class Config:
    KRAKEN_API_KEY: str = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_SECRET: str = os.getenv("KRAKEN_SECRET", "")
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    TIMEFRAME: str = os.getenv("TIMEFRAME", "15m")
    OHLCV_LIMIT: int = env_int("OHLCV_LIMIT", 150)
    SCAN_INTERVAL_SECONDS: int = env_int("SCAN_INTERVAL_SECONDS", 60)
    PER_SYMBOL_DELAY_SECONDS: float = env_float("PER_SYMBOL_DELAY_SECONDS", 1.2)

    MIN_24H_QUOTE_VOLUME_USD: float = env_float("MIN_24H_QUOTE_VOLUME_USD", 250000)
    MAX_OPEN_TRADES: int = min(env_int("MAX_OPEN_TRADES", 3), 3)
    MAX_CAPITAL_PER_TRADE: float = min(env_float("MAX_CAPITAL_PER_TRADE", 0.02), 0.02)
    DAILY_MAX_LOSS: float = min(env_float("DAILY_MAX_LOSS", 0.05), 0.05)

    EMA_FAST: int = env_int("EMA_FAST", 20)
    EMA_SLOW: int = env_int("EMA_SLOW", 50)
    RSI_PERIOD: int = env_int("RSI_PERIOD", 14)
    ATR_PERIOD: int = env_int("ATR_PERIOD", 14)

    RSI_BUY_MIN: float = env_float("RSI_BUY_MIN", 50)
    RSI_BUY_MAX: float = env_float("RSI_BUY_MAX", 70)
    RSI_EXIT: float = env_float("RSI_EXIT", 78)

    VOLUME_WINDOW: int = env_int("VOLUME_WINDOW", 20)
    VOLUME_BREAKOUT_MULTIPLIER: float = env_float("VOLUME_BREAKOUT_MULTIPLIER", 1.35)
    BREAKOUT_LOOKBACK: int = env_int("BREAKOUT_LOOKBACK", 20)
    MOMENTUM_LOOKBACK: int = env_int("MOMENTUM_LOOKBACK", 5)

    STOP_LOSS_ATR_MULTIPLIER: float = env_float("STOP_LOSS_ATR_MULTIPLIER", 2.0)
    TAKE_PROFIT_ATR_MULTIPLIER: float = env_float("TAKE_PROFIT_ATR_MULTIPLIER", 3.0)
    TRAILING_ATR_MULTIPLIER: float = env_float("TRAILING_ATR_MULTIPLIER", 2.0)

    MAX_STOP_LOSS_PERCENT: float = env_float("MAX_STOP_LOSS_PERCENT", 0.04)
    MIN_TAKE_PROFIT_PERCENT: float = env_float("MIN_TAKE_PROFIT_PERCENT", 0.025)

    MAX_CONSECUTIVE_LOSSES: int = env_int("MAX_CONSECUTIVE_LOSSES", 3)
    LOSS_COOLDOWN_SECONDS: int = env_int("LOSS_COOLDOWN_SECONDS", 10800)
    MIN_SECONDS_BETWEEN_TRADES: int = env_int("MIN_SECONDS_BETWEEN_TRADES", 600)
    SYMBOL_COOLDOWN_SECONDS: int = env_int("SYMBOL_COOLDOWN_SECONDS", 3600)

    DRY_RUN: bool = env_bool("DRY_RUN", False)
    STATE_FILE: str = os.getenv("STATE_FILE", "bot_state.json")

    TOP_SIGNALS_LIMIT: int = env_int("TOP_SIGNALS_LIMIT", 10)
    TELEGRAM_SIGNAL_INTERVAL_SECONDS: int = env_int("TELEGRAM_SIGNAL_INTERVAL_SECONDS", 1800)

    QUOTES = {"USD", "USDT"}
    STABLECOINS = {
        "USDT", "USDC", "DAI", "TUSD", "USDP", "USDD", "BUSD", "FDUSD",
        "PYUSD", "GUSD", "LUSD", "FRAX", "SUSD", "USDS", "USDE", "SDAI"
    }
    FIAT = {"USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD", "NZD", "SGD"}
    LEVERAGED_WORDS = {
        "UP", "DOWN", "BULL", "BEAR", "LONG", "SHORT",
        "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S"
    }

    def validate(self):
        missing = []
        if not self.KRAKEN_API_KEY:
            missing.append("KRAKEN_API_KEY")
        if not self.KRAKEN_SECRET:
            missing.append("KRAKEN_SECRET")
        if not self.TELEGRAM_TOKEN:
            missing.append("TELEGRAM_TOKEN")
        if not self.TELEGRAM_CHAT_ID:
            missing.append("TELEGRAM_CHAT_ID")
        if missing:
            raise RuntimeError("Variabili ambiente mancanti: " + ", ".join(missing))


# =========================
# TELEGRAM
# =========================

class TelegramPanel:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.base_url = f"https://api.telegram.org/bot{cfg.TELEGRAM_TOKEN}"
        self.chat_id = str(cfg.TELEGRAM_CHAT_ID)
        self.handlers: Dict[str, Callable[[str], str]] = {}
        self.session = requests.Session()
        self.offset = None
        self.stop_event = threading.Event()

    def send(self, text: str, silent: bool = False):
        try:
            if not text:
                return
            for i in range(0, len(text), 3900):
                self.session.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text[i:i + 3900],
                        "disable_notification": silent,
                    },
                    timeout=15,
                )
                time.sleep(0.2)
        except Exception as e:
            self.logger.error("Telegram send error: %s", e)

    def register(self, command: str, func: Callable[[str], str]):
        self.handlers[command.lower().replace("/", "")] = func

    def start(self):
        t = threading.Thread(target=self.loop, daemon=True)
        t.start()

    def loop(self):
        while not self.stop_event.is_set():
            try:
                params = {"timeout": 25, "allowed_updates": ["message"]}
                if self.offset is not None:
                    params["offset"] = self.offset

                r = self.session.get(
                    f"{self.base_url}/getUpdates",
                    params=params,
                    timeout=35,
                )
                data = r.json()

                for update in data.get("result", []):
                    self.offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    text = str(msg.get("text", "")).strip()

                    if chat_id != self.chat_id or not text.startswith("/"):
                        continue

                    cmd = text.split()[0].split("@")[0].replace("/", "").lower()
                    handler = self.handlers.get(cmd)

                    if handler:
                        reply = handler(text)
                        if reply:
                            self.send(reply)
                    else:
                        self.send("Comando non riconosciuto. Usa /help.")
            except Exception as e:
                self.logger.warning("Telegram polling error: %s", e)
                time.sleep(5)


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class Signal:
    symbol: str
    price: float
    score: float
    buy: bool
    reasons: List[str]
    metrics: Dict[str, float]
    timestamp: str


@dataclass
class Position:
    symbol: str
    base: str
    quote: str
    amount: float
    entry_price: float
    entry_time: str
    stop_loss: float
    take_profit: float
    trailing_stop: float
    highest_price: float
    order_id: str
    quote_cost: float
    fees: float
    score: float


# =========================
# STRATEGY
# =========================

class Strategy:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def to_df(self, ohlcv: List[List[float]]) -> pd.DataFrame:
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna().reset_index(drop=True)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 80:
            return pd.DataFrame()

        df = df.copy()
        close = df["close"]

        df["ema20"] = EMAIndicator(close, window=self.cfg.EMA_FAST).ema_indicator()
        df["ema50"] = EMAIndicator(close, window=self.cfg.EMA_SLOW).ema_indicator()
        df["rsi"] = RSIIndicator(close, window=self.cfg.RSI_PERIOD).rsi()
        df["atr"] = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.cfg.ATR_PERIOD,
        ).average_true_range()

        df["volume_avg"] = df["volume"].rolling(self.cfg.VOLUME_WINDOW).mean()
        df["breakout_high"] = df["high"].shift(1).rolling(self.cfg.BREAKOUT_LOOKBACK).max()
        df["momentum"] = df["close"].pct_change(self.cfg.MOMENTUM_LOOKBACK)
        df["atr_percent"] = df["atr"] / df["close"]
        df["quote_volume"] = df["close"] * df["volume"]

        return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    def analyze(self, symbol: str, ohlcv: List[List[float]]) -> Optional[Signal]:
        try:
            df = self.indicators(self.to_df(ohlcv))
            if df.empty:
                return None

            row = df.iloc[-1]

            price = float(row["close"])
            ema20 = float(row["ema20"])
            ema50 = float(row["ema50"])
            rsi = float(row["rsi"])
            atr = float(row["atr"])
            atr_percent = float(row["atr_percent"])
            volume = float(row["volume"])
            volume_avg = float(row["volume_avg"])
            breakout_high = float(row["breakout_high"])
            momentum = float(row["momentum"])
            quote_volume_24h = float(df["quote_volume"].tail(96).sum())

            liquid = quote_volume_24h >= self.cfg.MIN_24H_QUOTE_VOLUME_USD
            trend_up = ema20 > ema50
            rsi_ok = self.cfg.RSI_BUY_MIN <= rsi <= self.cfg.RSI_BUY_MAX
            volume_breakout = volume > volume_avg * self.cfg.VOLUME_BREAKOUT_MULTIPLIER
            breakout = price > breakout_high
            momentum_positive = momentum > 0
            volatility_ok = 0.002 <= atr_percent <= 0.18

            score = 0.0
            reasons = []

            if liquid:
                score += 15
                reasons.append("liquido")
            if trend_up:
                score += 20
                reasons.append("EMA20 sopra EMA50")
            if rsi_ok:
                score += 15
                reasons.append("RSI valido")
            if volume_breakout:
                score += 20
                reasons.append("volume breakout")
            if breakout:
                score += 20
                reasons.append("breakout rialzista")
            if momentum_positive:
                score += 10
                reasons.append("momentum positivo")
            if volatility_ok:
                score += 10
                reasons.append("volatilita valida")

            buy = all([
                liquid,
                trend_up,
                rsi_ok,
                volume_breakout,
                breakout,
                momentum_positive,
                volatility_ok,
            ])

            return Signal(
                symbol=symbol,
                price=price,
                score=round(score, 2),
                buy=buy,
                reasons=reasons,
                metrics={
                    "ema20": ema20,
                    "ema50": ema50,
                    "rsi": rsi,
                    "atr": atr,
                    "atr_percent": atr_percent,
                    "volume_ratio": volume / max(volume_avg, 1e-12),
                    "momentum": momentum,
                    "quote_volume_24h": quote_volume_24h,
                },
                timestamp=str(row["timestamp"]),
            )
        except Exception:
            return None

    def exit_signal(self, ohlcv: List[List[float]]) -> Dict[str, Any]:
        try:
            df = self.indicators(self.to_df(ohlcv))
            if df.empty:
                return {"exit": False, "reason": "", "metrics": {}}

            row = df.iloc[-1]
            price = float(row["close"])
            ema20 = float(row["ema20"])
            ema50 = float(row["ema50"])
            rsi = float(row["rsi"])
            momentum = float(row["momentum"])
            atr = float(row["atr"])

            metrics = {
                "price": price,
                "rsi": rsi,
                "momentum": momentum,
                "atr": atr,
            }

            if ema20 < ema50 and momentum < 0:
                return {"exit": True, "reason": "inversione trend", "metrics": metrics}

            if rsi >= self.cfg.RSI_EXIT and momentum <= 0:
                return {"exit": True, "reason": "RSI troppo alto", "metrics": metrics}

            return {"exit": False, "reason": "", "metrics": metrics}
        except Exception:
            return {"exit": False, "reason": "", "metrics": {}}


# =========================
# RISK
# =========================

class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.current_day = self.today()
        self.daily_start_equity = 0.0
        self.daily_realized_pnl = 0.0
        self.current_drawdown = 0.0
        self.consecutive_losses = 0
        self.pause_until = 0.0
        self.last_trade_at = 0.0
        self.symbol_last_trade_at: Dict[str, float] = {}
        self.load()

    def today(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def load(self):
        try:
            path = Path(self.cfg.STATE_FILE)
            if not path.exists():
                return

            data = json.loads(path.read_text(encoding="utf-8"))

            self.positions = {
                symbol: Position(**pos)
                for symbol, pos in data.get("positions", {}).items()
            }
            self.closed_trades = data.get("closed_trades", [])[-500:]
            self.current_day = data.get("current_day", self.today())
            self.daily_start_equity = float(data.get("daily_start_equity", 0))
            self.daily_realized_pnl = float(data.get("daily_realized_pnl", 0))
            self.consecutive_losses = int(data.get("consecutive_losses", 0))
            self.pause_until = float(data.get("pause_until", 0))
            self.last_trade_at = float(data.get("last_trade_at", 0))
            self.symbol_last_trade_at = data.get("symbol_last_trade_at", {})
        except Exception:
            pass

    def save(self):
        try:
            data = {
                "positions": {
                    symbol: asdict(pos)
                    for symbol, pos in self.positions.items()
                },
                "closed_trades": self.closed_trades[-500:],
                "current_day": self.current_day,
                "daily_start_equity": self.daily_start_equity,
                "daily_realized_pnl": self.daily_realized_pnl,
                "consecutive_losses": self.consecutive_losses,
                "pause_until": self.pause_until,
                "last_trade_at": self.last_trade_at,
                "symbol_last_trade_at": self.symbol_last_trade_at,
            }
            Path(self.cfg.STATE_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def update_equity(self, equity: float):
        today = self.today()

        if today != self.current_day:
            self.current_day = today
            self.daily_start_equity = equity
            self.daily_realized_pnl = 0
            self.current_drawdown = 0
            self.save()

        if self.daily_start_equity <= 0 and equity > 0:
            self.daily_start_equity = equity
            self.save()

        if self.daily_start_equity > 0:
            self.current_drawdown = max(
                0,
                (self.daily_start_equity - equity) / self.daily_start_equity,
            )

    def daily_stop_hit(self) -> bool:
        if self.daily_start_equity <= 0:
            return False

        realized_loss = self.daily_realized_pnl <= -self.daily_start_equity * self.cfg.DAILY_MAX_LOSS
        drawdown_loss = self.current_drawdown >= self.cfg.DAILY_MAX_LOSS

        return realized_loss or drawdown_loss

    def pause_active(self) -> bool:
        return time.time() < self.pause_until

    def pause_minutes(self) -> int:
        return int(max(0, self.pause_until - time.time()) // 60)

    def can_open(self, symbol: str, equity: float, quote_free: float) -> Tuple[bool, str]:
        now = time.time()

        if symbol in self.positions:
            return False, "posizione gia aperta"

        if len(self.positions) >= self.cfg.MAX_OPEN_TRADES:
            return False, "max 3 trade contemporanei raggiunto"

        if quote_free <= 0:
            return False, "saldo insufficiente"

        if self.daily_stop_hit():
            return False, "stop giornaliero perdita 5% attivo"

        if self.pause_active():
            return False, f"pausa rischio attiva {self.pause_minutes()} min"

        if now - self.last_trade_at < self.cfg.MIN_SECONDS_BETWEEN_TRADES:
            return False, "anti-overtrading attivo"

        if now - float(self.symbol_last_trade_at.get(symbol, 0)) < self.cfg.SYMBOL_COOLDOWN_SECONDS:
            return False, "cooldown simbolo attivo"

        return True, "ok"

    def trade_capital(self, equity: float, quote_free: float) -> float:
        return max(0, min(equity * self.cfg.MAX_CAPITAL_PER_TRADE, quote_free * 0.95))

    def levels(self, entry: float, atr: float) -> Dict[str, float]:
        atr = max(atr, entry * 0.002)

        stop_atr = entry - atr * self.cfg.STOP_LOSS_ATR_MULTIPLIER
        stop_percent = entry * (1 - self.cfg.MAX_STOP_LOSS_PERCENT)
        stop_loss = max(stop_atr, stop_percent)

        take_atr = entry + atr * self.cfg.TAKE_PROFIT_ATR_MULTIPLIER
        take_percent = entry * (1 + self.cfg.MIN_TAKE_PROFIT_PERCENT)
        take_profit = max(take_atr, take_percent)

        trailing_stop = max(stop_loss, entry - atr * self.cfg.TRAILING_ATR_MULTIPLIER)

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop": trailing_stop,
        }

    def add_position(self, pos: Position):
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

        new_trailing = pos.highest_price - atr * self.cfg.TRAILING_ATR_MULTIPLIER

        if new_trailing > pos.trailing_stop:
            pos.trailing_stop = new_trailing

        self.positions[symbol] = pos
        self.save()
        return pos

    def close(self, symbol: str, exit_price: float, reason: str, fees: float, order_id: str):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return None

        gross = (exit_price - pos.entry_price) * pos.amount
        net = gross - pos.fees - fees
        pnl_percent = net / max(pos.entry_price * pos.amount, 1e-12) * 100

        trade = {
            "symbol": symbol,
            "base": pos.base,
            "quote": pos.quote,
            "amount": pos.amount,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "entry_time": pos.entry_time,
            "exit_time": self.now_iso(),
            "reason": reason,
            "net_pnl": net,
            "pnl_percent": pnl_percent,
            "fees": pos.fees + fees,
            "entry_order_id": pos.order_id,
            "exit_order_id": order_id,
        }

        self.closed_trades.append(trade)
        self.daily_realized_pnl += net

        now = time.time()
        self.last_trade_at = now
        self.symbol_last_trade_at[symbol] = now

        if net < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.MAX_CONSECUTIVE_LOSSES:
                self.pause_until = now + self.cfg.LOSS_COOLDOWN_SECONDS
        else:
            self.consecutive_losses = 0

        self.save()
        return trade

    def unrealized_pnl(self, prices: Dict[str, float]) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.entry_price)
            total += (price - pos.entry_price) * pos.amount
        return total

    def total_closed_pnl(self) -> float:
        return sum(float(t.get("net_pnl", 0)) for t in self.closed_trades)


# =========================
# BOT
# =========================

class KrakenTradingBot:
    def __init__(self):
        self.cfg = Config()
        self.cfg.validate()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        self.logger = logging.getLogger("KrakenBot")

        self.exchange = None
        self.strategy = Strategy(self.cfg)
        self.risk = RiskManager(self.cfg)
        self.telegram = TelegramPanel(self.cfg, self.logger)

        self.markets: Dict[str, Any] = {}
        self.symbols: List[str] = []
        self.best_signals: List[Signal] = []
        self.last_prices: Dict[str, float] = {}
        self.last_balance: Dict[str, Any] = {}

        self.trading_enabled = True
        self.shutdown = False
        self.current_equity = 0.0
        self.last_error = "nessuno"
        self.scan_count = 0
        self.liquid_count = 0
        self.last_scan_start = "n/d"
        self.last_scan_end = "n/d"
        self.last_market_reload = 0.0
        self.last_signal_report = 0.0

        self.register_commands()

    def register_commands(self):
        self.telegram.register("saldo", lambda _: self.cmd_balance())
        self.telegram.register("status", lambda _: self.cmd_status())
        self.telegram.register("trades", lambda _: self.cmd_trades())
        self.telegram.register("profitto", lambda _: self.cmd_profit())
        self.telegram.register("mercato", lambda _: self.cmd_market())
        self.telegram.register("segnali", lambda _: self.format_signals())
        self.telegram.register("start", lambda _: self.cmd_start())
        self.telegram.register("stop", lambda _: self.cmd_stop())
        self.telegram.register("help", lambda _: self.cmd_help())

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def connect(self):
        self.exchange = ccxt.kraken({
            "apiKey": os.getenv("KRAKEN_API_KEY"),
            "secret": os.getenv("KRAKEN_SECRET"),
            "enableRateLimit": True,
        })

        self.exchange.options["adjustForTimeDifference"] = True

        self.telegram.start()
        self.telegram.send(
            "Bot avviato\n"
            "Exchange: Kraken reale\n"
            f"Timeframe: {self.cfg.TIMEFRAME}\n"
            f"Dry run: {self.cfg.DRY_RUN}"
        )

        self.load_markets()
        self.refresh_balance(send=True)

    def call(self, func: Callable, *args, **kwargs):
        last_error = None

        for attempt in range(1, 5):
            try:
                return func(*args, **kwargs)
            except ccxt.RateLimitExceeded as e:
                last_error = e
                time.sleep(4 * attempt)
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.RequestException) as e:
                last_error = e
                time.sleep(2 * attempt)
            except ccxt.ExchangeError as e:
                last_error = e
                time.sleep(2 * attempt)
            except Exception as e:
                last_error = e
                time.sleep(2 * attempt)

        raise RuntimeError(f"Errore Kraken dopo retry: {last_error}")

    def load_markets(self):
        self.markets = self.call(lambda: self.exchange.load_markets(reload=True))
        self.symbols = []

        for symbol, market in self.markets.items():
            if self.valid_market(symbol, market):
                self.symbols.append(symbol)

        self.symbols.sort()
        self.last_market_reload = time.time()

        self.telegram.send(
            "Connessione Kraken riuscita\n"
            f"Coin monitorate USD/USDT: {len(self.symbols)}"
        )

    def valid_market(self, symbol: str, market: Dict[str, Any]) -> bool:
        try:
            if market.get("active") is False:
                return False

            if market.get("spot") is False:
                return False

            base = str(market.get("base", "")).upper()
            quote = str(market.get("quote", "")).upper()
            raw = (base + symbol).upper().replace("/", "").replace("-", "").replace("_", "")

            if quote not in self.cfg.QUOTES:
                return False

            if base in self.cfg.STABLECOINS or base in self.cfg.FIAT:
                return False

            for word in self.cfg.LEVERAGED_WORDS:
                if raw.endswith(word) or raw.startswith(word):
                    return False

            return True
        except Exception:
            return False

    def refresh_balance(self, send: bool = False):
        try:
            self.last_balance = self.call(self.exchange.fetch_balance)
            self.current_equity = self.estimate_equity(self.last_balance)
            self.risk.update_equity(self.current_equity)

            if send:
                self.telegram.send(self.format_balance())
        except Exception as e:
            self.last_error = str(e)
            self.telegram.send(f"Errore saldo Kraken: {e}")

    def estimate_equity(self, balance: Dict[str, Any]) -> float:
        total = balance.get("total", {}) or {}

        equity = float(total.get("USD", 0) or 0)
        equity += float(total.get("USDT", 0) or 0)

        for asset, amount_raw in total.items():
            try:
                asset = str(asset).upper()
                amount = float(amount_raw or 0)

                if amount <= 0 or asset in ("USD", "USDT"):
                    continue

                symbol = ""
                if f"{asset}/USD" in self.markets:
                    symbol = f"{asset}/USD"
                elif f"{asset}/USDT" in self.markets:
                    symbol = f"{asset}/USDT"

                if not symbol:
                    continue

                ticker = self.call(self.exchange.fetch_ticker, symbol)
                price = float(ticker.get("last") or ticker.get("close") or 0)

                if price > 0:
                    equity += amount * price
                    self.last_prices[symbol] = price

                time.sleep(self.cfg.PER_SYMBOL_DELAY_SECONDS)
            except Exception:
                continue

        return equity

    def scan_market(self):
        self.scan_count += 1
        self.last_scan_start = self.now_iso()

        signals = []
        liquid = 0
        errors = 0

        for symbol in list(self.symbols):
            if self.shutdown:
                break

            try:
                ohlcv = self.call(
                    self.exchange.fetch_ohlcv,
                    symbol,
                    self.cfg.TIMEFRAME,
                    limit=self.cfg.OHLCV_LIMIT,
                )

                sig = self.strategy.analyze(symbol, ohlcv)

                if not sig:
                    time.sleep(self.cfg.PER_SYMBOL_DELAY_SECONDS)
                    continue

                signals.append(sig)
                self.last_prices[symbol] = sig.price

                if sig.metrics.get("quote_volume_24h", 0) >= self.cfg.MIN_24H_QUOTE_VOLUME_USD:
                    liquid += 1

                if sig.buy:
                    self.open_trade(sig)

                time.sleep(self.cfg.PER_SYMBOL_DELAY_SECONDS)

            except Exception as e:
                errors += 1
                self.last_error = str(e)
                self.logger.warning("Errore scan %s: %s", symbol, e)
                time.sleep(self.cfg.PER_SYMBOL_DELAY_SECONDS)

        self.best_signals = sorted(signals, key=lambda x: x.score, reverse=True)[:self.cfg.TOP_SIGNALS_LIMIT]
        self.liquid_count = liquid
        self.last_scan_end = self.now_iso()

        self.logger.info(
            "Scan completato: mercati=%s liquidi=%s errori=%s",
            len(self.symbols),
            liquid,
            errors,
        )

        if time.time() - self.last_signal_report >= self.cfg.TELEGRAM_SIGNAL_INTERVAL_SECONDS:
            self.last_signal_report = time.time()
            self.telegram.send(self.format_signals(), silent=True)

    def manage_positions(self):
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
                    self.cfg.TIMEFRAME,
                    limit=self.cfg.OHLCV_LIMIT,
                )

                exit_data = self.strategy.exit_signal(ohlcv)
                metrics = exit_data.get("metrics", {})

                price = float(metrics.get("price") or self.fetch_price(symbol))
                atr = float(metrics.get("atr") or pos.entry_price * 0.01)

                updated = self.risk.update_trailing(symbol, price, atr)
                if not updated:
                    continue

                self.last_prices[symbol] = price

                reason = ""

                if price <= updated.stop_loss:
                    reason = "stop loss"
                elif price <= updated.trailing_stop:
                    reason = "trailing stop"
                elif price >= updated.take_profit:
                    reason = "take profit"
                elif exit_data.get("exit"):
                    reason = str(exit_data.get("reason") or "uscita strategia")

                if reason:
                    self.close_trade(symbol, reason, price)

                time.sleep(self.cfg.PER_SYMBOL_DELAY_SECONDS)

            except Exception as e:
                self.last_error = str(e)
                self.telegram.send(f"Errore gestione posizione {symbol}: {e}")

    def open_trade(self, sig: Signal):
        try:
            if not self.trading_enabled:
                return

            market = self.markets.get(sig.symbol)
            if not market:
                return

            quote = str(market.get("quote", "")).upper()
            base = str(market.get("base", "")).upper()

            self.refresh_balance(send=False)

            free = self.last_balance.get("free", {}) or {}
            quote_free = float(free.get(quote, 0) or 0)

            allowed, reason = self.risk.can_open(sig.symbol, self.current_equity, quote_free)
            if not allowed:
                self.logger.info("Segnale ignorato %s: %s", sig.symbol, reason)
                return

            capital = self.risk.trade_capital(self.current_equity, quote_free)
            amount = capital / sig.price
            amount = float(self.exchange.amount_to_precision(sig.symbol, amount))

            if amount <= 0:
                return

            if not self.check_market_limits(sig.symbol, amount, capital):
                return

            if self.cfg.DRY_RUN:
                order = {
                    "id": f"dry-buy-{int(time.time())}",
                    "average": sig.price,
                    "filled": amount,
                    "cost": amount * sig.price,
                    "fee": {"cost": 0},
                }
            else:
                order = self.call(self.exchange.create_market_buy_order, sig.symbol, amount)

            entry = float(order.get("average") or order.get("price") or sig.price)
            filled = float(order.get("filled") or amount)
            cost = float(order.get("cost") or filled * entry)
            fees = self.extract_fees(order)

            levels = self.risk.levels(entry, float(sig.metrics.get("atr", entry * 0.01)))

            pos = Position(
                symbol=sig.symbol,
                base=base,
                quote=quote,
                amount=filled,
                entry_price=entry,
                entry_time=self.now_iso(),
                stop_loss=levels["stop_loss"],
                take_profit=levels["take_profit"],
                trailing_stop=levels["trailing_stop"],
                highest_price=entry,
                order_id=str(order.get("id", "")),
                quote_cost=cost,
                fees=fees,
                score=sig.score,
            )

            self.risk.add_position(pos)

            self.telegram.send(
                "TRADE APERTO\n"
                f"{sig.symbol}\n"
                f"Prezzo: {entry:.10g}\n"
                f"Quantita: {filled:.10g}\n"
                f"Capitale: {cost:.2f} {quote}\n"
                f"Stop loss: {pos.stop_loss:.10g}\n"
                f"Trailing stop: {pos.trailing_stop:.10g}\n"
                f"Take profit: {pos.take_profit:.10g}\n"
                f"Score: {sig.score:.2f}\n"
                f"Motivi: {', '.join(sig.reasons)}"
            )

        except Exception as e:
            self.last_error = str(e)
            self.telegram.send(f"Errore apertura trade {sig.symbol}: {e}")

    def close_trade(self, symbol: str, reason: str, fallback_price: float):
        try:
            pos = self.risk.positions.get(symbol)
            if not pos:
                return

            self.refresh_balance(send=False)

            free = self.last_balance.get("free", {}) or {}
            available = float(free.get(pos.base, pos.amount) or 0)
            amount = min(pos.amount, available if available > 0 else pos.amount)
            amount = float(self.exchange.amount_to_precision(symbol, amount))

            if amount <= 0:
                self.telegram.send(f"Impossibile chiudere {symbol}: saldo non disponibile")
                return

            if self.cfg.DRY_RUN:
                order = {
                    "id": f"dry-sell-{int(time.time())}",
                    "average": fallback_price,
                    "filled": amount,
                    "cost": amount * fallback_price,
                    "fee": {"cost": 0},
                }
            else:
                order = self.call(self.exchange.create_market_sell_order, symbol, amount)

            exit_price = float(order.get("average") or order.get("price") or fallback_price)
            fees = self.extract_fees(order)

            closed = self.risk.close(
                symbol=symbol,
                exit_price=exit_price,
                reason=reason,
                fees=fees,
                order_id=str(order.get("id", "")),
            )

            if not closed:
                return

            self.telegram.send(
                "TRADE CHIUSO\n"
                f"{symbol}\n"
                f"Motivo: {reason}\n"
                f"Entry: {closed['entry_price']:.10g}\n"
                f"Exit: {closed['exit_price']:.10g}\n"
                f"PnL netto: {closed['net_pnl']:.2f} {closed['quote']}\n"
                f"PnL %: {closed['pnl_percent']:.2f}%\n"
                f"Perdite consecutive: {self.risk.consecutive_losses}"
            )

        except Exception as e:
            self.last_error = str(e)
            self.telegram.send(f"Errore chiusura trade {symbol}: {e}")

    def fetch_price(self, symbol: str) -> float:
        try:
            ticker = self.call(self.exchange.fetch_ticker, symbol)
            price = float(ticker.get("last") or ticker.get("close") or 0)
            if price > 0:
                self.last_prices[symbol] = price
            return price
        except Exception:
            return self.last_prices.get(symbol, 0)

    def check_market_limits(self, symbol: str, amount: float, cost: float) -> bool:
        try:
            limits = self.markets.get(symbol, {}).get("limits", {}) or {}
            amount_min = (limits.get("amount", {}) or {}).get("min")
            cost_min = (limits.get("cost", {}) or {}).get("min")

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

            fee = order.get("fee")
            if isinstance(fee, dict):
                total += float(fee.get("cost") or 0)

            for item in order.get("fees") or []:
                if isinstance(item, dict):
                    total += float(item.get("cost") or 0)

            return total
        except Exception:
            return 0.0

    def format_balance(self) -> str:
        free = self.last_balance.get("free", {}) or {}
        total = self.last_balance.get("total", {}) or {}

        return (
            "SALDO ACCOUNT\n"
            f"Equity stimata: {self.current_equity:.2f} USD\n"
            f"USD free: {float(free.get('USD', 0) or 0):.2f}\n"
            f"USDT free: {float(free.get('USDT', 0) or 0):.2f}\n"
            f"USD totale: {float(total.get('USD', 0) or 0):.2f}\n"
            f"USDT totale: {float(total.get('USDT', 0) or 0):.2f}\n"
            f"PnL giornaliero: {self.risk.daily_realized_pnl:.2f} USD\n"
            f"Drawdown: {self.risk.current_drawdown * 100:.2f}%"
        )

    def format_signals(self) -> str:
        if not self.best_signals:
            return "SEGNALI\nNessun segnale disponibile."

        lines = ["MIGLIORI SEGNALI"]

        for i, sig in enumerate(self.best_signals, 1):
            m = sig.metrics
            lines.append(
                f"{i}. {sig.symbol} | score {sig.score:.2f} | "
                f"prezzo {sig.price:.10g} | RSI {m.get('rsi', 0):.1f} | "
                f"vol x{m.get('volume_ratio', 0):.2f} | "
                f"mom {m.get('momentum', 0) * 100:.2f}% | "
                f"BUY {'SI' if sig.buy else 'NO'}"
            )

        return "\n".join(lines)

    def cmd_balance(self) -> str:
        self.refresh_balance(send=False)
        return self.format_balance()

    def cmd_status(self) -> str:
        return (
            "STATUS BOT\n"
            f"Trading attivo: {self.trading_enabled}\n"
            f"Dry run: {self.cfg.DRY_RUN}\n"
            f"Mercati monitorati: {len(self.symbols)}\n"
            f"Coin liquide ultimo scan: {self.liquid_count}\n"
            f"Trade aperti: {len(self.risk.positions)}/{self.cfg.MAX_OPEN_TRADES}\n"
            f"Equity: {self.current_equity:.2f} USD\n"
            f"PnL giornaliero: {self.risk.daily_realized_pnl:.2f} USD\n"
            f"Drawdown: {self.risk.current_drawdown * 100:.2f}%\n"
            f"Perdite consecutive: {self.risk.consecutive_losses}\n"
            f"Pausa rischio: {self.risk.pause_minutes()} min\n"
            f"Scan completati: {self.scan_count}\n"
            f"Ultimo scan start: {self.last_scan_start}\n"
            f"Ultimo scan fine: {self.last_scan_end}\n"
            f"Ultimo errore: {self.last_error}"
        )

    def cmd_trades(self) -> str:
        lines = ["TRADES APERTI"]

        if not self.risk.positions:
            lines.append("Nessun trade aperto.")

        for pos in self.risk.positions.values():
            price = self.last_prices.get(pos.symbol, pos.entry_price)
            pnl = (price - pos.entry_price) * pos.amount

            lines.append(
                f"{pos.symbol} | qty {pos.amount:.10g} | "
                f"entry {pos.entry_price:.10g} | last {price:.10g} | "
                f"PnL {pnl:.2f} {pos.quote} | "
                f"SL {pos.stop_loss:.10g} | TS {pos.trailing_stop:.10g} | TP {pos.take_profit:.10g}"
            )

        lines.append("")
        lines.append("ULTIMI TRADE CHIUSI")

        recent = self.risk.closed_trades[-5:]

        if not recent:
            lines.append("Nessun trade chiuso.")

        for trade in reversed(recent):
            lines.append(
                f"{trade['symbol']} | PnL {float(trade['net_pnl']):.2f} "
                f"{trade['quote']} | {float(trade['pnl_percent']):.2f}% | {trade['reason']}"
            )

        return "\n".join(lines)

    def cmd_profit(self) -> str:
        unrealized = self.risk.unrealized_pnl(self.last_prices)

        return (
            "PROFITTO\n"
            f"PnL giornaliero realizzato: {self.risk.daily_realized_pnl:.2f} USD\n"
            f"PnL aperto stimato: {unrealized:.2f} USD\n"
            f"PnL totale chiuso: {self.risk.total_closed_pnl():.2f} USD\n"
            f"Drawdown: {self.risk.current_drawdown * 100:.2f}%\n"
            f"Trade chiusi totali: {len(self.risk.closed_trades)}"
        )

    def cmd_market(self) -> str:
        return (
            "MERCATO\n"
            "Exchange: Kraken\n"
            f"Coppie USD/USDT filtrate: {len(self.symbols)}\n"
            f"Coin liquide ultimo scan: {self.liquid_count}\n"
            f"Timeframe: {self.cfg.TIMEFRAME}\n"
            f"Volume minimo 24h: {self.cfg.MIN_24H_QUOTE_VOLUME_USD:.0f} USD\n"
            f"Scan completati: {self.scan_count}"
        )

    def cmd_start(self) -> str:
        self.trading_enabled = True
        return "Trading riattivato."

    def cmd_stop(self) -> str:
        self.trading_enabled = False
        return "Trading sospeso. Le posizioni aperte restano gestite."

    def cmd_help(self) -> str:
        return (
            "COMANDI\n"
            "/saldo\n"
            "/status\n"
            "/trades\n"
            "/profitto\n"
            "/mercato\n"
            "/segnali\n"
            "/start\n"
            "/stop\n"
            "/help"
        )

    def request_shutdown(self, signum, frame):
        self.shutdown = True

    def run(self):
        signal.signal(signal.SIGTERM, self.request_shutdown)
        signal.signal(signal.SIGINT, self.request_shutdown)

        while not self.shutdown:
            try:
                self.connect()

                while not self.shutdown:
                    if time.time() - self.last_market_reload > 3600:
                        self.load_markets()

                    self.refresh_balance(send=False)
                    self.manage_positions()

                    if self.trading_enabled and not self.risk.daily_stop_hit():
                        self.scan_market()
                    else:
                        self.logger.info("Trading in pausa")

                    time.sleep(self.cfg.SCAN_INTERVAL_SECONDS)

            except Exception as e:
                self.last_error = str(e)
                self.logger.exception("Errore loop principale: %s", e)

                try:
                    self.telegram.send(f"Errore loop principale: {e}")
                except Exception:
                    pass

                time.sleep(self.cfg.SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    KrakenTradingBot().run()
