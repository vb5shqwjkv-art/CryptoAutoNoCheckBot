import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
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


QUOTE_CURRENCIES = {"USD", "USDT"}
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
    return logging.getLogger("KrakenRailwayBot")


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


@dataclass
class Config:
    kraken_api_key: str = os.getenv("KRAKEN_API_KEY", "").strip()
    kraken_secret: str = os.getenv("KRAKEN_SECRET", "").strip()
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "").strip()
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    timeframe: str = os.getenv("TIMEFRAME", "15m").strip()
    ohlcv_limit: int = env_int("OHLCV_LIMIT", 150)
    scan_interval_seconds: int = env_int("SCAN_INTERVAL_SECONDS", 60)
    per_symbol_delay_seconds: float = env_float("PER_SYMBOL_DELAY_SECONDS", 1.2)
    market_refresh_seconds: int = env_int("MARKET_REFRESH_SECONDS", 3600)
    retry_attempts: int = env_int("RETRY_ATTEMPTS", 4)
    retry_sleep_seconds: float = env_float("RETRY_SLEEP_SECONDS", 2.0)

    min_24h_quote_volume_usd: float = env_float("MIN_24H_QUOTE_VOLUME_USD", 250000.0)
    max_open_trades: int = min(env_int("MAX_OPEN_TRADES", 3), 3)
    min_trade_amount: float = env_float("MIN_TRADE_AMOUNT", 0.45)
    max_trade_amount: float = env_float("MAX_TRADE_AMOUNT", 1.35)
    daily_max_loss: float = min(env_float("DAILY_MAX_LOSS", 0.05), 0.05)

    ema_fast: int = env_int("EMA_FAST", 20)
    ema_slow: int = env_int("EMA_SLOW", 50)
    rsi_period: int = env_int("RSI_PERIOD", 14)
    rsi_buy_min: float = env_float("RSI_BUY_MIN", 50.0)
    rsi_buy_max: float = env_float("RSI_BUY_MAX", 70.0)
    rsi_exit: float = env_float("RSI_EXIT", 78.0)
    atr_period: int = env_int("ATR_PERIOD", 14)
    volume_window: int = env_int("VOLUME_WINDOW", 20)
    volume_breakout_multiplier: float = env_float("VOLUME_BREAKOUT_MULTIPLIER", 1.35)
    breakout_lookback: int = env_int("BREAKOUT_LOOKBACK", 20)
    momentum_lookback: int = env_int("MOMENTUM_LOOKBACK", 5)

    min_atr_percent: float = env_float("MIN_ATR_PERCENT", 0.002)
    max_atr_percent: float = env_float("MAX_ATR_PERCENT", 0.18)
    stop_loss_atr_multiplier: float = env_float("STOP_LOSS_ATR_MULTIPLIER", 2.0)
    take_profit_atr_multiplier: float = env_float("TAKE_PROFIT_ATR_MULTIPLIER", 3.0)
    trailing_atr_multiplier: float = env_float("TRAILING_ATR_MULTIPLIER", 2.0)
    max_stop_loss_percent: float = env_float("MAX_STOP_LOSS_PERCENT", 0.04)
    min_take_profit_percent: float = env_float("MIN_TAKE_PROFIT_PERCENT", 0.025)

    max_consecutive_losses: int = env_int("MAX_CONSECUTIVE_LOSSES", 3)
    loss_cooldown_seconds: int = env_int("LOSS_COOLDOWN_SECONDS", 10800)
    min_seconds_between_trades: int = env_int("MIN_SECONDS_BETWEEN_TRADES", 600)
    symbol_cooldown_seconds: int = env_int("SYMBOL_COOLDOWN_SECONDS", 3600)
    telegram_signal_interval_seconds: int = env_int("TELEGRAM_SIGNAL_INTERVAL_SECONDS", 1800)
    top_signals_limit: int = env_int("TOP_SIGNALS_LIMIT", 10)

    dry_run: bool = env_bool("DRY_RUN", False)
    state_file: str = os.getenv("STATE_FILE", "bot_state.json").strip()

    def telegram_enabled(self) -> bool:
        return bool(
            self.telegram_token
            and self.telegram_chat_id
            and self.telegram_token.upper() != "DISABLED"
            and self.telegram_chat_id.upper() != "DISABLED"
        )


class TelegramPanel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = cfg.telegram_enabled()
        self.base_url = f"https://api.telegram.org/bot{cfg.telegram_token}"
        self.chat_id = str(cfg.telegram_chat_id)
        self.session = requests.Session()
        self.handlers: Dict[str, Callable[[str], str]] = {}
        self.offset: Optional[int] = None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def register(self, command: str, handler: Callable[[str], str]) -> None:
        self.handlers[command.strip().lower().replace("/", "")] = handler

    def send(self, text: str, silent: bool = False) -> bool:
        if not self.enabled:
            LOGGER.info("Telegram disattivato: %s", text.replace("\n", " | ")[:300])
            return False

        try:
            ok = True
            for start in range(0, len(text), 3900):
                response = self.session.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text[start:start + 3900],
                        "disable_notification": silent,
                    },
                    timeout=20,
                )
                if response.status_code >= 400:
                    ok = False
                    LOGGER.error("Errore Telegram sendMessage %s: %s", response.status_code, response.text[:500])
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

                response = self.session.get(f"{self.base_url}/getUpdates", params=params, timeout=35)

                if response.status_code >= 400:
                    LOGGER.error("Telegram getUpdates %s: %s", response.status_code, response.text[:500])
                    time.sleep(5)
                    continue

                data = response.json()

                for update in data.get("result", []):
                    self.offset = int(update.get("update_id", 0)) + 1
                    self.handle_update(update)

            except Exception as exc:
                LOGGER.warning("Errore polling Telegram: %s", exc)
                time.sleep(5)

    def handle_update(self, update: Dict[str, Any]) -> None:
        try:
            message = update.get("message") or {}
            chat = message.get("chat") or {}
            chat_id = str(chat.get("id", ""))
            text = str(message.get("text", "")).strip()

            if chat_id != self.chat_id or not text.startswith("/"):
                return

            command = text.split()[0].split("@")[0].replace("/", "").lower()
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
            needed = max(self.cfg.ema_slow, self.cfg.breakout_lookback, self.cfg.volume_window, 80)
            if len(df) < needed:
                return pd.DataFrame()

            df = df.copy()
            close = df["close"]

            df["ema20"] = EMAIndicator(close=close, window=self.cfg.ema_fast).ema_indicator()
            df["ema50"] = EMAIndicator(close=close, window=self.cfg.ema_slow).ema_indicator()
            df["rsi"] = RSIIndicator(close=close, window=self.cfg.rsi_period).rsi()
            df["atr"] = AverageTrueRange(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=self.cfg.atr_period,
            ).average_true_range()

            df["volume_avg"] = df["volume"].rolling(self.cfg.volume_window).mean()
            df["breakout_high"] = df["high"].shift(1).rolling(self.cfg.breakout_lookback).max()
            df["momentum"] = df["close"].pct_change(self.cfg.momentum_lookback)
            df["atr_percent"] = df["atr"] / df["close"]
            df["quote_volume"] = df["close"] * df["volume"]

            return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def analyze(self, symbol: str, ohlcv: List[List[float]]) -> Optional[Signal]:
        try:
            df = self.indicators(self.dataframe(ohlcv))
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

            liquid = quote_volume_24h >= self.cfg.min_24h_quote_volume_usd
            trend_up = ema20 > ema50
            rsi_ok = self.cfg.rsi_buy_min <= rsi <= self.cfg.rsi_buy_max
            volume_breakout = volume > volume_avg * self.cfg.volume_breakout_multiplier
            breakout = price > breakout_high
            momentum_positive = momentum > 0
            volatility_ok = self.cfg.min_atr_percent <= atr_percent <= self.cfg.max_atr_percent

            score = 0.0
            reasons: List[str] = []

            if liquid:
                score += 15
                reasons.append("liquido")

            if trend_up:
                ema_strength = min(max((ema20 - ema50) / max(price, 1e-12), 0.0) * 100.0, 8.0)
                score += 20 + ema_strength
                reasons.append("EMA20 sopra EMA50")

            if rsi_ok:
                score += 15
                reasons.append("RSI valido")

            if volume_breakout:
                volume_strength = min(
                    max(volume / max(volume_avg, 1e-12) - self.cfg.volume_breakout_multiplier, 0.0) * 8.0,
                    12.0,
                )
                score += 20 + volume_strength
                reasons.append("volume breakout")

            if breakout:
                breakout_strength = min(max(price / max(breakout_high, 1e-12) - 1.0, 0.0) * 100.0, 8.0)
                score += 20 + breakout_strength
                reasons.append("breakout rialzista")

            if momentum_positive:
                momentum_strength = min(max(momentum, 0.0) * 100.0, 10.0)
                score += 10 + momentum_strength
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
            df = self.indicators(self.dataframe(ohlcv))
            if df.empty:
                return {"exit": False, "reason": "", "metrics": {}}

            row = df.iloc[-1]

            price = float(row["close"])
            ema20 = float(row["ema20"])
            ema50 = float(row["ema50"])
            rsi = float(row["rsi"])
            momentum = float(row["momentum"])
            atr = float(row["atr"])

            metrics = {"price": price, "rsi": rsi, "momentum": momentum, "atr": atr}

            if ema20 < ema50 and momentum < 0:
                return {"exit": True, "reason": "inversione trend", "metrics": metrics}

            if rsi >= self.cfg.rsi_exit and momentum <= 0:
                return {"exit": True, "reason": "RSI troppo alto", "metrics": metrics}

            return {"exit": False, "reason": "", "metrics": metrics}
        except Exception:
            return {"exit": False, "reason": "", "metrics": {}}


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

    def load(self) -> None:
        try:
            path = Path(self.cfg.state_file)
            if not path.exists():
                return

            data = json.loads(path.read_text(encoding="utf-8"))

            self.positions = {
                symbol: Position(**raw)
                for symbol, raw in data.get("positions", {}).items()
            }

            self.closed_trades = list(data.get("closed_trades", []))[-500:]
            self.current_day = str(data.get("current_day", self.today()))
            self.daily_start_equity = float(data.get("daily_start_equity", 0.0))
            self.daily_realized_pnl = float(data.get("daily_realized_pnl", 0.0))
            self.current_drawdown = float(data.get("current_drawdown", 0.0))
            self.consecutive_losses = int(data.get("consecutive_losses", 0))
            self.pause_until = float(data.get("pause_until", 0.0))
            self.last_trade_at = float(data.get("last_trade_at", 0.0))
            self.symbol_last_trade_at = {
                str(k): float(v)
                for k, v in data.get("symbol_last_trade_at", {}).items()
            }
        except Exception:
            pass

    def save(self) -> None:
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
                "current_drawdown": self.current_drawdown,
                "consecutive_losses": self.consecutive_losses,
                "pause_until": self.pause_until,
                "last_trade_at": self.last_trade_at,
                "symbol_last_trade_at": self.symbol_last_trade_at,
            }

            Path(self.cfg.state_file).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def update_equity(self, equity: float) -> None:
        try:
            today = self.today()

            if today != self.current_day:
                self.current_day = today
                self.daily_start_equity = max(0.0, equity)
                self.daily_realized_pnl = 0.0
                self.current_drawdown = 0.0
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

        realized_stop = self.daily_realized_pnl <= -self.daily_start_equity * self.cfg.daily_max_loss
        drawdown_stop = self.current_drawdown >= self.cfg.daily_max_loss

        return realized_stop or drawdown_stop

    def pause_active(self) -> bool:
        return time.time() < self.pause_until

    def pause_minutes(self) -> int:
        return int(max(0.0, self.pause_until - time.time()) // 60)

    def can_open(self, symbol: str, quote_free: float) -> Tuple[bool, str]:
        now = time.time()

        if symbol in self.positions:
            return False, "posizione gia aperta"

        if len(self.positions) >= self.cfg.max_open_trades:
            return False, "limite massimo trade aperti"

        if quote_free < self.cfg.min_trade_amount:
            return False, "saldo troppo basso per minimo trade"

        if self.daily_stop_hit():
            return False, "stop giornaliero perdita attivo"

        if self.pause_active():
            return False, f"pausa rischio attiva {self.pause_minutes()} min"

        if now - self.last_trade_at < self.cfg.min_seconds_between_trades:
            return False, "anti-overtrading attivo"

        if now - self.symbol_last_trade_at.get(symbol, 0.0) < self.cfg.symbol_cooldown_seconds:
            return False, "cooldown simbolo attivo"

        return True, "ok"

    def trade_capital(self, quote_free: float, signal_data: Optional[Signal]) -> float:
        available = max(0.0, quote_free * 0.95)

        if available < self.cfg.min_trade_amount:
            return 0.0

        max_amount = max(self.cfg.min_trade_amount, self.cfg.max_trade_amount)
        multiplier = 1.0

        if signal_data is not None:
            metrics = signal_data.metrics
            volume_ratio = float(metrics.get("volume_ratio", 1.0) or 1.0)
            momentum = float(metrics.get("momentum", 0.0) or 0.0)
            rsi = float(metrics.get("rsi", 50.0) or 50.0)
            atr_percent = float(metrics.get("atr_percent", 0.0) or 0.0)

            strength = 0

            if signal_data.score >= 105:
                strength += 1
            if signal_data.score >= 115:
                strength += 1
            if volume_ratio >= 2.0:
                strength += 1
            if momentum >= 0.01:
                strength += 1
            if 55 <= rsi <= 65:
                strength += 1
            if 0.004 <= atr_percent <= 0.08:
                strength += 1

            multiplier = 1.0 + min(strength, 4) * 0.5

        desired = self.cfg.min_trade_amount * multiplier

        return max(0.0, min(desired, max_amount, available))

    def levels(self, entry: float, atr: float) -> Dict[str, float]:
        atr = max(float(atr), entry * self.cfg.min_atr_percent)

        stop_atr = entry - atr * self.cfg.stop_loss_atr_multiplier
        stop_percent = entry * (1.0 - self.cfg.max_stop_loss_percent)
        stop_loss = max(stop_atr, stop_percent)

        take_atr = entry + atr * self.cfg.take_profit_atr_multiplier
        take_percent = entry * (1.0 + self.cfg.min_take_profit_percent)
        take_profit = max(take_atr, take_percent)

        trailing_stop = max(stop_loss, entry - atr * self.cfg.trailing_atr_multiplier)

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop": trailing_stop,
        }

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

        atr = max(atr, pos.entry_price * self.cfg.min_atr_percent)
        candidate = pos.highest_price - atr * self.cfg.trailing_atr_multiplier

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

        gross = (exit_price - pos.entry_price) * pos.amount
        net = gross - pos.fees - fees
        pnl_percent = net / max(pos.entry_price * pos.amount, 1e-12) * 100.0

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
        total = 0.0

        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.entry_price)
            total += (price - pos.entry_price) * pos.amount

        return total

    def total_closed_pnl(self) -> float:
        return sum(float(t.get("net_pnl", 0.0)) for t in self.closed_trades)


class KrakenTradingBot:
    def __init__(self):
        self.cfg = Config()
        self.telegram = TelegramPanel(self.cfg)
        self.exchange: Optional[Any] = None
        self.strategy = Strategy(self.cfg)
        self.risk = RiskManager(self.cfg)

        self.markets: Dict[str, Any] = {}
        self.symbols: List[str] = []
        self.best_signals: List[Signal] = []
        self.last_prices: Dict[str, float] = {}
        self.last_balance: Dict[str, Any] = {}

        self.current_equity = 0.0
        self.trading_enabled = True
        self.shutdown = False
        self.last_error = "nessuno"

        self.scan_count = 0
        self.liquid_count = 0
        self.last_scan_start = "n/d"
        self.last_scan_end = "n/d"
        self.last_market_reload = 0.0
        self.last_signal_report = 0.0

        self.register_commands()

    def register_commands(self) -> None:
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
        last_error: Optional[Exception] = None

        for attempt in range(1, self.cfg.retry_attempts + 1):
            try:
                return func(*args, **kwargs)

            except ccxt.RateLimitExceeded as exc:
                last_error = exc
                LOGGER.warning("Rate limit Kraken. Retry %s", attempt)
                time.sleep(self.cfg.retry_sleep_seconds * attempt * 2)

            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.RequestException) as exc:
                last_error = exc
                LOGGER.warning("Errore rete Kraken. Retry %s: %s", attempt, exc)
                time.sleep(self.cfg.retry_sleep_seconds * attempt)

            except ccxt.ExchangeError as exc:
                last_error = exc
                LOGGER.warning("Errore exchange Kraken. Retry %s: %s", attempt, exc)
                time.sleep(self.cfg.retry_sleep_seconds * attempt)

            except Exception as exc:
                last_error = exc
                LOGGER.warning("Errore chiamata Kraken. Retry %s: %s", attempt, exc)
                time.sleep(self.cfg.retry_sleep_seconds * attempt)

        raise RuntimeError(f"Errore Kraken dopo retry: {last_error}")

    def connect(self) -> None:
        LOGGER.info("Inizializzazione Kraken")
        self.init_exchange()

        self.telegram.start_polling()

        self.telegram.send(
            "Bot avviato\n"
            "Exchange: Kraken reale\n"
            f"Timeframe: {self.cfg.timeframe}\n"
            f"Dry run: {self.cfg.dry_run}\n"
            f"Min trade: {self.cfg.min_trade_amount:.2f}\n"
            f"Max trade: {self.cfg.max_trade_amount:.2f}"
        )

        self.load_markets()
        self.refresh_balance(send=True)

    def valid_market(self, symbol: str, market: Dict[str, Any]) -> bool:
        try:
            if market.get("active") is False:
                return False

            if market.get("spot") is False:
                return False

            base = str(market.get("base", "")).upper()
            quote = str(market.get("quote", "")).upper()
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

        self.symbols = [
            symbol
            for symbol, market in self.markets.items()
            if self.valid_market(symbol, market)
        ]

        self.symbols.sort()
        self.last_market_reload = time.time()

        LOGGER.info("Mercati monitorati USD/USDT: %s", len(self.symbols))

        self.telegram.send(
            "Connessione Kraken riuscita\n"
            f"Coin monitorate USD/USDT: {len(self.symbols)}"
        )

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
        total = balance.get("total", {}) or {}

        equity = float(total.get("USD", 0.0) or 0.0)
        equity += float(total.get("USDT", 0.0) or 0.0)

        for asset, amount_raw in total.items():
            try:
                asset = str(asset).upper()
                amount = float(amount_raw or 0.0)

                if amount <= 0 or asset in {"USD", "USDT"}:
                    continue

                symbol = ""

                if f"{asset}/USD" in self.markets:
                    symbol = f"{asset}/USD"
                elif f"{asset}/USDT" in self.markets:
                    symbol = f"{asset}/USDT"

                if not symbol:
                    continue

                ticker = self.call(self.exchange.fetch_ticker, symbol)
                price = float(ticker.get("last") or ticker.get("close") or 0.0)

                if price > 0:
                    equity += amount * price
                    self.last_prices[symbol] = price

                time.sleep(self.cfg.per_symbol_delay_seconds)

            except Exception:
                continue

        return equity

    def scan_market(self) -> None:
        self.scan_count += 1
        self.last_scan_start = self.now_iso()

        signals: List[Signal] = []
        liquid = 0
        errors = 0

        LOGGER.info("Inizio scan mercato: %s simboli", len(self.symbols))

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

                if sig.metrics.get("quote_volume_24h", 0.0) >= self.cfg.min_24h_quote_volume_usd:
                    liquid += 1

                if sig.buy:
                    self.open_trade(sig)

                time.sleep(self.cfg.per_symbol_delay_seconds)

            except Exception as exc:
                errors += 1
                self.last_error = str(exc)
                LOGGER.warning("Errore scan %s: %s", symbol, exc)
                time.sleep(self.cfg.per_symbol_delay_seconds)

        self.best_signals = sorted(
            signals,
            key=lambda item: item.score,
            reverse=True,
        )[:self.cfg.top_signals_limit]

        self.liquid_count = liquid
        self.last_scan_end = self.now_iso()

        LOGGER.info(
            "Scan completato: mercati=%s liquidi=%s errori=%s",
            len(self.symbols),
            liquid,
            errors,
        )

        if time.time() - self.last_signal_report >= self.cfg.telegram_signal_interval_seconds:
            self.last_signal_report = time.time()
            self.telegram.send(self.format_signals(), silent=True)

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

                time.sleep(self.cfg.per_symbol_delay_seconds)

            except Exception as exc:
                self.last_error = str(exc)
                LOGGER.exception("Errore posizione %s: %s", symbol, exc)
                self.telegram.send(f"Errore gestione posizione {symbol}: {exc}")

    def open_trade(self, sig: Signal) -> None:
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
            quote_free = float(free.get(quote, 0.0) or 0.0)

            allowed, reason = self.risk.can_open(sig.symbol, quote_free)

            if not allowed:
                LOGGER.info("Segnale ignorato %s: %s", sig.symbol, reason)
                return

            capital = self.risk.trade_capital(quote_free, sig)
            capital = self.adjust_capital_for_market_limits(sig.symbol, capital, quote_free)

            if capital <= 0:
                LOGGER.info("Segnale ignorato %s: capitale sotto minimo Kraken", sig.symbol)
                return

            amount = capital / sig.price
            amount = float(self.exchange.amount_to_precision(sig.symbol, amount))

            if amount <= 0:
                return

            if not self.check_market_limits(sig.symbol, amount, capital):
                return

            if self.cfg.dry_run:
                order = {
                    "id": f"dry-buy-{int(time.time())}",
                    "average": sig.price,
                    "filled": amount,
                    "cost": amount * sig.price,
                    "fee": {"cost": 0.0},
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

        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.exception("Errore apertura trade %s: %s", sig.symbol, exc)
            self.telegram.send(f"Errore apertura trade {sig.symbol}: {exc}")

    def close_trade(self, symbol: str, reason: str, fallback_price: float) -> None:
        try:
            pos = self.risk.positions.get(symbol)

            if not pos:
                return

            self.refresh_balance(send=False)

            free = self.last_balance.get("free", {}) or {}
            available = float(free.get(pos.base, pos.amount) or 0.0)
            amount = min(pos.amount, available if available > 0 else pos.amount)
            amount = float(self.exchange.amount_to_precision(symbol, amount))

            if amount <= 0:
                self.telegram.send(f"Impossibile chiudere {symbol}: saldo non disponibile")
                return

            if self.cfg.dry_run:
                order = {
                    "id": f"dry-sell-{int(time.time())}",
                    "average": fallback_price,
                    "filled": amount,
                    "cost": amount * fallback_price,
                    "fee": {"cost": 0.0},
                }
            else:
                order = self.call(self.exchange.create_market_sell_order, symbol, amount)

            exit_price = float(order.get("average") or order.get("price") or fallback_price)
            fees = self.extract_fees(order)

            closed = self.risk.close(symbol, exit_price, reason, fees, str(order.get("id", "")))

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

        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.exception("Errore chiusura trade %s: %s", symbol, exc)
            self.telegram.send(f"Errore chiusura trade {symbol}: {exc}")

    def adjust_capital_for_market_limits(self, symbol: str, capital: float, quote_free: float) -> float:
        try:
            if capital <= 0:
                return 0.0

            limits = self.markets.get(symbol, {}).get("limits", {}) or {}
            cost_min = (limits.get("cost", {}) or {}).get("min")
            available = max(0.0, quote_free * 0.95)
            max_amount = max(self.cfg.min_trade_amount, self.cfg.max_trade_amount)

            if cost_min is not None:
                minimum = float(cost_min) * 1.01

                if capital < minimum:
                    if minimum <= available and minimum <= max_amount:
                        return minimum
                    return 0.0

            return min(capital, available, max_amount)
        except Exception:
            return capital

    def fetch_price(self, symbol: str) -> float:
        try:
            ticker = self.call(self.exchange.fetch_ticker, symbol)
            price = float(ticker.get("last") or ticker.get("close") or 0.0)

            if price > 0:
                self.last_prices[symbol] = price

            return price
        except Exception:
            return self.last_prices.get(symbol, 0.0)

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
                total += float(fee.get("cost") or 0.0)

            for item in order.get("fees") or []:
                if isinstance(item, dict):
                    total += float(item.get("cost") or 0.0)

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

        for index, sig in enumerate(self.best_signals, 1):
            m = sig.metrics

            lines.append(
                f"{index}. {sig.symbol} | score {sig.score:.2f} | "
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
            f"Dry run: {self.cfg.dry_run}\n"
            f"Min trade: {self.cfg.min_trade_amount:.2f}\n"
            f"Max trade: {self.cfg.max_trade_amount:.2f}\n"
            f"Mercati monitorati: {len(self.symbols)}\n"
            f"Coin liquide ultimo scan: {self.liquid_count}\n"
            f"Trade aperti: {len(self.risk.positions)}/{self.cfg.max_open_trades}\n"
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
            f"Timeframe: {self.cfg.timeframe}\n"
            f"Volume minimo 24h: {self.cfg.min_24h_quote_volume_usd:.0f} USD\n"
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

    def request_shutdown(self, signum: int, frame: Any) -> None:
        self.shutdown = True

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.request_shutdown)
        signal.signal(signal.SIGINT, self.request_shutdown)

        LOGGER.info("Bot process avviato")

        while not self.shutdown:
            try:
                if not self.kraken_env_ok():
                    time.sleep(60)
                    continue

                self.connect()

                while not self.shutdown:
                    if time.time() - self.last_market_reload > self.cfg.market_refresh_seconds:
                        self.load_markets()

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

        self.telegram.send("Bot arrestato")


if __name__ == "__main__":
    print("Avvio container Railway...", flush=True)
    start_health_server()
    KrakenTradingBot().run()
