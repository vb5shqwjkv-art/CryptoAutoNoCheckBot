import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import ccxt
import requests

from config import Config
from risk import Position, RiskManager
from strategy import MarketSignal, Strategy
from telegram_bot import TelegramBot


class KrakenTradingBot:
    def __init__(self) -> None:
        self.config = Config.from_env()
        self.config.validate()
        self.logger = self._build_logger()
        self.exchange: Optional[ccxt.kraken] = None
        self.strategy = Strategy(self.config)
        self.risk = RiskManager(self.config)
        self.telegram = TelegramBot(self.config, self.logger)
        self.markets: Dict[str, Any] = {}
        self.symbols: List[str] = []
        self.trading_enabled = True
        self.shutdown_requested = False
        self.current_equity_usd = 0.0
        self.last_balance: Dict[str, Any] = {}
        self.last_prices: Dict[str, float] = {}
        self.best_signals: List[MarketSignal] = []
        self.last_scan_started_at = ""
        self.last_scan_finished_at = ""
        self.last_error = ""
        self.last_market_refresh_ts = 0.0
        self.last_signal_report_ts = 0.0
        self.scan_count = 0
        self.liquid_symbols_last_scan = 0
        self.register_telegram_commands()

    def _build_logger(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            stream=sys.stdout,
        )
        return logging.getLogger("kraken-bot")

    def register_telegram_commands(self) -> None:
        self.telegram.register_command("saldo", self.command_balance)
        self.telegram.register_command("status", self.command_status)
        self.telegram.register_command("trades", self.command_trades)
        self.telegram.register_command("profitto", self.command_profit)
        self.telegram.register_command("mercato", self.command_market)
        self.telegram.register_command("segnali", self.command_signals)
        self.telegram.register_command("start", self.command_start)
        self.telegram.register_command("stop", self.command_stop)
        self.telegram.register_command("help", self.command_help)

    def init_exchange(self) -> None:
        exchange = ccxt.kraken(
            {
                "apiKey": os.getenv("KRAKEN_API_KEY"),
                "secret": os.getenv("KRAKEN_SECRET"),
                "enableRateLimit": True,
            }
        )
        exchange.options["adjustForTimeDifference"] = True
        self.exchange = exchange

    def call_exchange(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                return func(*args, **kwargs)
            except ccxt.RateLimitExceeded as exc:
                last_exc = exc
                sleep_for = self.config.retry_base_sleep_seconds * attempt * 2
                self.logger.warning("Rate limit Kraken, retry in %.1fs", sleep_for)
                time.sleep(sleep_for)
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.RequestException) as exc:
                last_exc = exc
                sleep_for = self.config.retry_base_sleep_seconds * attempt
                self.logger.warning("Errore rete Kraken, retry in %.1fs: %s", sleep_for, exc)
                time.sleep(sleep_for)
            except ccxt.ExchangeError as exc:
                last_exc = exc
                self.logger.warning("Errore exchange Kraken: %s", exc)
                time.sleep(self.config.retry_base_sleep_seconds * attempt)
            except Exception as exc:
                last_exc = exc
                self.logger.exception("Errore chiamata Kraken: %s", exc)
                time.sleep(self.config.retry_base_sleep_seconds * attempt)
        raise RuntimeError(f"Kraken non disponibile dopo retry: {last_exc}")

    def load_markets(self) -> None:
        try:
            if self.exchange is None:
                raise RuntimeError("Exchange non inizializzato")
            self.logger.info("Caricamento mercati Kraken")
            markets = self.call_exchange(lambda: self.exchange.load_markets(reload=True))
            self.markets = markets
            self.symbols = [
                symbol
                for symbol, market in markets.items()
                if self.is_tradeable_market(symbol, market)
            ]
            self.symbols.sort()
            self.last_market_refresh_ts = time.time()
            self.logger.info("Mercati monitorati: %d", len(self.symbols))
            self.telegram.send_message(
                "Connessione Kraken riuscita\n"
                f"Mercati USD/USDT monitorati: {len(self.symbols)}"
            )
        except Exception as exc:
            self.last_error = str(exc)
            self.logger.exception("Errore caricamento mercati: %s", exc)
            self.telegram.send_message(f"Errore caricamento mercati Kraken: {exc}")
            raise

    def is_tradeable_market(self, symbol: str, market: Dict[str, Any]) -> bool:
        try:
            if not market.get("active", True):
                return False
            if market.get("spot") is False:
                return False
            base = str(market.get("base", "")).upper()
            quote = str(market.get("quote", "")).upper()
            market_id = str(market.get("id", "")).upper()
            if quote not in self.config.quote_currencies:
                return False
            if base in self.config.stablecoins or base in self.config.fiat_assets:
                return False
            if self.is_leveraged_token(base, symbol.upper(), market_id):
                return False
            return True
        except Exception:
            return False

    def is_leveraged_token(self, base: str, symbol: str, market_id: str) -> bool:
        try:
            raw = {base.upper(), symbol.upper(), market_id.upper()}
            for item in raw:
                clean = item.replace("/", "").replace("-", "").replace("_", "")
                for token in self.config.leveraged_tokens:
                    if clean.endswith(token) or clean.startswith(token):
                        return True
            return False
        except Exception:
            return True

    def connect(self) -> None:
        self.init_exchange()
        self.telegram.start_polling()
        self.telegram.send_message(
            "Bot avviato\n"
            "Exchange: Kraken reale\n"
            f"Timeframe: {self.config.timeframe}\n"
            f"Modalita dry run: {self.config.dry_run}"
        )
        self.load_markets()
        self.refresh_balance(send_telegram=True)

    def refresh_balance(self, send_telegram: bool = False) -> None:
        try:
            if self.exchange is None:
                return
            balance = self.call_exchange(self.exchange.fetch_balance)
            self.last_balance = balance
            equity = self.estimate_equity_usd(balance)
            self.current_equity_usd = equity
            self.risk.update_equity(equity)
            if send_telegram:
                self.telegram.send_message(self.format_balance(balance, equity))
        except Exception as exc:
            self.last_error = str(exc)
            self.logger.exception("Errore fetch_balance(): %s", exc)
            self.telegram.send_message(f"Errore saldo Kraken: {exc}")

    def estimate_equity_usd(self, balance: Dict[str, Any]) -> float:
        try:
            total = balance.get("total", {}) or {}
            equity = float(total.get("USD", 0.0) or 0.0)
            equity += float(total.get("USDT", 0.0) or 0.0)
            if self.exchange is None:
                return equity
            for asset, amount_raw in total.items():
                try:
                    asset = str(asset).upper()
                    amount = float(amount_raw or 0.0)
                    if amount <= 0 or asset in {"USD", "USDT"}:
                        continue
                    symbol = self.find_valuation_symbol(asset)
                    if not symbol:
                        continue
                    ticker = self.call_exchange(self.exchange.fetch_ticker, symbol)
                    price = float(ticker.get("last") or ticker.get("close") or 0.0)
                    if price > 0:
                        equity += amount * price
                        self.last_prices[symbol] = price
                        time.sleep(self.config.per_symbol_delay_seconds)
                except Exception:
                    continue
            return float(equity)
        except Exception:
            return 0.0

    def find_valuation_symbol(self, asset: str) -> str:
        try:
            for quote in ("USD", "USDT"):
                symbol = f"{asset}/{quote}"
                if symbol in self.markets:
                    return symbol
            return ""
        except Exception:
            return ""

    def scan_markets(self) -> None:
        try:
            if self.exchange is None:
                return
            self.scan_count += 1
            self.last_scan_started_at = self.now_iso()
            best: List[MarketSignal] = []
            liquid_count = 0
            errors = 0

            for symbol in list(self.symbols):
                if self.shutdown_requested:
                    break
                try:
                    ohlcv = self.call_exchange(
                        self.exchange.fetch_ohlcv,
                        symbol,
                        self.config.timeframe,
                        limit=self.config.ohlcv_limit,
                    )
                    signal = self.strategy.analyze(symbol, ohlcv)
                    if signal is None:
                        time.sleep(self.config.per_symbol_delay_seconds)
                        continue
                    best.append(signal)
                    self.last_prices[symbol] = signal.price
                    if (
                        signal.metrics.get("quote_volume_24h", 0.0)
                        >= self.config.min_24h_quote_volume_usd
                    ):
                        liquid_count += 1
                    if signal.buy:
                        self.try_open_trade(signal)
                    time.sleep(self.config.per_symbol_delay_seconds)
                except Exception as exc:
                    errors += 1
                    self.last_error = str(exc)
                    self.logger.warning("Errore scansione %s: %s", symbol, exc)
                    if errors % 10 == 0:
                        self.telegram.send_message(
                            f"Errori scansione mercato: {errors}\nUltimo: {symbol} {exc}",
                            disable_notification=True,
                        )
                    time.sleep(self.config.per_symbol_delay_seconds)

            self.best_signals = sorted(best, key=lambda item: item.score, reverse=True)[
                : self.config.top_signals_limit
            ]
            self.liquid_symbols_last_scan = liquid_count
            self.last_scan_finished_at = self.now_iso()
            self.logger.info(
                "Scan completato: totali=%d liquidi=%d errori=%d top=%d",
                len(self.symbols),
                liquid_count,
                errors,
                len(self.best_signals),
            )
            self.send_periodic_signal_report()
        except Exception as exc:
            self.last_error = str(exc)
            self.logger.exception("Errore scan mercati: %s", exc)
            self.telegram.send_message(f"Errore scan mercati: {exc}")

    def manage_positions(self) -> None:
        try:
            if self.exchange is None:
                return
            for symbol in list(self.risk.positions.keys()):
                if self.shutdown_requested:
                    break
                try:
                    position = self.risk.positions.get(symbol)
                    if position is None:
                        continue
                    ohlcv = self.call_exchange(
                        self.exchange.fetch_ohlcv,
                        symbol,
                        self.config.timeframe,
                        limit=self.config.ohlcv_limit,
                    )
                    exit_check = self.strategy.exit_signal(ohlcv)
                    metrics = exit_check.get("metrics", {})
                    price = float(metrics.get("price") or self.fetch_last_price(symbol))
                    atr = float(metrics.get("atr") or position.entry_price * 0.01)
                    updated = self.risk.update_trailing_stop(symbol, price, atr)
                    if updated is None:
                        continue
                    self.last_prices[symbol] = price

                    reason = ""
                    if price <= updated.stop_loss:
                        reason = "stop loss"
                    elif price <= updated.trailing_stop:
                        reason = "trailing stop"
                    elif price >= updated.take_profit:
                        reason = "take profit"
                    elif exit_check.get("exit"):
                        reason = str(exit_check.get("reason") or "uscita strategia")

                    if reason:
                        self.close_position(symbol, reason, price)
                    time.sleep(self.config.per_symbol_delay_seconds)
                except Exception as exc:
                    self.last_error = str(exc)
                    self.logger.exception("Errore gestione posizione %s: %s", symbol, exc)
                    self.telegram.send_message(f"Errore posizione {symbol}: {exc}")
        except Exception as exc:
            self.last_error = str(exc)
            self.logger.exception("Errore gestione posizioni: %s", exc)

    def try_open_trade(self, signal: MarketSignal) -> None:
        try:
            if not self.trading_enabled:
                return
            if self.exchange is None:
                return
            market = self.markets.get(signal.symbol)
            if not market:
                return
            quote = str(market.get("quote", "")).upper()
            base = str(market.get("base", "")).upper()
            self.refresh_balance(send_telegram=False)
            free = self.last_balance.get("free", {}) or {}
            quote_free = float(free.get(quote, 0.0) or 0.0)
            allowed, reason = self.risk.can_open_trade(
                signal.symbol, self.current_equity_usd, quote_free
            )
            if not allowed:
                self.logger.info("Segnale %s ignorato: %s", signal.symbol, reason)
                return
            capital = self.risk.capital_for_trade(self.current_equity_usd, quote_free)
            if capital <= 0:
                return
            amount = capital / signal.price
            amount = self.precise_amount(signal.symbol, amount)
            if amount <= 0:
                return
            if not self.order_respects_limits(signal.symbol, amount, capital):
                self.logger.info("Ordine %s sotto limiti mercato", signal.symbol)
                return

            if self.config.dry_run:
                order = {
                    "id": f"dry-buy-{int(time.time())}",
                    "average": signal.price,
                    "filled": amount,
                    "cost": amount * signal.price,
                    "fee": {"cost": 0.0},
                }
            else:
                order = self.call_exchange(
                    self.exchange.create_market_buy_order, signal.symbol, amount
                )

            entry_price = self.order_average(order, signal.price)
            filled = float(order.get("filled") or amount)
            quote_cost = float(order.get("cost") or filled * entry_price)
            fees = self.extract_fees(order)
            levels = self.risk.protective_levels(
                entry_price, signal.metrics.get("atr", entry_price * 0.01)
            )
            position = Position(
                symbol=signal.symbol,
                base=base,
                quote=quote,
                amount=filled,
                entry_price=entry_price,
                entry_time=self.now_iso(),
                stop_loss=levels["stop_loss"],
                take_profit=levels["take_profit"],
                trailing_stop=levels["trailing_stop"],
                highest_price=entry_price,
                order_id=str(order.get("id", "")),
                strategy_score=signal.score,
                quote_cost=quote_cost,
                fees=fees,
            )
            self.risk.register_position(position)
            self.telegram.send_message(
                "TRADE APERTO\n"
                f"{signal.symbol}\n"
                f"Prezzo: {entry_price:.10g}\n"
                f"Quantita: {filled:.10g}\n"
                f"Capitale: {quote_cost:.2f} {quote}\n"
                f"Stop loss: {position.stop_loss:.10g}\n"
                f"Trailing stop: {position.trailing_stop:.10g}\n"
                f"Take profit: {position.take_profit:.10g}\n"
                f"Score: {signal.score:.2f}\n"
                f"Motivi: {', '.join(signal.reasons)}"
            )
        except Exception as exc:
            self.last_error = str(exc)
            self.logger.exception("Errore apertura trade %s: %s", signal.symbol, exc)
            self.telegram.send_message(f"Errore apertura trade {signal.symbol}: {exc}")

    def close_position(self, symbol: str, reason: str, fallback_price: float) -> None:
        try:
            if self.exchange is None:
                return
            position = self.risk.positions.get(symbol)
            if position is None:
                return
            self.refresh_balance(send_telegram=False)
            free = self.last_balance.get("free", {}) or {}
            available_base = float(free.get(position.base, position.amount) or 0.0)
            amount = min(position.amount, available_base if available_base > 0 else position.amount)
            amount = self.precise_amount(symbol, amount)
            if amount <= 0:
                self.telegram.send_message(
                    f"ATTENZIONE\nImpossibile chiudere {symbol}: saldo {position.base} non disponibile"
                )
                return

            if self.config.dry_run:
                order = {
                    "id": f"dry-sell-{int(time.time())}",
                    "average": fallback_price,
                    "filled": amount,
                    "cost": amount * fallback_price,
                    "fee": {"cost": 0.0},
                }
            else:
                order = self.call_exchange(
                    self.exchange.create_market_sell_order, symbol, amount
                )
            exit_price = self.order_average(order, fallback_price)
            fees = self.extract_fees(order)
            closed = self.risk.close_position(
                symbol,
                exit_price,
                reason,
                fees=fees,
                exit_order_id=str(order.get("id", "")),
            )
            if not closed:
                return
            self.telegram.send_message(
                "TRADE CHIUSO\n"
                f"{symbol}\n"
                f"Motivo: {reason}\n"
                f"Entry: {closed['entry_price']:.10g}\n"
                f"Exit: {closed['exit_price']:.10g}\n"
                f"PnL netto: {closed['net_pnl']:.2f} {closed['quote']}\n"
                f"PnL %: {closed['pnl_percent']:.2f}%\n"
                f"Perdite consecutive: {self.risk.consecutive_losses}"
            )
            self.refresh_balance(send_telegram=False)
        except Exception as exc:
            self.last_error = str(exc)
            self.logger.exception("Errore chiusura trade %s: %s", symbol, exc)
            self.telegram.send_message(f"Errore chiusura trade {symbol}: {exc}")

    def fetch_last_price(self, symbol: str) -> float:
        try:
            if self.exchange is None:
                return self.last_prices.get(symbol, 0.0)
            ticker = self.call_exchange(self.exchange.fetch_ticker, symbol)
            price = float(ticker.get("last") or ticker.get("close") or 0.0)
            if price > 0:
                self.last_prices[symbol] = price
            return price
        except Exception:
            return self.last_prices.get(symbol, 0.0)

    def precise_amount(self, symbol: str, amount: float) -> float:
        try:
            if self.exchange is None:
                return 0.0
            return float(self.exchange.amount_to_precision(symbol, amount))
        except Exception:
            return 0.0

    def order_respects_limits(self, symbol: str, amount: float, cost: float) -> bool:
        try:
            market = self.markets.get(symbol, {})
            limits = market.get("limits", {}) or {}
            amount_limits = limits.get("amount", {}) or {}
            cost_limits = limits.get("cost", {}) or {}
            min_amount = amount_limits.get("min")
            min_cost = cost_limits.get("min")
            if min_amount is not None and amount < float(min_amount):
                return False
            if min_cost is not None and cost < float(min_cost):
                return False
            return True
        except Exception:
            return False

    def order_average(self, order: Dict[str, Any], fallback: float) -> float:
        try:
            average = order.get("average") or order.get("price") or fallback
            return float(average)
        except Exception:
            return float(fallback)

    def extract_fees(self, order: Dict[str, Any]) -> float:
        try:
            total = 0.0
            fee = order.get("fee") or {}
            if isinstance(fee, dict):
                total += float(fee.get("cost") or 0.0)
            for item in order.get("fees") or []:
                if isinstance(item, dict):
                    total += float(item.get("cost") or 0.0)
            return total
        except Exception:
            return 0.0

    def send_periodic_signal_report(self) -> None:
        try:
            now = time.time()
            if now - self.last_signal_report_ts < self.config.telegram_signal_interval_seconds:
                return
            self.last_signal_report_ts = now
            self.telegram.send_message(self.format_signals(), disable_notification=True)
        except Exception:
            pass

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.request_shutdown)
        signal.signal(signal.SIGINT, self.request_shutdown)
        while not self.shutdown_requested:
            try:
                self.connect()
                while not self.shutdown_requested:
                    if (
                        time.time() - self.last_market_refresh_ts
                        >= self.config.market_refresh_seconds
                    ):
                        self.load_markets()
                    self.refresh_balance(send_telegram=False)
                    self.manage_positions()
                    if self.trading_enabled and not self.risk.daily_stop_hit():
                        self.scan_markets()
                    else:
                        self.logger.info("Trading in pausa")
                    time.sleep(self.config.scan_interval_seconds)
            except Exception as exc:
                self.last_error = str(exc)
                self.logger.exception("Errore loop principale: %s", exc)
                self.telegram.send_message(f"Errore loop principale: {exc}")
                time.sleep(self.config.error_sleep_seconds)
        self.telegram.send_message("Bot arrestato")
        self.telegram.stop_polling()

    def request_shutdown(self, signum: int, frame: Any) -> None:
        self.shutdown_requested = True

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def format_balance(self, balance: Dict[str, Any], equity: float) -> str:
        try:
            free = balance.get("free", {}) or {}
            total = balance.get("total", {}) or {}
            lines = [
                "SALDO ACCOUNT",
                f"Equity stimata: {equity:.2f} USD",
                f"USD free: {float(free.get('USD', 0.0) or 0.0):.2f}",
                f"USDT free: {float(free.get('USDT', 0.0) or 0.0):.2f}",
                f"USD totale: {float(total.get('USD', 0.0) or 0.0):.2f}",
                f"USDT totale: {float(total.get('USDT', 0.0) or 0.0):.2f}",
                f"Drawdown giornaliero: {self.risk.current_drawdown * 100:.2f}%",
                f"PnL giornaliero realizzato: {self.risk.daily_realized_pnl:.2f} USD",
            ]
            return "\n".join(lines)
        except Exception as exc:
            return f"Errore formattazione saldo: {exc}"

    def format_signals(self) -> str:
        try:
            if not self.best_signals:
                return "SEGNALI\nNessun segnale disponibile."
            lines = ["MIGLIORI SEGNALI"]
            for index, signal in enumerate(self.best_signals, start=1):
                metrics = signal.metrics
                lines.append(
                    f"{index}. {signal.symbol} | score {signal.score:.2f} | "
                    f"prezzo {signal.price:.10g} | RSI {metrics.get('rsi', 0):.1f} | "
                    f"vol x{metrics.get('volume_ratio', 0):.2f} | "
                    f"mom {metrics.get('momentum', 0) * 100:.2f}% | "
                    f"BUY {'SI' if signal.buy else 'NO'}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Errore segnali: {exc}"

    def command_balance(self, text: str, message: Dict[str, Any]) -> str:
        self.refresh_balance(send_telegram=False)
        return self.format_balance(self.last_balance, self.current_equity_usd)

    def command_status(self, text: str, message: Dict[str, Any]) -> str:
        try:
            return (
                "STATUS BOT\n"
                f"Trading attivo: {self.trading_enabled}\n"
                f"Dry run: {self.config.dry_run}\n"
                f"Mercati monitorati: {len(self.symbols)}\n"
                f"Coin liquide ultimo scan: {self.liquid_symbols_last_scan}\n"
                f"Trade aperti: {len(self.risk.positions)}/{self.config.max_open_trades}\n"
                f"Equity: {self.current_equity_usd:.2f} USD\n"
                f"Drawdown: {self.risk.current_drawdown * 100:.2f}%\n"
                f"PnL giornaliero: {self.risk.daily_realized_pnl:.2f} USD\n"
                f"Perdite consecutive: {self.risk.consecutive_losses}\n"
                f"Pausa rischio: {self.risk.pause_remaining_minutes()} min\n"
                f"Ultimo scan start: {self.last_scan_started_at or 'n/d'}\n"
                f"Ultimo scan fine: {self.last_scan_finished_at or 'n/d'}\n"
                f"Ultimo errore: {self.last_error or 'nessuno'}"
            )
        except Exception as exc:
            return f"Errore status: {exc}"

    def command_trades(self, text: str, message: Dict[str, Any]) -> str:
        try:
            lines = ["TRADES APERTI"]
            if not self.risk.positions:
                lines.append("Nessun trade aperto.")
            for position in self.risk.positions.values():
                price = self.last_prices.get(position.symbol, position.entry_price)
                pnl = (price - position.entry_price) * position.amount
                lines.append(
                    f"{position.symbol} | qty {position.amount:.10g} | "
                    f"entry {position.entry_price:.10g} | last {price:.10g} | "
                    f"PnL {pnl:.2f} {position.quote} | SL {position.stop_loss:.10g} | "
                    f"TS {position.trailing_stop:.10g} | TP {position.take_profit:.10g}"
                )
            lines.append("")
            lines.append("ULTIMI TRADE CHIUSI")
            recent = self.risk.closed_trades[-5:]
            if not recent:
                lines.append("Nessun trade chiuso.")
            for trade in reversed(recent):
                lines.append(
                    f"{trade['symbol']} | PnL {float(trade['net_pnl']):.2f} "
                    f"{trade['quote']} | {float(trade['pnl_percent']):.2f}% | "
                    f"{trade['reason']}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Errore trades: {exc}"

    def command_profit(self, text: str, message: Dict[str, Any]) -> str:
        try:
            unrealized = self.risk.open_unrealized_pnl(self.last_prices)
            total_closed = self.risk.total_closed_pnl()
            return (
                "PROFITTO\n"
                f"PnL giornaliero realizzato: {self.risk.daily_realized_pnl:.2f} USD\n"
                f"PnL aperto stimato: {unrealized:.2f} USD\n"
                f"PnL totale chiuso: {total_closed:.2f} USD\n"
                f"Drawdown giornaliero: {self.risk.current_drawdown * 100:.2f}%\n"
                f"Trade chiusi totali: {len(self.risk.closed_trades)}"
            )
        except Exception as exc:
            return f"Errore profitto: {exc}"

    def command_market(self, text: str, message: Dict[str, Any]) -> str:
        try:
            return (
                "MERCATO\n"
                f"Exchange: Kraken\n"
                f"Coppie USD/USDT filtrate: {len(self.symbols)}\n"
                f"Coin liquide ultimo scan: {self.liquid_symbols_last_scan}\n"
                f"Timeframe: {self.config.timeframe}\n"
                f"Volume minimo 24h: {self.config.min_24h_quote_volume_usd:.0f} USD\n"
                f"Scan completati: {self.scan_count}\n"
                f"Ultimo scan fine: {self.last_scan_finished_at or 'n/d'}"
            )
        except Exception as exc:
            return f"Errore mercato: {exc}"

    def command_signals(self, text: str, message: Dict[str, Any]) -> str:
        return self.format_signals()

    def command_start(self, text: str, message: Dict[str, Any]) -> str:
        self.trading_enabled = True
        return "Trading riattivato. Il bot continua a gestire rischio e nuove entrate."

    def command_stop(self, text: str, message: Dict[str, Any]) -> str:
        self.trading_enabled = False
        return "Trading sospeso. Le posizioni aperte restano gestite da stop, trailing e take profit."

    def command_help(self, text: str, message: Dict[str, Any]) -> str:
        return (
            "COMANDI\n"
            "/saldo - saldo account\n"
            "/status - stato bot e strategia\n"
            "/trades - trade aperti e ultimi chiusi\n"
            "/profitto - pnl, drawdown e statistiche\n"
            "/mercato - mercato monitorato\n"
            "/segnali - migliori segnali\n"
            "/start - riattiva nuove entrate\n"
            "/stop - sospende nuove entrate\n"
            "/help - elenco comandi"
        )


if __name__ == "__main__":
    KrakenTradingBot().run()
