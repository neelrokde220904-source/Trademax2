"""SQLAlchemy models for the India AI Trader system."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class Trade(Base):
    """Executed or paper-traded order record."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(10), default="NSE")
    action = Column(String(10), nullable=False)  # BUY / SELL
    quantity = Column(Integer, nullable=False)
    order_type = Column(String(20), default="LIMIT")  # LIMIT / MARKET / SL / SL-M
    price = Column(Float, nullable=False)
    trigger_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    target = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)
    sector = Column(String(100), nullable=True)
    sector_source = Column(String(80), nullable=True)
    sector_observed_at = Column(String(40), nullable=True)
    # Horizon of the position: INTRADAY / SWING / BTST / POSITIONAL (legacy rows: DELIVERY = SWING).
    holding_period = Column(String(20), default="INTRADAY")
    horizon = Column(String(20), nullable=True)
    strategy_version = Column(String(40), nullable=True)
    exit_policy = Column(String(60), nullable=True)
    atr_timeframe = Column(String(40), nullable=True)
    book = Column(String(40), nullable=True, default="core")  # "core" or the sleeve name
    status = Column(String(20), default="PENDING")  # PENDING / EXECUTED / CANCELLED / REJECTED
    order_id = Column(String(50), nullable=True)  # Broker order ID
    is_paper = Column(Boolean, default=True)
    pnl = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_timestamp = Column(DateTime, nullable=True)
    strategy = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())


class Signal(Base):
    """Agent signal record — every signal from every agent is logged."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    agent_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    signal = Column(String(20), nullable=False)  # BUY / SELL / HOLD / STRONG_BUY / STRONG_SELL
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)
    data_json = Column(Text, nullable=True)  # Serialised dict of full signal payload
    created_at = Column(DateTime, default=func.now())


class Portfolio(Base):
    """Current portfolio snapshot — updated on every trade."""

    __tablename__ = "portfolio"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    total_value = Column(Float, nullable=False)
    available_capital = Column(Float, nullable=False)
    invested_value = Column(Float, nullable=False)
    realised_pnl = Column(Float, default=0.0)
    unrealised_pnl = Column(Float, default=0.0)
    positions_json = Column(Text, nullable=True)  # JSON of open positions
    created_at = Column(DateTime, default=func.now())


class BacktestResult(Base):
    """Backtest run results."""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(50), nullable=False)
    start_date = Column(String(20), nullable=False)
    end_date = Column(String(20), nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return_pct = Column(Float, nullable=True)
    cagr = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    max_drawdown_duration_days = Column(Integer, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    avg_holding_period_hours = Column(Float, nullable=True)
    vs_nifty_return_pct = Column(Float, nullable=True)
    metrics_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())


class DailyPnL(Base):
    """Daily PnL snapshot for equity curve."""

    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True, index=True)
    portfolio_value = Column(Float, nullable=False)
    realised_pnl = Column(Float, default=0.0)
    unrealised_pnl = Column(Float, default=0.0)
    trades_count = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())


class LLMModeSetting(Base):
    __tablename__ = "llm_mode_setting"

    id = Column(Integer, primary_key=True)
    mode = Column(String(20), nullable=False, default="claude")
    revision = Column(Integer, nullable=False, default=0)


class LLMRun(Base):
    __tablename__ = "llm_runs"

    id = Column(String(36), primary_key=True)
    kind = Column(String(80), nullable=False)
    mode = Column(String(20), nullable=False)
    revision = Column(Integer, nullable=False)
    fast_model = Column(String(100), nullable=False)
    decision_model = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False, default="running")
    degraded = Column(Boolean, nullable=False, default=False)
    started_at = Column(DateTime, nullable=False)
    heartbeat_at = Column(DateTime, nullable=False)
    finished_at = Column(DateTime)


class LLMUsage(Base):
    __tablename__ = "llm_usage"

    id = Column(String(36), primary_key=True)
    run_id = Column(String(36), nullable=False, index=True)
    mode = Column(String(20), nullable=False)
    model = Column(String(100), nullable=False)
    task = Column(String(80), nullable=False)
    created_at = Column(DateTime, nullable=False, index=True)
    status = Column(String(40), nullable=False)
    request_id = Column(String(160))
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    cache_read_tokens = Column(Integer)
    cache_write_tokens = Column(Integer)
    reasoning_tokens = Column(Integer)
    latency_ms = Column(Integer)
    estimated_usd = Column(Numeric(20, 10))
    pricing_version = Column(String(100))


class ResearchSnapshot(Base):
    """Captured public research inputs; no brokerage account state."""

    __tablename__ = "research_snapshots"

    id = Column(String(80), primary_key=True)
    observed_at = Column(DateTime, nullable=False, index=True)
    status = Column(String(20), nullable=False)
    problems_json = Column(Text, nullable=False)
    payload_type = Column(String(40), nullable=False)
    payload = Column(LargeBinary, nullable=False)
    payload_sha256 = Column(String(64), nullable=False)


class ResearchForecast(Base):
    """Hypothetical directional forecasts, separate from executed trades."""

    __tablename__ = "research_forecasts"
    __table_args__ = (UniqueConstraint("run_id", "agent", "symbol"),)

    id = Column(String(36), primary_key=True)
    run_id = Column(String(80), nullable=False, index=True)
    observed_at = Column(DateTime, nullable=False, index=True)
    agent = Column(String(40), nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    direction = Column(String(20), nullable=False)
    confidence = Column(Float)
    reference_price = Column(Float)
    reference_source = Column(String(30))
    block_reason = Column(Text)
    status = Column(String(30), nullable=False, index=True)
    settlement_note = Column(Text)
    settlement_session = Column(String(10))
    settlement_price = Column(Float)
    directional_return_pct = Column(Float)
    direction_correct = Column(Boolean)
    settled_at = Column(DateTime)


class CandidateReview(Base):
    """Durable local record of an AI-reviewed entry candidate."""

    __tablename__ = "candidate_reviews"

    candidate_id = Column(String(64), primary_key=True)
    run_id = Column(String(80), nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    contract_version = Column(String(40), nullable=False)
    portfolio_context_json = Column(Text, nullable=False)
    portfolio_context_sha256 = Column(String(64), nullable=False)
    decision_json = Column(Text, nullable=False)
    decision_sha256 = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))


class ResearchComparison(Base):
    """Persisted results of an explicitly requested paper-only comparison."""

    __tablename__ = "research_comparisons"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, nullable=False, index=True)
    status = Column(String(20), nullable=False)
    results_json = Column(Text, nullable=False)


class TradeReflectionSettlement(Base):
    """Idempotency and net-return record for post-trade reflection."""

    __tablename__ = "trade_reflection_settlements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, nullable=False, unique=True, index=True)  # ClosedTrade.id
    settled_at = Column(DateTime, nullable=False, default=func.now())
    net_pnl = Column(Float, nullable=False)
    net_return_pct = Column(Float, nullable=False)
    benchmark_return_pct = Column(Float, nullable=False)
    alpha_pct = Column(Float, nullable=False)
    reflection_json = Column(Text, nullable=False)


class StrategyAlphaWeight(Base):
    """Persisted exponentially updated realized alpha by strategy."""

    __tablename__ = "strategy_alpha_weights"

    strategy = Column(String(80), primary_key=True)
    samples = Column(Integer, nullable=False, default=0)
    mean_alpha_pct = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, nullable=False, default=func.now())


class ClosedTrade(Base):
    """FIFO-matched realized position lot derived from executed order records."""

    __tablename__ = "closed_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_order_id = Column(Integer, nullable=False, index=True)
    exit_order_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    action = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    entry_timestamp = Column(DateTime, nullable=False)
    exit_timestamp = Column(DateTime, nullable=False)
    gross_pnl = Column(Float, nullable=False)
    strategy = Column(String(50), nullable=False, default="unknown")
    holding_period = Column(String(20), nullable=False)
    is_paper = Column(Boolean, nullable=False, default=True)
    venue = Column(String(20), nullable=True)
    currency = Column(String(8), nullable=True)
    exit_policy = Column(String(60), nullable=True)
    book = Column(String(40), nullable=True)


class PaperFillAudit(Base):
    """Evidence for every paper fill attempt, including orders that did not fill."""

    __tablename__ = "paper_fill_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    trade_id = Column(Integer, nullable=True, index=True)
    order_id = Column(String(50), nullable=True)
    symbol = Column(String(50), nullable=False, index=True)
    action = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    order_type = Column(String(20), nullable=False)
    requested_price = Column(Float, nullable=False)
    reference_price = Column(Float, nullable=True)
    reference_source = Column(String(20), nullable=True)  # ws_tick / rest_ltp / requested_price
    reference_age_seconds = Column(Float, nullable=True)
    slippage_pct = Column(Float, nullable=False)
    fill_price = Column(Float, nullable=True)
    outcome = Column(String(30), nullable=False)  # Includes position-change and reference rejections.
    strategy = Column(String(50), nullable=True)


class AlertOutbox(Base):
    """Alerts are stored before delivery so an outage cannot silently drop them."""

    __tablename__ = "alert_outbox"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    dedup_key = Column(String(120), nullable=True, index=True)
    severity = Column(String(12), nullable=False, default="info")
    text = Column(Text, nullable=False)
    parse_mode = Column(String(12), nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    last_attempt_at = Column(DateTime, nullable=True)
    last_error = Column(String(120), nullable=True)
    delivered_at = Column(DateTime, nullable=True)


class CashLedgerEntry(Base):
    """Signed paper cash movements; sale proceeds of delivery products settle on T+1."""

    __tablename__ = "cash_ledger"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=func.now())
    currency = Column(String(8), nullable=False, default="INR")
    amount = Column(Float, nullable=False)  # + cash in, - cash out
    kind = Column(String(20), nullable=False)  # TRADE | FEE | EXTERNAL_FLOW | SETTLEMENT
    settles_on = Column(String(10), nullable=True, index=True)  # ISO date; None = settled at once
    settled = Column(Boolean, nullable=False, default=True)
    trade_id = Column(Integer, nullable=True, index=True)
    __table_args__ = (UniqueConstraint("trade_id", "kind", name="uq_cash_ledger_trade_kind"),)


class CorporateAction(Base):
    """A pending or processed corporate action; unprocessed ones block new entries."""

    __tablename__ = "corporate_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False, index=True)
    action_type = Column(String(30), nullable=False)  # SPLIT | BONUS | DIVIDEND | RIGHTS | MERGER | OTHER
    ex_date = Column(String(10), nullable=False, index=True)  # ISO date
    details = Column(String(200), nullable=True)
    source = Column(String(200), nullable=False)
    processed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=func.now())
    __table_args__ = (UniqueConstraint("symbol", "action_type", "ex_date", name="uq_corporate_action"),)


class EvaluationRun(Base):
    """Every strategy/policy configuration ever evaluated. Rows are appended, never deleted."""

    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    strategy = Column(String(60), nullable=False, index=True)
    exit_policy = Column(String(60), nullable=False)
    capital = Column(Float, nullable=False)
    start_date = Column(String(10), nullable=False)
    end_date = Column(String(10), nullable=False)
    folds = Column(Integer, nullable=False)
    purge_bars = Column(Integer, nullable=False)
    embargo_bars = Column(Integer, nullable=False)
    universe_json = Column(Text, nullable=False)
    data_digest = Column(String(64), nullable=False)
    config_digest = Column(String(64), nullable=False, index=True)
    trade_count = Column(Integer, nullable=False)
    outcome = Column(String(20), nullable=False)  # inconclusive | positive | negative
    metrics_json = Column(Text, nullable=False)
