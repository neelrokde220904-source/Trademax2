"""Base agent and shared TradingState TypedDict for the multi-agent system."""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Annotated, Any, TypedDict

import pytz
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")


class TradingState(TypedDict):
    """Shared state for the LangGraph multi-agent trading pipeline.

    v1.0 fields are preserved; v2.0 fields are added below the separator.
    """

    # Input
    watchlist: list[str]
    date: str

    # Market data (populated by market_data_agent)
    market_data: dict[str, Any]       # {symbol: DataFrame of OHLCV}
    market_data_provenance: dict[str, dict[str, Any]]
    live_quotes: dict[str, Any]       # {symbol: {ltp, volume, oi, ...}}
    india_vix: float
    gift_nifty: float
    fii_dii_data: dict[str, Any]
    news_snapshot: dict[str, Any]

    # Agent signals (populated by each analyst agent)
    technical_signals: dict[str, Any]
    fundamental_signals: dict[str, Any]
    sentiment_signals: dict[str, Any]
    options_signals: dict[str, Any]
    macro_signals: dict[str, Any]
    india_signals: dict[str, Any]

    # Risk (populated by risk_manager_agent)
    risk_assessment: dict[str, Any]
    portfolio_value: float
    available_capital: float
    current_positions: dict[str, Any]
    portfolio_snapshot_valid: bool

    # Final decisions (populated by portfolio_manager_agent)
    trading_decisions: list[dict[str, Any]]

    # Execution (populated by order_manager)
    executed_orders: list[dict[str, Any]]

    # Meta
    agent_logs: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

    # ────────────── v2.0 additions ──────────────────────

    # Quantitative screen evidence, carried through to the order router
    screener_results: dict[str, Any]

    # HMM Regime (populated by hmm_regime node)
    hmm_regime: dict[str, Any]          # {regime, directive, confidence, ...}

    # L2 / Order Flow (populated by l2_order_flow node)
    ofi_signals: dict[str, Any]         # {symbol: {ofi_zscore, signal, absorption}}

    # Event-Driven Kafka (populated by kafka_events node)
    active_macro_event: dict[str, Any]  # Latest critical event from Kafka stream

    # Geopolitical NLP (populated by geopolitical_nlp node)
    geopolitical_risks: dict[str, Any]  # {risk_level, events, stock_impacts}

    # Bull/Bear Debate (populated by bull_bear_synthesis node)
    debate_results: dict[str, Any]      # {net_conviction, bull_score, bear_score, ...}
    review_contract_results: dict[str, Any]
    scan_started_monotonic: float | None  # time.monotonic() when the pipeline began
    strategy_candidates: dict[str, Any]   # {symbol: candidate} from enabled strategy sleeves (empty by default)
    # MARL Portfolio Optimizer (populated by marl_optimizer node)
    optimal_allocations: dict[str, float]  # {symbol: target_weight}

    # AI Delta Hedger (populated by delta_hedger node)
    portfolio_greeks: dict[str, Any]    # {portfolio_delta, hedges_needed, ...}

    # Global Quotes (populated by global broker feeds)
    global_quotes: dict[str, Any]       # {symbol: {price, exchange, currency}}

    # FEMA / LRS (populated by compliance node)
    lrs_status: dict[str, Any]          # {remaining_usd, tcs_pct, ytd_utilized}

    # Evolved Alpha Signals (from overnight alpha miner)
    evolved_alpha_signals: dict[str, Any]  # {factor_id: signal_value}

    # Broker Balances (populated by banking_gateway)
    broker_balances: dict[str, Any]     # {angel: ..., ibkr: ..., alpaca: ...}

    # Episodic Reflection Heuristics
    reflection_heuristics: list[dict[str, Any]]
    similar_past_trades: dict[str, Any]
    llm_profile: dict[str, Any]
    execution_enabled: bool
    five_minute_bars: dict[str, Any]
    research_mode: bool
    research_as_of: str
    research_daily_loss_hit: bool
    research_fundamental_inputs: dict[str, Any]
    research_options_inputs: dict[str, Any]
    research_macro_inputs: dict[str, Any]
    research_india_inputs: dict[str, Any]
    research_capture: dict[str, Any]


def create_initial_state(watchlist: list[str] | None = None) -> TradingState:
    """Create a fresh TradingState with default values."""
    from config.settings import settings

    return TradingState(
        watchlist=watchlist or settings.default_watchlist,
        date=datetime.now(IST).strftime("%Y-%m-%d"),
        market_data={},
        market_data_provenance={},
        live_quotes={},
        india_vix=0.0,
        gift_nifty=0.0,
        fii_dii_data={},
        news_snapshot={},
        technical_signals={},
        fundamental_signals={},
        sentiment_signals={},
        options_signals={},
        macro_signals={},
        india_signals={},
        risk_assessment={},
        portfolio_value=settings.initial_capital,
        available_capital=settings.initial_capital,
        current_positions={},
        portfolio_snapshot_valid=False,
        trading_decisions=[],
        executed_orders=[],
        agent_logs=[],
        errors=[],
        # v2.0 defaults
        screener_results={},
        hmm_regime={},
        ofi_signals={},
        active_macro_event={},
        geopolitical_risks={},
        debate_results={},
        review_contract_results={},
        scan_started_monotonic=None,
        strategy_candidates={},
        portfolio_greeks={},
        global_quotes={},
        lrs_status={},
        evolved_alpha_signals={},
        broker_balances={},
        reflection_heuristics=[],
        similar_past_trades={},
        llm_profile={},
        execution_enabled=True,
        five_minute_bars={},
        research_mode=False,
        research_as_of="",
        research_daily_loss_hit=False,
        research_fundamental_inputs={},
        research_options_inputs={},
        research_macro_inputs={},
        research_india_inputs={},
        research_capture={},
    )


class BaseAgent(ABC):
    """Base class for all trading agents."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._logger = logger.bind(agent=name)

    @abstractmethod
    def run(self, state: TradingState) -> dict[str, Any]:
        """Execute the agent's analysis and return state updates.

        Returns:
            Dict of state keys to update.
        """
        ...

    def log(self, message: str, *args: Any) -> str:
        """Log a message and return it for the agent_logs list."""
        formatted = message.format(*args) if args else message
        self._logger.info(formatted)
        return f"[{self.name}] {datetime.now(IST).strftime('%H:%M:%S')} — {formatted}"
