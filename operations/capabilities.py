"""Static inventory of what each pipeline node and module really does.

Status meanings:
- active: its output can change an entry decision or an exit.
- advisory: produces information that reviews or logs may read; it cannot place or block an order alone.
- experimental: runs but is not validated; it must not influence orders.
- disabled: not wired in, or switched off by default.

Every graph node must appear here (a test enforces it) so the dashboard never
overstates what is running.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class Capability(TypedDict):
    name: str
    status: Literal["active", "advisory", "experimental", "disabled"]
    module: str
    notes: str


CAPABILITIES: list[Capability] = [
    {"name": "market_data", "status": "active", "module": "agents/market_data_agent.py",
     "notes": "Fetches bars and quotes; stale or incomplete data blocks entries."},
    {"name": "kafka_events", "status": "experimental", "module": "data/event_driven_kafka_stream.py",
     "notes": "Optional event stream; off unless KAFKA_ENABLED is set and never gates an order."},
    {"name": "l2_order_flow", "status": "experimental", "module": "data/l2_order_flow_analyzer.py",
     "notes": "Order-flow features are informational only."},
    {"name": "hmm_regime", "status": "advisory", "module": "agents/graph.py, risk/hmm_regime_classifier.py",
     "notes": "Regime label is advisory, but a degraded or panic regime is a hard entry gate."},
    {"name": "quant_screener", "status": "active", "module": "agents/quantitative_screener.py",
     "notes": "Liquidity, spread, volume and volatility screen; rejected symbols cannot be entered."},
    {"name": "strategy_candidates", "status": "disabled", "module": "agents/strategy_candidates.py",
     "notes": "Runs only strategy sleeves whose own flag is on; every flag defaults to off."},
    {"name": "shared_news", "status": "advisory", "module": "agents/graph.py",
     "notes": "One news snapshot shared by the analysts."},
    {"name": "technical", "status": "active", "module": "agents/technical_agent.py",
     "notes": "Deterministic CompositeTrend signal, stop and target; the only source of entry levels."},
    {"name": "fundamental", "status": "advisory", "module": "agents/fundamental_agent.py",
     "notes": "Evidence for the AI review; not a gate."},
    {"name": "sentiment", "status": "active", "module": "agents/sentiment_agent.py",
     "notes": "Required sentiment evidence per candidate; missing or invalid evidence blocks the entry."},
    {"name": "options", "status": "advisory", "module": "agents/options_agent.py",
     "notes": "No open-interest provider is available, so options analytics are limited."},
    {"name": "macro", "status": "advisory", "module": "agents/macro_agent.py",
     "notes": "Index, crude and FX context for the AI review."},
    {"name": "india_specific", "status": "advisory", "module": "agents/india_specific_agent.py",
     "notes": "F&O ban, circuit and expiry context; expiry comes from the contract master, or is unknown."},
    {"name": "geopolitical_nlp", "status": "advisory", "module": "agents/geopolitical_nlp_agent.py",
     "notes": "Runs after macro; rule-based fallback reads the nested macro data."},
    {"name": "bull_bear_debate", "status": "active", "module": "agents/bull_bear_synthesis_agent.py",
     "notes": "Required AI review: an entry needs an aligned, stored, unchanged debate."},
    {"name": "risk_manager", "status": "active", "module": "agents/risk_manager_agent.py",
     "notes": "Deterministic approval and sizing inputs; entries need explicit approval."},
    {"name": "marl_optimizer", "status": "experimental", "module": "risk/marl_portfolio_optimizer.py",
     "notes": "Allocation weights are logged only and do not size orders."},
    {"name": "delta_hedger", "status": "experimental", "module": "risk/ai_options_delta_hedger.py",
     "notes": "Greeks logging only; derivatives are out of scope."},
    {"name": "compliance_check", "status": "advisory", "module": "config/fema_lrs_controller.py",
     "notes": "FEMA/LRS bookkeeping for foreign markets; inactive for Indian equity."},
    {"name": "portfolio_manager", "status": "active", "module": "agents/portfolio_manager_agent.py",
     "notes": "Strict required AI review that can approve or hold; cannot create levels."},
    {"name": "order_router", "status": "active", "module": "agents/graph.py",
     "notes": "All entry gates (fail closed) plus exits, which never depend on those gates."},
    {"name": "episodic_reflection", "status": "advisory", "module": "agents/episodic_reflection_agent.py",
     "notes": "Post-trade notes; never feeds back into live decisions."},
    {"name": "signal_gate", "status": "disabled", "module": "models/signal_gate/gate.py",
     "notes": "Calibrated trade filter. Off (SIGNAL_GATE_ENABLED) until enough settled outcomes exist."},
    {"name": "strategy_selector", "status": "disabled", "module": "models/signal_gate/strategy_selector.py",
     "notes": "Rule-based sleeve chooser; used only by strategy_candidates when a sleeve is enabled."},
    {"name": "alpha_miner", "status": "disabled", "module": "agents/evolutionary_alpha_miner.py",
     "notes": "Research only; never wired to orders."},
    {"name": "stop_loss_manager", "status": "active", "module": "risk/stop_loss_manager.py",
     "notes": "Stops, targets, time exits; independent of AI, pause, budget and new-contract checks."},
    {"name": "dead_mans_switch", "status": "active", "module": "risk/dead_mans_switch.py",
     "notes": "Halts new entries only; exits are not gated by it."},
    {"name": "alpaca_us_paper", "status": "disabled", "module": "broker/alpaca_paper_adapter.py",
     "notes": "US paper sleeve; off until FOREIGN_US_PAPER_ENABLED and paper keys exist."},
    {"name": "crypto_sim", "status": "disabled", "module": "broker/crypto_sim.py",
     "notes": "Local crypto spot simulator; off until CRYPTO_SIM_ENABLED."},
]


def capability_names() -> set[str]:
    return {item["name"] for item in CAPABILITIES}


def current_capabilities() -> list[Capability]:
    """The inventory with flag-controlled foreign sleeves reflecting the loaded settings."""
    from config.settings import settings

    enabled = {
        "alpaca_us_paper": bool(
            settings.foreign_us_paper_enabled and settings.alpaca_paper
            and settings.alpaca_api_key and settings.alpaca_secret_key
        ),
        "crypto_sim": bool(settings.crypto_sim_enabled),
    }
    result: list[Capability] = []
    for item in CAPABILITIES:
        if enabled.get(item["name"]):
            item = Capability(
                name=item["name"], status="experimental", module=item["module"],
                notes="Scheduled long-only trend strategy is active in paper/simulation only; "
                      "results are experimental and not validated.",
            )
        result.append(item)
    return result
