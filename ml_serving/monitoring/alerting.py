"""Alert manager for monitoring thresholds and triggering actions."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Condition(str, Enum):
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    EQUAL = "eq"


@dataclass
class AlertRule:
    """A rule that triggers an alert when a condition is met."""

    name: str
    metric: str
    condition: Condition
    threshold: float
    severity: Severity = Severity.WARNING
    action: str = "log"  # "log" or "webhook"
    webhook_url: str = ""
    cooldown_seconds: float = 60.0
    last_triggered: float = 0.0


@dataclass
class Alert:
    """A triggered alert."""

    rule_name: str
    metric: str
    metric_value: float
    threshold: float
    severity: Severity
    timestamp: float = field(default_factory=time.time)
    message: str = ""


class AlertManager:
    """Manage alert rules and check metric values against thresholds."""

    def __init__(self) -> None:
        self._rules: dict[str, AlertRule] = {}
        self._alerts: list[Alert] = []
        self._metric_providers: dict[str, Callable[[], float]] = {}
        self._webhook_fn: Callable[[str, Alert], None] | None = None
        self._lock = threading.Lock()

    def add_rule(
        self,
        name: str,
        metric: str,
        condition: Condition | str,
        threshold: float,
        severity: Severity | str = Severity.WARNING,
        action: str = "log",
        webhook_url: str = "",
        cooldown_seconds: float = 60.0,
    ) -> AlertRule:
        """Add a new alert rule."""
        if isinstance(condition, str):
            condition = Condition(condition)
        if isinstance(severity, str):
            severity = Severity(severity)

        rule = AlertRule(
            name=name,
            metric=metric,
            condition=condition,
            threshold=threshold,
            severity=severity,
            action=action,
            webhook_url=webhook_url,
            cooldown_seconds=cooldown_seconds,
        )
        with self._lock:
            self._rules[name] = rule
        return rule

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule."""
        with self._lock:
            self._rules.pop(name, None)

    def register_metric_provider(
        self, metric_name: str, provider: Callable[[], float]
    ) -> None:
        """Register a callable that returns the current value for a metric."""
        self._metric_providers[metric_name] = provider

    def set_webhook_handler(self, fn: Callable[[str, Alert], None]) -> None:
        """Set the function used to send webhook alerts."""
        self._webhook_fn = fn

    def check_rules(self, metric_values: dict[str, float] | None = None) -> list[Alert]:
        """Evaluate all rules against current metric values.

        If metric_values is not provided, uses registered metric providers.
        """
        if metric_values is None:
            metric_values = {}
            for name, provider in self._metric_providers.items():
                try:
                    metric_values[name] = provider()
                except Exception:
                    logger.warning("metric_provider_failed", metric=name, exc_info=True)

        triggered: list[Alert] = []
        now = time.time()

        with self._lock:
            rules = list(self._rules.values())

        for rule in rules:
            if rule.metric not in metric_values:
                continue

            value = metric_values[rule.metric]

            if not self._check_condition(value, rule.condition, rule.threshold):
                continue

            # Check cooldown
            if now - rule.last_triggered < rule.cooldown_seconds:
                continue

            alert = Alert(
                rule_name=rule.name,
                metric=rule.metric,
                metric_value=value,
                threshold=rule.threshold,
                severity=rule.severity,
                message=f"{rule.name}: {rule.metric}={value:.4f} {rule.condition.value} {rule.threshold}",
            )

            rule.last_triggered = now
            triggered.append(alert)
            self._execute_action(rule, alert)

        with self._lock:
            self._alerts.extend(triggered)

        return triggered

    def get_alerts(self, limit: int = 100) -> list[Alert]:
        """Return recent alerts."""
        with self._lock:
            return list(self._alerts[-limit:])

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        with self._lock:
            self._alerts.clear()

    @staticmethod
    def _check_condition(value: float, condition: Condition, threshold: float) -> bool:
        if condition == Condition.GREATER_THAN:
            return value > threshold
        elif condition == Condition.LESS_THAN:
            return value < threshold
        elif condition == Condition.GREATER_EQUAL:
            return value >= threshold
        elif condition == Condition.LESS_EQUAL:
            return value <= threshold
        elif condition == Condition.EQUAL:
            return value == threshold
        return False

    def _execute_action(self, rule: AlertRule, alert: Alert) -> None:
        """Execute the configured action for a triggered alert."""
        if rule.action == "log":
            logger.warning(
                "alert_triggered",
                rule=rule.name,
                metric=rule.metric,
                value=alert.metric_value,
                threshold=rule.threshold,
                severity=rule.severity.value,
            )
        elif rule.action == "webhook" and self._webhook_fn:
            try:
                self._webhook_fn(rule.webhook_url, alert)
            except Exception:
                logger.error("webhook_failed", url=rule.webhook_url, exc_info=True)
