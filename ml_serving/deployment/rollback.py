"""Rollback management for deployed services."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import httpx
import structlog

from ml_serving.deployment.cloud_run import CloudRunDeployer, DeploymentInfo

logger = structlog.get_logger()


@dataclass
class Checkpoint:
    """A deployment checkpoint for rollback."""

    service_name: str
    revision: str
    image_uri: str
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class RollbackManager:
    """Manage deployment checkpoints and rollback operations."""

    def __init__(self, deployer: CloudRunDeployer) -> None:
        self._deployer = deployer
        self._checkpoints: dict[str, list[Checkpoint]] = {}
        self._lock = threading.Lock()

    def create_checkpoint(
        self,
        service_name: str,
        revision: str,
        image_uri: str = "",
        metadata: dict | None = None,
    ) -> Checkpoint:
        """Create a checkpoint before a new deployment."""
        checkpoint = Checkpoint(
            service_name=service_name,
            revision=revision,
            image_uri=image_uri,
            metadata=metadata or {},
        )

        with self._lock:
            if service_name not in self._checkpoints:
                self._checkpoints[service_name] = []
            self._checkpoints[service_name].append(checkpoint)

        logger.info(
            "checkpoint_created",
            service=service_name,
            revision=revision,
        )
        return checkpoint

    def rollback(self, service_name: str) -> DeploymentInfo:
        """Rollback to the most recent checkpoint."""
        with self._lock:
            checkpoints = self._checkpoints.get(service_name, [])
            if not checkpoints:
                raise ValueError(f"No checkpoints for service '{service_name}'")
            checkpoint = checkpoints[-1]

        if checkpoint.image_uri:
            info = self._deployer.update(service_name, checkpoint.image_uri)
        else:
            info = DeploymentInfo(
                service_name=service_name,
                revision=checkpoint.revision,
                status="ROLLED_BACK",
            )

        logger.info(
            "service_rolled_back",
            service=service_name,
            to_revision=checkpoint.revision,
        )

        # Remove the used checkpoint
        with self._lock:
            if checkpoints:
                checkpoints.pop()

        return info

    def auto_rollback(
        self,
        service_name: str,
        health_check_url: str,
        threshold: int = 3,
        timeout: float = 5.0,
        http_client: httpx.Client | None = None,
    ) -> bool:
        """Check health after deployment and auto-rollback if unhealthy.

        Retries `threshold` times. Returns True if healthy, False if rolled back.
        """
        client = http_client or httpx.Client()
        failures = 0

        for attempt in range(threshold):
            try:
                resp = client.get(health_check_url, timeout=timeout)
                if resp.status_code == 200:
                    logger.info(
                        "health_check_passed",
                        service=service_name,
                        attempt=attempt + 1,
                    )
                    return True
                failures += 1
            except Exception:
                failures += 1
                logger.warning(
                    "health_check_failed",
                    service=service_name,
                    attempt=attempt + 1,
                )

            if attempt < threshold - 1:
                time.sleep(1)

        # All checks failed — rollback
        logger.warning(
            "auto_rollback_triggered",
            service=service_name,
            failures=failures,
        )
        self.rollback(service_name)
        return False

    def list_checkpoints(self, service_name: str) -> list[Checkpoint]:
        """List all checkpoints for a service."""
        with self._lock:
            return list(self._checkpoints.get(service_name, []))

    def clear_checkpoints(self, service_name: str) -> None:
        """Remove all checkpoints for a service."""
        with self._lock:
            self._checkpoints.pop(service_name, None)
