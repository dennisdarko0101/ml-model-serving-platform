"""Google Cloud Run deployment management."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

logger = structlog.get_logger()


@dataclass
class DeploymentInfo:
    """Information about a Cloud Run deployment."""

    service_name: str
    url: str = ""
    revision: str = ""
    status: str = "unknown"
    created_at: float = field(default_factory=time.time)
    region: str = ""
    image_uri: str = ""


@dataclass
class ServiceStatus:
    """Current status of a Cloud Run service."""

    service_name: str
    url: str = ""
    latest_revision: str = ""
    status: str = "unknown"
    ready: bool = False
    traffic: list[dict[str, Any]] = field(default_factory=list)


class CloudRunClient(Protocol):
    """Protocol for Cloud Run API operations (enables test mocking)."""

    def create_service(self, **kwargs: Any) -> Any: ...
    def update_service(self, **kwargs: Any) -> Any: ...
    def get_service(self, **kwargs: Any) -> Any: ...
    def delete_service(self, **kwargs: Any) -> Any: ...


class CloudRunDeployer:
    """Deploy and manage services on Google Cloud Run.

    Uses subprocess calls to gcloud CLI by default.
    Accepts an optional client for testing.
    """

    def __init__(
        self,
        project_id: str,
        region: str = "us-central1",
        client: CloudRunClient | None = None,
    ) -> None:
        self._project_id = project_id
        self._region = region
        self._client = client
        self._run_command = _run_gcloud if client is None else None

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def region(self) -> str:
        return self._region

    def build_image(
        self,
        image_name: str,
        tag: str = "latest",
        dockerfile: str = "docker/Dockerfile",
    ) -> str:
        """Build and push a Docker image to GCR.

        Returns the full image URI.
        """
        image_uri = f"gcr.io/{self._project_id}/{image_name}:{tag}"

        if self._client is not None:
            logger.info("image_build_mock", uri=image_uri)
            return image_uri

        _run_gcloud([
            "gcloud", "builds", "submit",
            "--tag", image_uri,
            "--project", self._project_id,
        ])

        logger.info("image_built", uri=image_uri)
        return image_uri

    def deploy(
        self,
        image_uri: str,
        service_name: str,
        *,
        memory: str = "512Mi",
        cpu: str = "1",
        min_instances: int = 0,
        max_instances: int = 10,
        env_vars: dict[str, str] | None = None,
        port: int = 8000,
    ) -> DeploymentInfo:
        """Deploy a new Cloud Run service."""
        if self._client is not None:
            result = self._client.create_service(
                service_name=service_name,
                image_uri=image_uri,
                region=self._region,
                memory=memory,
                cpu=cpu,
                min_instances=min_instances,
                max_instances=max_instances,
            )
            url = getattr(result, "url", f"https://{service_name}-xyz.a.run.app")
            revision = getattr(result, "revision", f"{service_name}-00001")
            return DeploymentInfo(
                service_name=service_name,
                url=url,
                revision=revision,
                status="ACTIVE",
                region=self._region,
                image_uri=image_uri,
            )

        cmd = [
            "gcloud", "run", "deploy", service_name,
            "--image", image_uri,
            "--region", self._region,
            "--project", self._project_id,
            "--memory", memory,
            "--cpu", cpu,
            "--min-instances", str(min_instances),
            "--max-instances", str(max_instances),
            "--port", str(port),
            "--allow-unauthenticated",
            "--quiet",
        ]

        if env_vars:
            env_str = ",".join(f"{k}={v}" for k, v in env_vars.items())
            cmd.extend(["--set-env-vars", env_str])

        _run_gcloud(cmd)

        info = DeploymentInfo(
            service_name=service_name,
            status="DEPLOYING",
            region=self._region,
            image_uri=image_uri,
        )

        try:
            status = self.get_status(service_name)
            info.url = status.url
            info.revision = status.latest_revision
            info.status = status.status
        except Exception:
            pass

        logger.info("service_deployed", service=service_name, image=image_uri)
        return info

    def update(self, service_name: str, image_uri: str) -> DeploymentInfo:
        """Update an existing Cloud Run service with a new image."""
        if self._client is not None:
            result = self._client.update_service(
                service_name=service_name,
                image_uri=image_uri,
            )
            url = getattr(result, "url", f"https://{service_name}-xyz.a.run.app")
            revision = getattr(result, "revision", f"{service_name}-00002")
            return DeploymentInfo(
                service_name=service_name,
                url=url,
                revision=revision,
                status="ACTIVE",
                region=self._region,
                image_uri=image_uri,
            )

        _run_gcloud([
            "gcloud", "run", "deploy", service_name,
            "--image", image_uri,
            "--region", self._region,
            "--project", self._project_id,
            "--quiet",
        ])

        logger.info("service_updated", service=service_name, image=image_uri)
        return DeploymentInfo(
            service_name=service_name,
            status="ACTIVE",
            region=self._region,
            image_uri=image_uri,
        )

    def get_status(self, service_name: str) -> ServiceStatus:
        """Get the current status of a Cloud Run service."""
        if self._client is not None:
            result = self._client.get_service(service_name=service_name)
            return ServiceStatus(
                service_name=service_name,
                url=getattr(result, "url", ""),
                latest_revision=getattr(result, "revision", ""),
                status=getattr(result, "status", "ACTIVE"),
                ready=getattr(result, "ready", True),
            )

        output = _run_gcloud([
            "gcloud", "run", "services", "describe", service_name,
            "--region", self._region,
            "--project", self._project_id,
            "--format", "value(status.url,status.latestCreatedRevisionName)",
        ])

        lines = output.strip().split("\t") if output else ["", ""]
        return ServiceStatus(
            service_name=service_name,
            url=lines[0] if len(lines) > 0 else "",
            latest_revision=lines[1] if len(lines) > 1 else "",
            status="ACTIVE",
            ready=True,
        )

    def get_url(self, service_name: str) -> str:
        """Get the URL for a Cloud Run service."""
        status = self.get_status(service_name)
        return status.url

    def delete(self, service_name: str) -> None:
        """Delete a Cloud Run service."""
        if self._client is not None:
            self._client.delete_service(service_name=service_name)
            logger.info("service_deleted", service=service_name)
            return

        _run_gcloud([
            "gcloud", "run", "services", "delete", service_name,
            "--region", self._region,
            "--project", self._project_id,
            "--quiet",
        ])
        logger.info("service_deleted", service=service_name)

    def estimate_cost(
        self,
        memory_mb: int = 512,
        cpu: float = 1.0,
        avg_requests_per_day: int = 10000,
        avg_latency_ms: int = 100,
    ) -> dict[str, float]:
        """Estimate monthly Cloud Run cost."""
        hours_per_month = 730
        request_cost_per_million = 0.40
        cpu_cost_per_vcpu_second = 0.00002400
        memory_cost_per_gib_second = 0.00000250

        monthly_requests = avg_requests_per_day * 30
        request_cost = (monthly_requests / 1_000_000) * request_cost_per_million

        total_seconds = monthly_requests * (avg_latency_ms / 1000)
        cpu_cost = total_seconds * cpu * cpu_cost_per_vcpu_second
        mem_cost = total_seconds * (memory_mb / 1024) * memory_cost_per_gib_second

        total = request_cost + cpu_cost + mem_cost

        return {
            "monthly_requests": monthly_requests,
            "request_cost_usd": round(request_cost, 2),
            "cpu_cost_usd": round(cpu_cost, 2),
            "memory_cost_usd": round(mem_cost, 2),
            "total_estimated_usd": round(total, 2),
        }


def _run_gcloud(cmd: list[str]) -> str:
    """Run a gcloud CLI command and return stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout
