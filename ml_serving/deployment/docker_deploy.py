"""Local Docker deployment for development and on-premise environments."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import structlog

logger = structlog.get_logger()


class DockerClient(Protocol):
    """Protocol for Docker operations (enables test mocking)."""

    def build(self, **kwargs: Any) -> Any: ...
    def run(self, **kwargs: Any) -> Any: ...
    def stop(self, **kwargs: Any) -> Any: ...
    def ps(self, **kwargs: Any) -> list: ...


@dataclass
class ContainerInfo:
    """Information about a running container."""

    container_id: str
    name: str
    image: str
    status: str = "running"
    port: int = 8000
    created_at: float = field(default_factory=time.time)


class DockerDeployer:
    """Build, run, and manage local Docker containers.

    Uses subprocess calls to docker CLI by default.
    Accepts an optional client for testing.
    """

    def __init__(self, client: DockerClient | None = None) -> None:
        self._client = client

    def build(
        self,
        tag: str = "ml-serving:latest",
        dockerfile: str = "docker/Dockerfile",
        context: str = ".",
    ) -> str:
        """Build a Docker image. Returns the image tag."""
        if self._client is not None:
            self._client.build(tag=tag, dockerfile=dockerfile, context=context)
            logger.info("docker_image_built", tag=tag)
            return tag

        _run_docker([
            "docker", "build",
            "-f", dockerfile,
            "-t", tag,
            context,
        ])

        logger.info("docker_image_built", tag=tag)
        return tag

    def run(
        self,
        image: str,
        *,
        name: str = "ml-serving",
        port: int = 8000,
        env_vars: dict[str, str] | None = None,
        detach: bool = True,
    ) -> ContainerInfo:
        """Run a Docker container."""
        if self._client is not None:
            result = self._client.run(
                image=image, name=name, port=port, env_vars=env_vars
            )
            container_id = getattr(result, "id", "mock-container-id")
            return ContainerInfo(
                container_id=container_id,
                name=name,
                image=image,
                port=port,
            )

        cmd = [
            "docker", "run",
            "--name", name,
            "-p", f"{port}:{port}",
        ]

        if detach:
            cmd.append("-d")

        if env_vars:
            for k, v in env_vars.items():
                cmd.extend(["-e", f"{k}={v}"])

        cmd.append(image)

        output = _run_docker(cmd)
        container_id = output.strip()[:12] if output else "unknown"

        logger.info("container_started", name=name, image=image, port=port)
        return ContainerInfo(
            container_id=container_id,
            name=name,
            image=image,
            port=port,
        )

    def stop(self, container_name: str) -> None:
        """Stop and remove a container."""
        if self._client is not None:
            self._client.stop(name=container_name)
            logger.info("container_stopped", name=container_name)
            return

        _run_docker(["docker", "stop", container_name])
        _run_docker(["docker", "rm", container_name])
        logger.info("container_stopped", name=container_name)

    def health_check(
        self,
        url: str = "http://localhost:8000/health",
        timeout: float = 5.0,
        retries: int = 3,
        http_client: httpx.Client | None = None,
    ) -> bool:
        """Check if a deployed service is healthy."""
        client = http_client or httpx.Client()

        for attempt in range(retries):
            try:
                resp = client.get(url, timeout=timeout)
                if resp.status_code == 200:
                    logger.info("health_check_passed", url=url)
                    return True
            except Exception:
                logger.debug(
                    "health_check_retry",
                    url=url,
                    attempt=attempt + 1,
                )
            if attempt < retries - 1:
                time.sleep(1)

        logger.warning("health_check_failed", url=url, retries=retries)
        return False

    def compose_up(self, compose_file: str = "docker/docker-compose.yml") -> None:
        """Start services with docker-compose."""
        if self._client is not None:
            return

        _run_docker(["docker-compose", "-f", compose_file, "up", "-d"])
        logger.info("compose_started", file=compose_file)

    def compose_down(self, compose_file: str = "docker/docker-compose.yml") -> None:
        """Stop services with docker-compose."""
        if self._client is not None:
            return

        _run_docker(["docker-compose", "-f", compose_file, "down"])
        logger.info("compose_stopped", file=compose_file)


def _run_docker(cmd: list[str]) -> str:
    """Run a docker CLI command and return stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout
