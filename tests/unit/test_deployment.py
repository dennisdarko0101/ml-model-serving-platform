"""Tests for Cloud Run and Docker deployment (all GCP/Docker calls mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ml_serving.deployment.cloud_run import CloudRunDeployer, DeploymentInfo, ServiceStatus
from ml_serving.deployment.docker_deploy import DockerDeployer


# ===================================================================
# CloudRunDeployer (mocked client)
# ===================================================================


class TestCloudRunDeployer:
    @pytest.fixture()
    def mock_client(self):
        client = MagicMock()
        # create_service returns an object with url and revision
        result = MagicMock()
        result.url = "https://ml-serving-abc.a.run.app"
        result.revision = "ml-serving-00001"
        result.status = "ACTIVE"
        result.ready = True
        client.create_service.return_value = result
        client.update_service.return_value = result
        client.get_service.return_value = result
        return client

    @pytest.fixture()
    def deployer(self, mock_client):
        return CloudRunDeployer(
            project_id="test-project",
            region="us-central1",
            client=mock_client,
        )

    def test_build_image(self, deployer: CloudRunDeployer):
        uri = deployer.build_image("ml-serving", tag="v1.0")
        assert uri == "gcr.io/test-project/ml-serving:v1.0"

    def test_build_image_default_tag(self, deployer: CloudRunDeployer):
        uri = deployer.build_image("ml-serving")
        assert uri.endswith(":latest")

    def test_deploy(self, deployer: CloudRunDeployer, mock_client):
        info = deployer.deploy(
            "gcr.io/test-project/ml-serving:v1",
            "ml-serving",
            memory="1Gi",
            cpu="2",
            min_instances=1,
            max_instances=5,
        )
        assert info.service_name == "ml-serving"
        assert info.status == "ACTIVE"
        assert info.url == "https://ml-serving-abc.a.run.app"
        assert info.revision == "ml-serving-00001"
        mock_client.create_service.assert_called_once()

    def test_deploy_with_env_vars(self, deployer: CloudRunDeployer, mock_client):
        info = deployer.deploy(
            "gcr.io/test-project/ml-serving:v1",
            "ml-serving",
            env_vars={"LOG_LEVEL": "DEBUG"},
        )
        assert info.status == "ACTIVE"

    def test_update(self, deployer: CloudRunDeployer, mock_client):
        result = MagicMock()
        result.url = "https://ml-serving-abc.a.run.app"
        result.revision = "ml-serving-00002"
        mock_client.update_service.return_value = result

        info = deployer.update("ml-serving", "gcr.io/test-project/ml-serving:v2")
        assert info.service_name == "ml-serving"
        assert info.revision == "ml-serving-00002"
        mock_client.update_service.assert_called_once()

    def test_get_status(self, deployer: CloudRunDeployer):
        status = deployer.get_status("ml-serving")
        assert status.service_name == "ml-serving"
        assert status.ready
        assert status.status == "ACTIVE"

    def test_get_url(self, deployer: CloudRunDeployer):
        url = deployer.get_url("ml-serving")
        assert "ml-serving" in url

    def test_delete(self, deployer: CloudRunDeployer, mock_client):
        deployer.delete("ml-serving")
        mock_client.delete_service.assert_called_once_with(service_name="ml-serving")

    def test_project_and_region(self, deployer: CloudRunDeployer):
        assert deployer.project_id == "test-project"
        assert deployer.region == "us-central1"

    def test_estimate_cost(self, deployer: CloudRunDeployer):
        cost = deployer.estimate_cost(
            memory_mb=512,
            cpu=1.0,
            avg_requests_per_day=10000,
            avg_latency_ms=100,
        )
        assert "total_estimated_usd" in cost
        assert cost["total_estimated_usd"] > 0
        assert cost["monthly_requests"] == 300000

    def test_estimate_cost_low_traffic(self, deployer: CloudRunDeployer):
        cost = deployer.estimate_cost(avg_requests_per_day=100)
        assert cost["total_estimated_usd"] < 1.0

    def test_estimate_cost_high_traffic(self, deployer: CloudRunDeployer):
        cost = deployer.estimate_cost(
            avg_requests_per_day=1000000,
            memory_mb=2048,
            cpu=4.0,
        )
        assert cost["total_estimated_usd"] > 10


# ===================================================================
# DockerDeployer (mocked client)
# ===================================================================


class TestDockerDeployer:
    @pytest.fixture()
    def mock_client(self):
        client = MagicMock()
        result = MagicMock()
        result.id = "abc123def456"
        client.run.return_value = result
        return client

    @pytest.fixture()
    def deployer(self, mock_client):
        return DockerDeployer(client=mock_client)

    def test_build(self, deployer: DockerDeployer, mock_client):
        tag = deployer.build(tag="ml-serving:test")
        assert tag == "ml-serving:test"
        mock_client.build.assert_called_once()

    def test_build_default_tag(self, deployer: DockerDeployer):
        tag = deployer.build()
        assert tag == "ml-serving:latest"

    def test_run(self, deployer: DockerDeployer, mock_client):
        info = deployer.run(
            "ml-serving:latest",
            name="test-container",
            port=9000,
            env_vars={"LOG_LEVEL": "DEBUG"},
        )
        assert info.name == "test-container"
        assert info.port == 9000
        assert info.image == "ml-serving:latest"
        assert info.container_id == "abc123def456"
        mock_client.run.assert_called_once()

    def test_stop(self, deployer: DockerDeployer, mock_client):
        deployer.stop("test-container")
        mock_client.stop.assert_called_once_with(name="test-container")

    def test_health_check_success(self, deployer: DockerDeployer):
        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_http.get.return_value = mock_response

        result = deployer.health_check(
            url="http://localhost:8000/health",
            http_client=mock_http,
        )
        assert result is True

    def test_health_check_failure(self, deployer: DockerDeployer):
        mock_http = MagicMock()
        mock_http.get.side_effect = ConnectionError("refused")

        result = deployer.health_check(
            url="http://localhost:8000/health",
            retries=1,
            http_client=mock_http,
        )
        assert result is False

    def test_health_check_retry_then_success(self, deployer: DockerDeployer):
        mock_http = MagicMock()
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        mock_http.get.side_effect = [fail_resp, ok_resp]

        result = deployer.health_check(
            url="http://localhost:8000/health",
            retries=2,
            http_client=mock_http,
        )
        assert result is True
