"""Tests for rollback management."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ml_serving.deployment.cloud_run import CloudRunDeployer, DeploymentInfo
from ml_serving.deployment.rollback import Checkpoint, RollbackManager


@pytest.fixture()
def mock_client():
    client = MagicMock()
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
def deployer(mock_client):
    return CloudRunDeployer(
        project_id="test-project",
        region="us-central1",
        client=mock_client,
    )


@pytest.fixture()
def manager(deployer):
    return RollbackManager(deployer)


class TestRollbackManager:
    def test_create_checkpoint(self, manager: RollbackManager):
        cp = manager.create_checkpoint(
            "ml-serving", "ml-serving-00001", image_uri="gcr.io/p/ml:v1"
        )
        assert cp.service_name == "ml-serving"
        assert cp.revision == "ml-serving-00001"
        assert cp.image_uri == "gcr.io/p/ml:v1"

    def test_list_checkpoints(self, manager: RollbackManager):
        manager.create_checkpoint("svc", "rev-001")
        manager.create_checkpoint("svc", "rev-002")
        cps = manager.list_checkpoints("svc")
        assert len(cps) == 2
        assert cps[0].revision == "rev-001"
        assert cps[1].revision == "rev-002"

    def test_list_checkpoints_empty(self, manager: RollbackManager):
        assert manager.list_checkpoints("unknown") == []

    def test_rollback(self, manager: RollbackManager, mock_client):
        manager.create_checkpoint("svc", "rev-001", image_uri="gcr.io/p/ml:v1")
        info = manager.rollback("svc")
        assert info.service_name == "svc"
        mock_client.update_service.assert_called_once()

    def test_rollback_removes_checkpoint(self, manager: RollbackManager):
        manager.create_checkpoint("svc", "rev-001", image_uri="gcr.io/p/ml:v1")
        manager.rollback("svc")
        assert len(manager.list_checkpoints("svc")) == 0

    def test_rollback_no_checkpoints_raises(self, manager: RollbackManager):
        with pytest.raises(ValueError, match="No checkpoints"):
            manager.rollback("unknown")

    def test_rollback_without_image_uri(self, manager: RollbackManager):
        manager.create_checkpoint("svc", "rev-001")
        info = manager.rollback("svc")
        assert info.status == "ROLLED_BACK"
        assert info.revision == "rev-001"

    def test_multiple_rollbacks(self, manager: RollbackManager):
        manager.create_checkpoint("svc", "rev-001", image_uri="gcr.io/p/ml:v1")
        manager.create_checkpoint("svc", "rev-002", image_uri="gcr.io/p/ml:v2")

        # First rollback goes to rev-002
        info1 = manager.rollback("svc")
        assert len(manager.list_checkpoints("svc")) == 1

        # Second rollback goes to rev-001
        info2 = manager.rollback("svc")
        assert len(manager.list_checkpoints("svc")) == 0

    def test_auto_rollback_healthy(self, manager: RollbackManager):
        manager.create_checkpoint("svc", "rev-001", image_uri="gcr.io/p/ml:v1")

        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_http.get.return_value = mock_response

        result = manager.auto_rollback(
            "svc",
            "https://svc.a.run.app/health",
            threshold=3,
            http_client=mock_http,
        )
        assert result is True  # Healthy, no rollback

    def test_auto_rollback_unhealthy(self, manager: RollbackManager, mock_client):
        manager.create_checkpoint("svc", "rev-001", image_uri="gcr.io/p/ml:v1")

        mock_http = MagicMock()
        mock_http.get.side_effect = ConnectionError("refused")

        result = manager.auto_rollback(
            "svc",
            "https://svc.a.run.app/health",
            threshold=2,
            http_client=mock_http,
        )
        assert result is False  # Unhealthy, rolled back
        mock_client.update_service.assert_called_once()

    def test_auto_rollback_partial_failure(self, manager: RollbackManager):
        manager.create_checkpoint("svc", "rev-001", image_uri="gcr.io/p/ml:v1")

        mock_http = MagicMock()
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        mock_http.get.side_effect = [fail_resp, ok_resp]

        result = manager.auto_rollback(
            "svc",
            "https://svc.a.run.app/health",
            threshold=3,
            http_client=mock_http,
        )
        assert result is True  # Second attempt succeeded

    def test_clear_checkpoints(self, manager: RollbackManager):
        manager.create_checkpoint("svc", "rev-001")
        manager.create_checkpoint("svc", "rev-002")
        manager.clear_checkpoints("svc")
        assert manager.list_checkpoints("svc") == []

    def test_checkpoint_metadata(self, manager: RollbackManager):
        cp = manager.create_checkpoint(
            "svc", "rev-001",
            metadata={"deployed_by": "ci", "commit": "abc123"},
        )
        assert cp.metadata["deployed_by"] == "ci"
        assert cp.metadata["commit"] == "abc123"
