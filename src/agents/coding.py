"""Unit tests covering the cloud provider abstraction layer."""

from __future__ import annotations

import pytest

from src.providers import (
    PROVIDER_REGISTRY,
    UnknownProviderError,
    available_providers,
    get_provider,
)
from src.providers.base import (
    CloudProvider,
    ProvisionedResource,
    ResourceKind,
    ResourceSpec,
    ResourceStatus,
)


def test_registry_contains_all_major_providers() -> None:
    assert {"aws", "gcp", "azure", "digitalocean"}.issubset(set(PROVIDER_REGISTRY))


def test_available_providers_is_sorted() -> None:
    names = available_providers()
    assert names == sorted(names)
    assert "aws" in names


def test_get_provider_returns_correct_concrete_class() -> None:
    provider = get_provider("aws", region="us-east-1")
    assert isinstance(provider, CloudProvider)
    assert provider.name == "aws"
    assert provider.region == "us-east-1"


def test_get_provider_unknown_name_raises() -> None:
    with pytest.raises(UnknownProviderError):
        get_provider("alibaba")


def test_fake_provision_returns_active_resource(fake_provider: CloudProvider) -> None:
    spec = ResourceSpec(kind=ResourceKind.COMPUTE, name="svc-web")
    resource = fake_provider.provision(spec)

    assert isinstance(resource, ProvisionedResource)
    assert resource.status is ResourceStatus.ACTIVE
    assert resource.provider_id.startswith("fake-")
    assert fake_provider.get_status(resource.provider_id) is ResourceStatus.ACTIVE


def test_validate_spec_rejects_bad_names_and_replicas(fake_provider: CloudProvider) -> None:
    bad = ResourceSpec(kind=ResourceKind.COMPUTE, name="Bad_Name!", replicas=0)
    errors = fake_provider.validate_spec(bad)

    assert any("name" in error for error in errors)
    assert any("replicas" in error for error in errors)

    good = ResourceSpec(kind=ResourceKind.STORAGE, name="assets-bucket", replicas=1)
    assert fake_provider.validate_spec(good) == []


def test_destroy_flips_status_and_reports_missing(fake_provider: CloudProvider) -> None:
    spec = ResourceSpec(kind=ResourceKind.QUEUE, name="jobs-queue")
    resource = fake_provider.provision(spec)

    assert fake_provider.destroy(resource.provider_id) is True
    assert fake_provider.get_status(resource.provider_id) is ResourceStatus.DESTROYED
    assert fake_provider.destroy("fake-does-not-exist") is False


def test_unsupported_kind_flagged_by_validator() -> None:
    class ComputeOnly(CloudProvider):
        name = "compute-only"

        @property
        def supported_kinds(self) -> frozenset[ResourceKind]:
            return frozenset({ResourceKind.COMPUTE})

        def provision(self, spec: ResourceSpec) -> ProvisionedResource:  # pragma: no cover
            raise NotImplementedError

        def destroy(self, provider_id: str) -> bool:  # pragma: no cover
            return False

        def get_status(self, provider_id: str) -> ResourceStatus:  # pragma: no cover
            return ResourceStatus.FAILED

    provider = ComputeOnly()
    errors = provider.validate_spec(ResourceSpec(kind=ResourceKind.CDN, name="edge"))
    assert any("does not support kind" in error for error in errors)
