"""Fixtures for the integration tier.

Integration tests drive the real FastAPI application through its public HTTP
and WebSocket surface. They must never touch a real model, a real GPU, the
network or the user's data root -- the whole point of the fake providers is
that an end-to-end turn is reproducible on a bare CI machine (RULE 11).

Two deliberate pieces of isolation:

* ``host_config`` neuters the legacy block, so the provider registry contains
  only the fake providers and selection cannot pick Voicebox/Ollama just
  because the developer's own ``config.json`` happens to enable them.
* ``bots_dir`` points at ``tmp_path``, so creating a bot never writes into the
  real data root.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def host_config(tmp_path):
    """A hermetic SystemConfig: no legacy backends.

    ``legacy.enabled = False`` keeps the desktop Voicebox/Ollama adapters out of
    the candidate set, so these tests exercise the fake providers regardless of
    what the developer's own ``config.json`` contains.
    """

    from local_voice_companion.config.schema import SystemConfig

    config = SystemConfig()
    config.legacy.enabled = False
    config.legacy.voicebox_url = ""
    config.legacy.ollama_url = ""
    return config


@pytest.fixture
def app(host_config, tmp_path, isolated_registry):
    """A FastAPI app wired to fake providers only.

    The isolated registry is passed explicitly: the app must register into the
    registry it was handed, not into the module-level singleton, or two apps in
    one process would share a provider set.
    """

    from local_voice_companion.api.app import create_app

    return create_app(
        config=host_config,
        registry=isolated_registry,
        bots_dir=tmp_path / "bots",
        register_default_providers=False,
    )


@pytest.fixture
def client(app):
    """Synchronous HTTP client. WebSocket tests use ``client.websocket_connect``."""

    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def bot_id(client) -> str:
    """A bot created through the public API, ready for sessions."""

    response = client.post(
        "/api/v1/bots",
        json={"id": "itest-bot", "name": "Integration Bot", "language": "zh"},
    )
    assert response.status_code == 201, response.text
    return response.json()["bot"]["id"]


@pytest.fixture
def session_id(client, bot_id) -> str:
    response = client.post("/api/v1/sessions", json={"bot_id": bot_id})
    assert response.status_code == 201, response.text
    return response.json()["session"]["id"]
