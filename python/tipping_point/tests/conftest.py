"""Shared fixtures. `no_network` refuses every connection that leaves the machine (loopback stays
open, as asyncio's self-pipe on Windows uses it); the market and book tests use it on every test,
so a broken refusal can never download real data (#30 M1's accident)."""

import socket
import urllib.request
from typing import Any

import pytest

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    connect = socket.socket.connect

    def refuse(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            raise AssertionError(f"a test reached the network: {address!r}")
        connect(self, address)

    def urlopen(request: Any, *args: Any, **kwargs: Any) -> None:
        url = getattr(request, "full_url", request)
        raise AssertionError(f"a test opened {url}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
