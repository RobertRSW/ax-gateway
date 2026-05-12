import io
import json

import pytest

from ax_cli.commands import gateway as gateway_cmd


def _make_handler(host_header, *, method="GET", path="/healthz", body=b""):
    handler_cls = gateway_cmd._build_gateway_ui_handler(activity_limit=5, refresh_ms=1000)

    class FakeHandler(handler_cls):
        def __init__(self):
            self.path = path
            headers = {}
            if host_header is not None:
                headers["Host"] = host_header
            if method == "POST":
                headers["Content-Length"] = str(len(body))
            self.headers = headers
            self.rfile = io.BytesIO(body)
            self.status = None
            self.body = b""

        def send_response(self, status):
            self.status = status

        def send_header(self, *args, **kwargs):
            return None

        def end_headers(self):
            return None

        @property
        def wfile(self):
            outer = self

            class Writer:
                def write(self, data):
                    outer.body += data

            return Writer()

    return FakeHandler()


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.1:8765", "localhost", "localhost:9000", "LocalHost:8765"],
)
def test_loopback_host_passes(host):
    assert gateway_cmd._is_request_host_allowed(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "evil.example.com",
        "evil.example.com:8765",
        "192.168.1.1",
        "127.0.0.1.evil.com",
        "",
        None,
        "  ",
    ],
)
def test_non_loopback_host_blocked(host):
    assert gateway_cmd._is_request_host_allowed(host) is False


def test_get_with_loopback_host_is_served():
    handler = _make_handler("127.0.0.1:8765", method="GET", path="/healthz")
    handler.do_GET()

    assert handler.status == 200
    assert json.loads(handler.body.decode("utf-8")) == {"ok": True}


def test_get_with_evil_host_is_rejected_403():
    handler = _make_handler("evil.example.com", method="GET", path="/healthz")
    handler.do_GET()

    assert handler.status == 403
    payload = json.loads(handler.body.decode("utf-8"))
    assert "Host" in payload["error"]


def test_get_without_host_header_is_rejected_403():
    handler = _make_handler(None, method="GET", path="/healthz")
    handler.do_GET()

    assert handler.status == 403


def test_post_with_evil_host_short_circuits_before_handler(tmp_path, monkeypatch):
    # Wire AX_CONFIG_DIR so any code paths that read registry don't blow up;
    # the rejection should fire before that code runs anyway.
    monkeypatch.setenv("AX_CONFIG_DIR", str(tmp_path))
    handler = _make_handler("evil.example.com", method="POST", path="/api/agents", body=b"{}")
    handler.do_POST()

    assert handler.status == 403


def test_post_with_loopback_host_reaches_handler(tmp_path, monkeypatch):
    monkeypatch.setenv("AX_CONFIG_DIR", str(tmp_path))
    # Send a request body that the route will accept far enough to NOT be a 403.
    # /api/agents requires a name; an empty body reaches the handler and returns
    # a 4xx other than 403, proving the Host check did not short-circuit.
    handler = _make_handler("127.0.0.1:8765", method="POST", path="/api/agents", body=b"{}")
    handler.do_POST()

    assert handler.status != 403
