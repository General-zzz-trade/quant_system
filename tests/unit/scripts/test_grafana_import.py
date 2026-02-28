"""Tests for scripts/grafana_import.py — dashboard generation, export, and import."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.grafana_import import (
    export_to_file,
    generate_dashboard,
    import_to_grafana,
    main,
)


# ── generate_dashboard ──────────────────────────────────────


@patch("scripts.grafana_import.GrafanaDashboardGenerator")
@patch("scripts.grafana_import.DashboardConfig")
def test_generate_dashboard_default(mock_config_cls, mock_gen_cls):
    mock_gen = MagicMock()
    mock_gen.generate.return_value = {"dashboard": {"uid": "test"}}
    mock_gen_cls.return_value = mock_gen

    result = generate_dashboard()

    mock_config_cls.assert_called_once_with(
        title="Quant Trading System",
        refresh_interval="10s",
        time_range="24h",
        strategies=(),
        datasource="Prometheus",
        uid="quant-system-main",
    )
    mock_gen.generate.assert_called_once()
    assert result == {"dashboard": {"uid": "test"}}


@patch("scripts.grafana_import.GrafanaDashboardGenerator")
@patch("scripts.grafana_import.DashboardConfig")
def test_generate_dashboard_with_strategies(mock_config_cls, mock_gen_cls):
    mock_gen = MagicMock()
    mock_gen.generate.return_value = {"dashboard": {"panels": []}}
    mock_gen_cls.return_value = mock_gen

    result = generate_dashboard(strategies=("momentum", "mean_revert"))

    mock_config_cls.assert_called_once_with(
        title="Quant Trading System",
        refresh_interval="10s",
        time_range="24h",
        strategies=("momentum", "mean_revert"),
        datasource="Prometheus",
        uid="quant-system-main",
    )
    assert "dashboard" in result


# ── export_to_file ──────────────────────────────────────────


@patch("scripts.grafana_import.generate_dashboard")
def test_export_to_file_creates_json(mock_gen, tmp_path):
    mock_gen.return_value = {"dashboard": {"uid": "x", "panels": []}}

    out = tmp_path / "sub" / "dashboard.json"
    export_to_file(out)

    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["dashboard"]["uid"] == "x"


@patch("scripts.grafana_import.generate_dashboard")
def test_export_to_file_passes_strategies(mock_gen, tmp_path):
    mock_gen.return_value = {"dashboard": {}}

    out = tmp_path / "dash.json"
    export_to_file(out, strategies=("alpha",))

    mock_gen.assert_called_once_with(("alpha",))


@patch("scripts.grafana_import.generate_dashboard")
def test_export_to_file_pretty_printed(mock_gen, tmp_path):
    mock_gen.return_value = {"dashboard": {"k": "v"}}

    out = tmp_path / "dash.json"
    export_to_file(out)

    text = out.read_text(encoding="utf-8")
    assert "\n" in text  # indent=2 produces multi-line


# ── import_to_grafana ───────────────────────────────────────


@patch("scripts.grafana_import.generate_dashboard")
def test_import_to_grafana_success(mock_gen):
    mock_gen.return_value = {"dashboard": {"uid": "abc", "panels": []}}

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"url": "/d/abc"}).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        import_to_grafana("http://grafana:3000", "my-token")

    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "http://grafana:3000/api/dashboards/db"
    assert req.get_header("Authorization") == "Bearer my-token"
    assert req.get_header("Content-type") == "application/json"
    assert req.get_method() == "POST"

    payload = json.loads(req.data)
    assert payload["overwrite"] is True
    assert "dashboard" in payload


@patch("scripts.grafana_import.generate_dashboard")
def test_import_to_grafana_strips_trailing_slash(mock_gen):
    mock_gen.return_value = {"dashboard": {}}

    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"url": "ok"}'
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        import_to_grafana("http://grafana:3000/", "tok")

    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "http://grafana:3000/api/dashboards/db"


@patch("scripts.grafana_import.generate_dashboard")
def test_import_to_grafana_http_error(mock_gen):
    import urllib.error

    mock_gen.return_value = {"dashboard": {}}

    err = urllib.error.HTTPError(
        url="http://grafana:3000/api/dashboards/db",
        code=401,
        msg="Unauthorized",
        hdrs=MagicMock(),
        fp=MagicMock(read=MagicMock(return_value=b"bad token")),
    )

    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(SystemExit) as exc_info:
            import_to_grafana("http://grafana:3000", "bad")
        assert exc_info.value.code == 1


# ── main (CLI) ──────────────────────────────────────────────


@patch("scripts.grafana_import.export_to_file")
def test_main_export_flag(mock_export):
    with patch("sys.argv", ["grafana_import.py", "--export", "/tmp/out.json"]):
        main()
    mock_export.assert_called_once()
    assert mock_export.call_args[0][0] == Path("/tmp/out.json")


@patch("scripts.grafana_import.import_to_grafana")
def test_main_import_with_url_and_token(mock_import):
    with patch("sys.argv", ["grafana_import.py", "--url", "http://g:3000", "--token", "tok"]):
        main()
    mock_import.assert_called_once_with("http://g:3000", "tok", ())


@patch("scripts.grafana_import.export_to_file")
def test_main_default_export(mock_export):
    with patch("sys.argv", ["grafana_import.py"]):
        main()
    mock_export.assert_called_once()
    path = mock_export.call_args[0][0]
    assert path.name == "trading.json"
    assert "deploy" in str(path)


@patch("scripts.grafana_import.import_to_grafana")
def test_main_strategies_parsed(mock_import):
    with patch("sys.argv", ["grafana_import.py", "--url", "http://g:3000",
                             "--token", "t", "--strategies", "a, b , c"]):
        main()
    strategies = mock_import.call_args[0][2]
    assert strategies == ("a", "b", "c")


@patch("scripts.grafana_import.export_to_file")
def test_main_empty_strategies(mock_export):
    with patch("sys.argv", ["grafana_import.py", "--export", "/tmp/x.json", "--strategies", ""]):
        main()
    strategies = mock_export.call_args[0][1]
    assert strategies == ()
