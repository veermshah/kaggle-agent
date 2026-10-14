import json
import os
import threading
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from agents.base import build_context
from experiments import service


def test_vision_fold_trains_and_resumes(tmp_path):
    torch = pytest.importorskip("torch")
    from PIL import Image
    from models.vision import VisionConfig, train_fold

    rng = np.random.default_rng(0)
    paths, labels = [], []
    for i in range(48):
        label = i % 2
        base = np.zeros((32, 32, 3), dtype=np.uint8)
        base[..., 0 if label else 2] = 200  # red vs blue images
        img = np.clip(base + rng.integers(0, 40, base.shape), 0, 255).astype(np.uint8)
        p = tmp_path / f"img{i}.png"
        Image.fromarray(img).save(p)
        paths.append(p)
        labels.append(label)
    y = np.array(labels)
    cfg = VisionConfig(backbone="resnet18", num_classes=2, image_size=32, batch_size=8, epochs=3, lr=1e-3)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    out = train_fold(cfg, paths[:32], y[:32], paths[32:], y[32:], paths[32:], device=device,
                     ckpt_dir=tmp_path / "ckpt")
    assert out["val_proba"].shape == (16, 2) and out["test_proba"].shape == (16, 2)
    assert np.allclose(out["val_proba"].sum(axis=1), 1, atol=1e-3)
    assert ((out["val_proba"].argmax(1) == y[32:]).mean()) >= 0.75
    assert (tmp_path / "ckpt" / "last.pt").exists()
    # A finished fold resumes from its checkpoint instead of retraining.
    again = train_fold(cfg, paths[:32], y[:32], paths[32:], y[32:], paths[32:], device=device,
                       ckpt_dir=tmp_path / "ckpt")
    np.testing.assert_allclose(again["val_proba"], out["val_proba"], atol=1e-2)


def test_anthropic_agent_with_fake_client(make_comp, tracker):
    from agents.anthropic_agent import AnthropicAgent
    calls = {}

    class FakeMessages:
        def create(self, **kwargs):
            calls.update(kwargs)
            text = json.dumps({"proposals": [{"spec": {"model": "xgb", "params": {"max_depth": 4}},
                                              "hypothesis": "shallower trees regularize", "arm": "depth"}]})
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])

    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    ws = make_comp("binary")
    props = AnthropicAgent(client=client).propose(build_context(ws, tracker), 3)
    assert props[0].spec["model"] == "xgb" and props[0].arm == "claude:depth"
    assert calls["model"] == "claude-opus-5-5" and calls["fallbacks"] == "default"


def test_api_server_requires_token_and_routes(make_comp, tracker, monkeypatch):
    from agents.api_server import make_server
    ws = make_comp("binary")
    token = "x" * 32
    server = make_server("127.0.0.1", 0, token)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    def call(method, path, body=None, auth=True):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {token}"} if auth else {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        assert call("GET", "/health", auth=False)[0] == 200
        assert call("GET", "/status", auth=False)[0] == 401
        code, payload = call("GET", f"/status?competition={ws.slug}")
        assert code == 200 and payload["result"]["competition"] == ws.slug
        code, payload = call("POST", "/experiments", {"spec": {"model": "lgbm"}, "hypothesis": "remote idea"})
        assert code == 200 and payload["result"]["status"] == "queued"
        exp_id = payload["result"]["experiment_id"]
        assert tracker.get_experiment(exp_id)["proposed_by"] == "remote-agent"
        code, payload = call("POST", "/experiments", {"spec": {"model": "lgbm", "evil": 1}, "hypothesis": "x"})
        assert code == 400 and payload["ok"] is False
        assert call("POST", "/submissions/1/approve", {})[0] == 404  # no approval route exists
    finally:
        server.shutdown()


def test_api_server_refuses_weak_token():
    from agents.api_server import make_server
    with pytest.raises(RuntimeError):
        make_server("127.0.0.1", 0, "short")


def test_dashboard_renders(make_comp, tracker, monkeypatch):
    monkeypatch.setenv("KAGGLE_AGENT_DASHBOARD_REFRESH", "0")
    from streamlit.testing.v1 import AppTest
    ws = make_comp("binary")
    out = service.queue_experiment(ws, tracker, {"model": "lgbm"}, "baseline")
    service.execute(out["experiment_id"], tracker)
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Best CV" in m.label for m in at.metric)
