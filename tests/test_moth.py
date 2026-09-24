"""Bring-your-own-key Moth path, against a mock of moth-api (no network)."""
import sys, os, json, pathlib, time, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
os.environ.setdefault("RCV_JOBS_DIR", tempfile.mkdtemp())
os.environ["RCV_DOTENV"] = ""
os.environ.pop("MOTH_API_KEY", None)
import httpx, pytest
from fastapi.testclient import TestClient
import moth, serve, sim

C = TestClient(serve.app)
CLIP = pathlib.Path(__file__).parent.parent / "demo" / "source.mp4"
GOOD = "moth_test_good_key"
FAST = {"n_sites": 6, "depth": 4, "master_s": 0.4, "max_height": 270, "tail_s": 0.2}
SCHEMA = {"type": "object", "properties": {
    "n_sites": {"type": "integer"}, "depth": {"type": "integer"}, "theta_zz": {"type": "number"},
    "theta_x": {"type": "number"}, "theta_z": {"type": "number"}, "kick": {"type": "string"},
    "machine": {"type": "string", "enum": ["aer", "fake_fez"], "default": "aer"},
    "shots": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096},
    "lattice": {"type": "string"}, "width": {}, "height": {}, "via": {"type": "string", "enum": ["direct", "mothbackend"]}}}


class FakeMoth:
    def __init__(self, finish="inline"):
        self.finish, self.calls, self.submitted, self.polls = finish, [], None, 0
        self.traj = sim.measure_ir(n_sites=6, depth=4).to_trajectory()

    def __call__(self, req: httpx.Request):
        self.calls.append((req.method, req.url.path, req.headers.get("authorization")))
        if req.url.host == "s3.example":
            return httpx.Response(200, json=self.traj)
        if req.headers.get("authorization") != f"Bearer {GOOD}":
            return httpx.Response(401, json={"title": "Unauthorized", "status": 401, "detail": "authentication required"})
        path = req.url.path.removeprefix("/api/v1")
        if path == "/me":
            return httpx.Response(200, json={"id": "u1", "email": "a@example.com", "platform_role": "player"})
        if path == "/engines/otoc-echo-v1":
            return httpx.Response(200, json={"id": "otoc-echo-v1", "params_schema": SCHEMA})
        if path == "/engines/otoc-echo-v1/process":
            self.submitted = json.loads(req.content)
            return httpx.Response(202, json={"job_id": "mj1", "status": "queued"})
        if path == "/jobs/mj1/status":
            self.polls += 1
            if self.polls < 2:
                return httpx.Response(200, json={"job_id": "mj1", "status": "processing", "progress": 0.3})
            if self.finish == "fail":
                return httpx.Response(200, json={"job_id": "mj1", "status": "failed",
                                                 "error": {"type": "processing_failed", "message": "qpu offline", "retryable": False}})
            return httpx.Response(200, json={"job_id": "mj1", "status": "completed"})
        if path == "/jobs/mj1/result":
            if self.finish == "nested":                       # the real moth-api shape
                t = self.traj
                return httpx.Response(200, json={"$schema": "x", "result": {"output": {
                    "data": t["data"], "extras": {**t["extras"], "taps": [{}] * 3, "params": {}, "spec": {}},
                    "provenance": {"backend": "aer", "engine_id": "otoc-echo-v1"}, "result_type": "trajectory"}}})
            if self.finish == "outputs":
                return httpx.Response(200, json={"outputs": [{"slot": "trajectory", "url": "https://s3.example/t.json",
                                                              "content_type": "application/json", "filename": "t.json"}]})
            return httpx.Response(200, json={"result": self.traj})
        return httpx.Response(404, json={"title": "Not Found", "status": 404})


@pytest.fixture
def fake(monkeypatch):
    f = FakeMoth()
    monkeypatch.setattr(moth, "TRANSPORT", httpx.MockTransport(f))
    monkeypatch.setattr(moth, "POLL_S", 0.0)
    moth.clear()
    yield f
    moth.clear()


def wait(jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = C.get(f"/jobs/{jid}").json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.2)
    raise TimeoutError


def submit(moth_opts):
    return C.post("/process", files={"video": ("clip.mp4", CLIP.read_bytes(), "video/mp4")},
                  data={"params": json.dumps(FAST), "moth": json.dumps(moth_opts)})


def test_starts_off_and_rejects_bad_keys(fake):
    assert C.get("/moth").json()["state"] == "off"
    assert C.post("/moth/key", json={"key": "sk-nope"}).status_code == 422          # wrong prefix, no call made
    r = C.post("/moth/key", json={"key": "moth_revoked"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "moth_unauthorized"
    assert C.get("/moth").json()["state"] == "bad"


def test_key_applies_and_is_never_echoed(fake):
    r = C.post("/moth/key", json={"key": GOOD})
    assert r.status_code == 200 and r.json()["state"] == "on" and r.json()["account"] == "a@example.com"
    for path in ("/moth", "/moth/options", "/"):
        assert GOOD not in C.get(path).text
    assert C.delete("/moth/key").json()["state"] == "off"


def test_options_come_from_engine_schema(fake):
    C.post("/moth/key", json={"key": GOOD})
    o = C.get("/moth/options").json()
    assert o["machine"]["enum"] == ["aer", "fake_fez"] and o["shots"]["default"] == 4096


def test_moth_measure_then_render(fake):
    C.post("/moth/key", json={"key": GOOD})
    r = submit({"machine": "fake_fez", "shots": 2000}); assert r.status_code == 202, r.text
    j = wait(r.json()["id"]); assert j["status"] == "succeeded", j
    assert j["ir_source"] == "moth" and j["result"]["data"]["machine"] == "fake_fez"
    sent = fake.submitted["params"]                       # only schema-declared fields go out
    assert sent["machine"] == "fake_fez" and sent["shots"] == 2000 and sent["n_sites"] == 6
    assert "seed" not in sent and "shape" not in sent and sent["via"] == "mothbackend"
    ir = C.get(j["result"]["data"]["files"]["ir"]).json()
    assert ir["extras"]["moth_job"] == "mj1" and ir["data"]["series"] == fake.traj["data"]["series"]


def test_build_params_maps_square_and_aer():
    p = moth.build_params({"lattice": "square", "shape": [3, 3], "n_sites": 12, "depth": 4}, {"machine": "aer"}, SCHEMA)
    assert p == {"lattice": "square", "width": 3, "height": 3, "depth": 4, "machine": "aer"}   # aer: no via


def test_presigned_output_result(fake):
    fake.finish = "outputs"
    C.post("/moth/key", json={"key": GOOD})
    j = wait(submit({}).json()["id"]); assert j["status"] == "succeeded", j
    assert ("GET", "/t.json", None) in fake.calls           # presigned url fetched without the key


def test_real_nested_result_shape(fake):
    fake.finish = "nested"
    C.post("/moth/key", json={"key": GOOD})
    j = wait(submit({}).json()["id"]); assert j["status"] == "succeeded", j
    ir = C.get(j["result"]["data"]["files"]["ir"]).json()
    assert ir["data"]["series"]["F_re"] == fake.traj["data"]["series"]["F_re"]
    assert ir["extras"]["provenance"]["backend"] == "aer" and "taps" not in ir["extras"]


def test_unwrap_square_width_height():
    env = moth.unwrap({"result": {"output": {"data": {"sites": 9}, "extras": {"lattice": "square", "width": 3, "height": 3}}}})
    assert env["extras"]["shape"] == [3, 3]


def test_moth_job_failure_is_typed(fake):
    fake.finish = "fail"
    C.post("/moth/key", json={"key": GOOD})
    j = wait(submit({}).json()["id"])
    assert j["status"] == "failed" and j["error"]["code"] == "moth_failed" and "qpu offline" in j["error"]["message"]


def test_moth_source_needs_a_key(fake):
    r = submit({})
    assert r.status_code == 401 and r.json()["error"]["code"] == "moth_unauthorized"
