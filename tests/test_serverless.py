"""Serverless /render (Vercel): Blob in, render, Blob out, one call. Blob is faked here."""
import sys, os, json, pathlib, shutil, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
os.environ.setdefault("RCV_JOBS_DIR", tempfile.mkdtemp())
os.environ["RCV_DOTENV"] = ""
os.environ.pop("MOTH_API_KEY", None)
import pytest
from fastapi.testclient import TestClient
import serve

C = TestClient(serve.app)
DEMO = pathlib.Path(__file__).parent.parent / "demo" / "source.mp4"
FAST = {"n_sites": 6, "depth": 4, "master_s": 0.4, "max_height": 270, "tail_s": 0.2}
BLOB = "https://abc123.public.blob.vercel-storage.com/"


@pytest.fixture
def blob(monkeypatch):
    store = {}                                            # url -> bytes

    def fetch(url, dest, limit_mb=200):
        if not serve.BLOB_URL.match(url):
            raise serve.EngineError("invalid_video", "files must come from this app's Blob store")
        dest.write_bytes(store[url] if url in store else DEMO.read_bytes()); return dest

    def put(path, data, content_type):
        url = BLOB + path.replace("/", "-"); store[url] = data; return url
    monkeypatch.setattr(serve, "_fetch_blob", fetch)
    monkeypatch.setattr(serve, "_blob_put", put)
    return store


def render(body):
    """POST /render and read the NDJSON stream: returns (progress events, final job or error)."""
    r = C.post("/render", json=body)
    if r.headers["content-type"].startswith("application/json"):
        return r, [], r.json()
    lines = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    return r, [l["progress"] for l in lines if "progress" in l], lines[-1]


def test_render_streams_progress_then_blob_urls(blob):
    r, prog, last = render({"params": FAST, "video_url": BLOB + "clips/a.mp4"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    j = last["job"]; assert j["status"] == "succeeded", last
    stages = {e["stage"] for e in prog}
    assert "rendering" in stages and any((e.get("frames_total") or 0) > 0 for e in prog)
    files = j["result"]["data"]["files"]
    assert set(files) == {"video", "ir", "taps", "spacetime"} and all(u.startswith(BLOB) for u in files.values())
    assert blob[files["video"]][4:8] == b"ftyp" and blob[files["spacetime"]][:4] == b"\x89PNG"
    assert j["result"]["data"]["ir_ref"] == files["ir"] and j["id"] not in serve.JOBS   # nothing kept server-side


def test_reuse_ir_url_and_cross_echo(blob):
    first = render({"params": FAST, "video_url": BLOB + "clips/a.mp4"})[2]["job"]
    r = render({"params": FAST, "video_url": BLOB + "clips/a.mp4", "echo_url": BLOB + "clips/b.mp4",
                "ir_url": first["result"]["data"]["ir_ref"]})[2]["job"]
    assert r["status"] == "succeeded" and r["ir_source"] == "reference" and r["cross_echo"], r


def test_inline_ir_upload(blob):
    ir = serve._measure(serve.Params(**FAST)).to_trajectory()
    j = render({"params": FAST, "video_url": BLOB + "clips/a.mp4", "ir": json.loads(json.dumps(ir))})[2]["job"]
    assert j["status"] == "succeeded" and j["ir_source"] == "upload"


def test_only_blob_urls_are_fetched():
    r = C.post("/render", json={"params": FAST, "video_url": "http://169.254.169.254/latest/meta-data"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_video"


def test_moth_needs_a_key(blob):
    r = C.post("/render", json={"params": FAST, "video_url": BLOB + "clips/a.mp4", "moth": {}})
    assert r.status_code == 401 and r.json()["error"]["code"] == "moth_unauthorized"


def test_bad_params_are_typed(blob):
    r = C.post("/render", json={"params": {"mix": 3}, "video_url": BLOB + "clips/a.mp4"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_params"


def test_time_budget_fails_loudly(blob, monkeypatch):
    monkeypatch.setattr(serve, "RENDER_BUDGET_S", 1e-6)                 # already past the deadline
    r, prog, last = render({"params": FAST, "video_url": BLOB + "clips/a.mp4"})
    assert last["error"]["code"] == "timeout" and "ran out of time" in last["error"]["message"], last


def test_render_failure_is_a_final_error_line(blob):
    r, prog, last = render({"params": {**FAST, "min_level": 1.0}, "video_url": BLOB + "clips/a.mp4"})
    assert last["error"]["code"] == "no_live_taps", last


def test_fps_cap_resamples(tmp_path):
    import videoio, subprocess
    src = tmp_path / "hi.mp4"
    subprocess.run([videoio.FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=60", "-t", "1",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    info, frames = videoio.read_frames(str(src), max_fps=30)
    assert info["fps"] == 30.0 and 28 <= sum(1 for _ in frames) <= 32


def test_inputs_are_deleted_after_render(blob, monkeypatch):
    gone = []
    monkeypatch.setattr(serve, "_blob_delete", lambda urls: gone.extend(urls) or len(urls))
    a, b = BLOB + "clips/a.mp4", BLOB + "clips/b.mp4"
    render({"params": FAST, "video_url": a, "echo_url": b})
    assert sorted(gone) == [a, b]
    gone.clear()
    render({"params": {**FAST, "min_level": 1.0}, "video_url": a})      # failed renders clean up too
    assert gone == [a]


def test_cleanup_is_cron_only_and_drops_old_blobs(monkeypatch):
    import datetime as dt, types
    now = dt.datetime.now(dt.timezone.utc)
    items = {"renders/": [types.SimpleNamespace(url=BLOB + "renders/old", uploaded_at=now - dt.timedelta(days=9)),
                          types.SimpleNamespace(url=BLOB + "renders/new", uploaded_at=now - dt.timedelta(days=1))],
             "clips/": [types.SimpleNamespace(url=BLOB + "clips/orphan", uploaded_at=now - dt.timedelta(hours=3))]}
    gone = []
    monkeypatch.setattr(serve, "_blob_list", lambda prefix: items[prefix])
    monkeypatch.setattr(serve, "_blob_delete", lambda urls: gone.extend(urls) or len(urls))
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    assert C.get("/cleanup").status_code == 401
    assert C.get("/cleanup", headers={"Authorization": "Bearer nope"}).status_code == 401
    r = C.get("/cleanup", headers={"Authorization": "Bearer s3cret"})
    assert r.json() == {"deleted": {"renders": 1, "clips": 1}} and sorted(gone) == [BLOB + "clips/orphan", BLOB + "renders/old"]
