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


def test_render_in_one_call_returns_blob_urls(blob):
    r = C.post("/render", json={"params": FAST, "video_url": BLOB + "clips/a.mp4"})
    j = r.json(); assert r.status_code == 200 and j["status"] == "succeeded", j
    files = j["result"]["data"]["files"]
    assert set(files) == {"video", "ir", "taps", "spacetime"} and all(u.startswith(BLOB) for u in files.values())
    assert blob[files["video"]][4:8] == b"ftyp" and blob[files["spacetime"]][:4] == b"\x89PNG"
    assert j["result"]["data"]["ir_ref"] == files["ir"] and j["id"] not in serve.JOBS   # nothing kept server-side


def test_reuse_ir_url_and_cross_echo(blob):
    first = C.post("/render", json={"params": FAST, "video_url": BLOB + "clips/a.mp4"}).json()
    r = C.post("/render", json={"params": FAST, "video_url": BLOB + "clips/a.mp4", "echo_url": BLOB + "clips/b.mp4",
                                "ir_url": first["result"]["data"]["ir_ref"]}).json()
    assert r["status"] == "succeeded" and r["ir_source"] == "reference" and r["cross_echo"], r


def test_inline_ir_upload(blob):
    ir = serve._measure(serve.Params(**FAST)).to_trajectory()
    j = C.post("/render", json={"params": FAST, "video_url": BLOB + "clips/a.mp4", "ir": json.loads(json.dumps(ir))}).json()
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
