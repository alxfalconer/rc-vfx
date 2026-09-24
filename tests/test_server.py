import sys, os, json, pathlib, time, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
os.environ["RCV_JOBS_DIR"] = tempfile.mkdtemp()
os.environ["RCV_DOTENV"] = ""            # never pick up a real key in tests
os.environ.pop("MOTH_API_KEY", None)
from fastapi.testclient import TestClient
import serve

C = TestClient(serve.app)
CLIP = pathlib.Path(__file__).parent.parent / "demo" / "source.mp4"
FAST = {"n_sites": 6, "depth": 4, "master_s": 0.4, "max_height": 270, "tail_s": 0.2}


def wait(jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = C.get(f"/jobs/{jid}").json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.2)
    raise TimeoutError


def submit(params=None, **extra):
    files = {"video": ("clip.mp4", CLIP.read_bytes(), "video/mp4")}
    if "ir" in extra:
        files["ir"] = ("ir.json", extra.pop("ir"), "application/json")
    return C.post("/process", files=files, data={"params": json.dumps(params or FAST), **extra})


def test_engine_record():
    e = C.get("/engine").json()
    assert e["id"] == "retrocausal-echo-video-v1" and "params_schema" in e
    assert {c["code"] for c in e["error_codes"]} >= {"invalid_params", "invalid_ir", "no_live_taps"}

def test_validate_rejects_bad_params():
    r = C.post("/validate", json={"mix": 2})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_params"
    assert C.post("/validate", json={"nope": 1}).status_code == 422         # extra forbidden
    assert C.post("/validate", json={"n_sites": 4, "kick_site": 9}).status_code == 422

def test_estimate():
    j = C.post("/estimate", json=FAST).json()
    assert j["credits"] == 2 and j["taps"] > 0

def test_process_end_to_end_and_chain_ir():
    r = submit(); assert r.status_code == 202
    j = wait(r.json()["id"]); assert j["status"] == "succeeded", j
    files = j["result"]["data"]["files"]
    v = C.get(files["video"]); assert v.status_code == 200 and v.content[4:8] == b"ftyp"
    assert C.get(files["spacetime"]).content[:4] == b"\x89PNG"
    ir = C.get(files["ir"]).json(); assert ir["result_type"] == "trajectory"
    assert j["credits"] == 2
    # reuse by reference: no measurement, 1 credit, identical tap map
    j2 = wait(submit(ir_ref=j["result"]["data"]["ir_ref"]).json()["id"])
    assert j2["status"] == "succeeded" and j2["credits"] == 1 and j2["ir_source"] == "reference"
    assert C.get(j2["result"]["data"]["files"]["ir"]).json()["data"] == ir["data"]
    # and by upload
    j3 = wait(submit(ir=json.dumps(ir)).json()["id"])
    assert j3["status"] == "succeeded" and j3["ir_source"] == "upload"

def test_bad_inputs_are_typed():
    assert submit(ir_ref="job:nothere/ir").json()["error"]["code"] == "invalid_ir"
    assert submit(ir=b"{not json").json()["error"]["code"] == "invalid_ir"
    r = C.post("/process", files={"video": ("x.mp4", b"not a video", "video/mp4")})
    assert r.json()["error"]["code"] == "invalid_video"

def test_no_live_taps_fails_cleanly():
    j = wait(submit({**FAST, "min_level": 1.0}).json()["id"])
    assert j["status"] == "failed" and j["error"]["code"] == "no_live_taps"

def test_ui_served():
    assert "Retrocausal Echo" in C.get("/").text


def test_cross_echo_job():
    files = {"video": ("a.mp4", CLIP.read_bytes(), "video/mp4"), "echo": ("b.mp4", CLIP.read_bytes(), "video/mp4")}
    r = C.post("/process", files=files, data={"params": json.dumps(FAST)}); assert r.status_code == 202, r.text
    j = wait(r.json()["id"]); assert j["status"] == "succeeded", j
    assert j["cross_echo"] and j["result"]["data"]["cross_echo"]
    bad = C.post("/process", files={**files, "echo": ("b.mp4", b"nope", "video/mp4")}, data={"params": json.dumps(FAST)})
    assert bad.json()["error"]["code"] == "invalid_video"
