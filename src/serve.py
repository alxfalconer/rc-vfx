"""Local dev server for retrocausal-echo-video-v1.

Mirrors the platform contract (/engine /validate /estimate /process) with moth-api-style
job polling, plus a browser UI at /.  Run:  ./run.sh   ->  http://127.0.0.1:8000

The physics path uses the local exact sim (sim.py) as a stand-in for measure_ir; supply an
`ir` file (an otoc-echo-v1 trajectory) or `job:<id>/ir` to render hardware-measured F instead.
"""
from __future__ import annotations
import json, math, os, pathlib, re, shutil, tempfile, threading, time, traceback, uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
import httpx
from pydantic import BaseModel, ValidationError

import moth
import sim
from diagnostics import impulse_spacetime_png
from echo_ir import EchoIR
from params import ERROR_CODES, MAX_INPUT_S, MAX_UPLOAD_MB, Params
from render import RenderParams, VideoEcho
from videoio import Writer, probe, read_frames

ENGINE_ID, VERSION = "retrocausal-echo-video-v1", "0.1.0-dev"
_DEFAULT_ROOT = pathlib.Path("/tmp/rcv-jobs") if os.environ.get("VERCEL") else pathlib.Path(__file__).resolve().parent.parent / "jobs"
ROOT = pathlib.Path(os.environ.get("RCV_JOBS_DIR", _DEFAULT_ROOT))           # Vercel: only /tmp is writable
ROOT.mkdir(parents=True, exist_ok=True)
STATIC = pathlib.Path(__file__).resolve().parent / "static"
moth.load_env_file(os.environ.get("RCV_DOTENV", pathlib.Path(__file__).resolve().parent.parent / ".env"))   # MOTH_API_KEY, optional
moth.init_from_env()

app = FastAPI(title="Retrocausal Echo — Video (dev)", version=VERSION)
# public deployment (RCV_PUBLIC=1): the UI lives on another origin (Vercel) and calls this API directly
if os.environ.get("RCV_CORS_ORIGINS"):
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in os.environ["RCV_CORS_ORIGINS"].split(",") if o.strip()],
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-Moth-Key", "X-Client-Id"])
JID = re.compile(r"^[0-9a-f]{12,32}$")
PRIVATE = ("params", "client")                 # never returned: request params and the owning visitor
POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("RCV_WORKERS", "1")))   # renders are CPU-bound
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


class EngineError(Exception):
    def __init__(self, code, message, status=422):
        super().__init__(message); self.code, self.message, self.status = code, message, status


@app.exception_handler(EngineError)
async def _engine_error(_, e: EngineError):
    return JSONResponse(status_code=e.status, content={"error": {"code": e.code, "message": e.message}})


@app.exception_handler(moth.MothError)
async def _moth_error(_, e: moth.MothError):
    return JSONResponse(status_code=e.status, content={"error": {"code": e.code, "message": e.message}})


# ---------------------------------------------------------------- moth key (bring your own)
# The visitor's key travels in X-Moth-Key on each request and is never stored here; only a hash
# of it is cached briefly to avoid re-checking /me. MOTH_API_KEY (.env) is a local-dev fallback.
@app.get("/moth")
def moth_status(x_moth_key: str | None = Header(None)):
    return moth.status(x_moth_key)


@app.post("/moth/key")
def moth_key(key: str = Body(..., embed=True)):
    moth.validate(key)                        # raises 401/422 for a bad key
    return moth.status(key)


@app.delete("/moth/key")
def moth_clear():
    return moth.clear() if not moth.public() else moth.status()


@app.get("/moth/options")
def moth_options(x_moth_key: str | None = Header(None)):
    return moth.options(x_moth_key)


@app.get("/config.js")
def config_js():
    """Local dev: the UI talks to this same origin. The Vercel build writes its own config.js."""
    return Response('window.RCV_API = window.RCV_API || "";\nwindow.RCV_MODE = window.RCV_MODE || "server";\n',
                    media_type="application/javascript")


def _parse_params(raw: str | dict | None) -> Params:
    try:
        data = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
        return Params(**data)
    except (ValidationError, ValueError, TypeError) as e:
        msg = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()) \
            if isinstance(e, ValidationError) else str(e)
        raise EngineError("invalid_params", msg)


# ---------------------------------------------------------------- engine record
@app.get("/engine")
def engine():
    return {
        "id": ENGINE_ID, "version": VERSION, "name": "Retrocausal Echo — Video",
        "tagline": "A multi-tap frame delay whose tap map is a measured OTOC.",
        "execution": {"mode": "handler", "local_physics": "exact statevector stand-in for measure_ir"},
        "data": "multipart/form-data -> video/mp4",
        "input": {"files": [{"name": "video", "required": True, "max_mb": MAX_UPLOAD_MB},
                            {"name": "ir", "required": False, "by_reference": True,
                             "accepts": "trajectory JSON, or job:<id>/ir"}],
                  "limits": {"max_input_s": MAX_INPUT_S, "max_height": 720}},
        "output": {"files": ["video", "ir", "taps", "spacetime"], "result_type": "media"},
        "pricing": {"credits_per_run": 2, "credits_if_ir_supplied": 1},
        "params_schema": Params.model_json_schema(),
        "error_codes": [{"code": k, "message": v} for k, v in ERROR_CODES.items()],
    }


@app.post("/validate")
def validate(params: dict | None = None):
    p = _parse_params(params)
    return {"ok": True, "params": p.model_dump()}


@app.post("/estimate")
def estimate(params: dict | None = None, input_seconds: float = 6.0, fps: float = 30.0):
    p = _parse_params(params)
    ir = _measure(p)                    # cheap locally; tells us tap count and tail honestly
    ve = VideoEcho(ir, fps, (8, 64), RenderParams(**p.render_kwargs()))
    frames = int(min(input_seconds, MAX_INPUT_S) * fps) + ve.tail_frames(0)
    return {"credits": 2, "resources": {"credits_if_ir_supplied": 1},
            "taps": len(ve.taps), "output_frames": frames, "output_seconds": frames / fps,
            "warnings": ve.warnings}


# ---------------------------------------------------------------- jobs
def _measure(p: Params) -> EchoIR:
    return sim.measure_ir(**p.sim_kwargs())


def _owns(jid: str, client: str | None) -> bool:
    if not moth.public():
        return True
    with LOCK:
        return jid in JOBS and JOBS[jid].get("client") == (client or None)


def _resolve_ir(ir_bytes: bytes | None, ir_ref: str | None, client: str | None = None) -> EchoIR | None:
    if ir_ref:
        if not ir_ref.startswith("job:") or not ir_ref.endswith("/ir"):
            raise EngineError("invalid_ir", "reference must look like job:<id>/ir")
        jid = ir_ref[4:-3]
        path = ROOT / jid / "ir.json"
        if not JID.match(jid) or not _owns(jid, client) or not path.exists():
            raise EngineError("invalid_ir", f"no ir output for job {jid}")
        ir_bytes = path.read_bytes()
    if not ir_bytes:
        return None
    try:
        ir = EchoIR.from_trajectory(ir_bytes)
        assert ir.F.ndim == 2 and ir.F.size > 0
        return ir
    except Exception as e:
        raise EngineError("invalid_ir", f"could not read trajectory: {e}")


def _set(jid, **kw):
    with LOCK:
        JOBS[jid].update(kw)


def _run(jid: str, p: Params, video_path: pathlib.Path, ir: EchoIR | None, moth_opts: dict | None = None,
         echo_path: pathlib.Path | None = None, moth_key: str | None = None, d: pathlib.Path | None = None):
    d = d or ROOT / jid
    try:
        _set(jid, status="running", stage="measuring" if ir is None else "loading ir")
        ir_supplied = ir is not None
        if ir is None and moth_opts is not None:
            ir = moth.measure(p.sim_kwargs(), moth_opts,
                              lambda st, prog: _set(jid, stage=f"moth · {st}", progress=0.0), key=moth_key)
        elif ir is None:
            ir = _measure(p)
        info, frames = read_frames(str(video_path), max_height=p.max_height, max_seconds=MAX_INPUT_S)
        echo_frames, echo_s = None, 0.0
        if echo_path is not None:          # cross-echo: B feeds the taps, fitted to A's frame
            einfo, echo_frames = read_frames(str(echo_path), max_seconds=MAX_INPUT_S,
                                             size=(info["width"], info["height"]), fps=info["fps"])
            echo_s = min(einfo["duration"] or MAX_INPUT_S, MAX_INPUT_S)
        rp = RenderParams(**p.render_kwargs())
        try:
            ve = VideoEcho(ir, info["fps"], (info["height"], info["width"]), rp)
        except ValueError as e:
            raise EngineError("no_live_taps" if "no live taps" in str(e) else "invalid_params", str(e))
        n_in = int(max(min(info["duration"] or MAX_INPUT_S, MAX_INPUT_S), echo_s) * info["fps"])
        total = max(1, n_in + ve.tail_frames(n_in))
        w = Writer(str(d / "video.mp4"), info["width"], info["height"], info["fps"])
        _set(jid, stage="rendering", frames_total=total, taps=len(ve.taps), warnings=ve.warnings)
        t0 = time.time()
        for i, f in enumerate(ve.process(frames, echo_frames)):
            w.write(f)
            if i % 5 == 0:
                _set(jid, frames_done=i + 1, progress=min(0.99, (i + 1) / total))
        n_out = w.close()
        if n_out == 0:
            raise EngineError("invalid_video", "no frames decoded")
        (d / "ir.json").write_text(json.dumps(ir.to_trajectory()))
        (d / "taps.json").write_text(json.dumps(ve.tap_map(), indent=1))
        (d / "spacetime.png").write_bytes(impulse_spacetime_png(ir, rp, info["fps"]))
        warnings = list(ve.warnings)
        if (info["duration"] or 0) > MAX_INPUT_S:
            warnings.append(f"input trimmed to {MAX_INPUT_S}s")
        _set(jid, status="succeeded", stage="done", progress=1.0, frames_done=n_out,
             render_s=round(time.time() - t0, 2), warnings=warnings,
             credits=1 if ir_supplied else 2,
             result={"result_type": "media",
                     "data": {"files": {k: f"/jobs/{jid}/files/{k}" for k in ("video", "ir", "taps", "spacetime")},
                              "video": {"width": info["width"], "height": info["height"], "fps": info["fps"],
                                        "frames": n_out, "seconds": round(n_out / info["fps"], 3)},
                              "ir_ref": f"job:{jid}/ir", "machine": ir.meta.get("machine", "supplied"),
                              "cross_echo": echo_path is not None}})
    except (EngineError, moth.MothError) as e:
        _set(jid, status="failed", error={"code": e.code, "message": e.message})
    except Exception as e:
        traceback.print_exc()
        _set(jid, status="failed", error={"code": "render_failed", "message": f"{type(e).__name__}: {e}"})
    finally:
        video_path.unlink(missing_ok=True)
        if echo_path is not None:
            echo_path.unlink(missing_ok=True)


async def _save_upload(up: UploadFile, dest: pathlib.Path, d: pathlib.Path):
    size = 0
    with open(dest, "wb") as fh:
        while chunk := await up.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                fh.close(); shutil.rmtree(d)
                raise EngineError("invalid_video", f"upload exceeds {MAX_UPLOAD_MB} MB", 413)
            fh.write(chunk)
    try:
        probe(str(dest))
    except Exception:
        shutil.rmtree(d)
        raise EngineError("invalid_video", f"ffprobe could not read a video stream in {up.filename or 'upload'}")


@app.post("/process", status_code=202)
async def process(video: UploadFile = File(...), ir: UploadFile | None = File(None),
                  echo: UploadFile | None = File(None),
                  ir_ref: str | None = Form(None), params: str | None = Form(None),
                  moth_opts: str | None = Form(None, alias="moth"),
                  x_moth_key: str | None = Header(None), x_client_id: str | None = Header(None)):
    p = _parse_params(params)
    if moth.public():
        _sweep()
    ir_obj = _resolve_ir(await ir.read() if ir else None, ir_ref, x_client_id)
    mo = None
    if moth_opts is not None and ir_obj is None:
        try:
            mo = json.loads(moth_opts) if moth_opts.strip() else {}
            assert isinstance(mo, dict)
        except (ValueError, AssertionError):
            raise EngineError("invalid_params", "moth must be a JSON object")
        if not moth.resolve(x_moth_key):
            raise EngineError("moth_unauthorized", "No Moth key applied; add one with the key button.", 401)
    jid = uuid.uuid4().hex[:32 if moth.public() else 12]
    d = ROOT / jid; d.mkdir(parents=True)
    src = d / ("input" + pathlib.Path(video.filename or "in.mp4").suffix)
    await _save_upload(video, src, d)
    echo_src = None
    if echo is not None and echo.filename:
        echo_src = d / ("echo" + pathlib.Path(echo.filename).suffix)
        await _save_upload(echo, echo_src, d)
    with LOCK:
        JOBS[jid] = {"id": jid, "status": "queued", "stage": "queued", "progress": 0.0,
                     "created": time.time(), "params": p.model_dump(),
                     "ir_source": "reference" if ir_ref else ("upload" if ir else ("moth" if mo is not None else "measured")),
                     "cross_echo": echo_src is not None, "client": x_client_id or None}
    POOL.submit(_run, jid, p, src, ir_obj, mo, echo_src, moth.resolve(x_moth_key) if mo is not None else None)
    return {"id": jid, "status": "queued", "poll": f"/jobs/{jid}"}


def _sweep():
    """Public mode: forget finished jobs (and their files) after RCV_JOB_TTL_H hours."""
    ttl, now = float(os.environ.get("RCV_JOB_TTL_H", 24)) * 3600, time.time()
    with LOCK:
        old = [j for j, v in JOBS.items() if now - v["created"] > ttl and v["status"] in ("succeeded", "failed")]
        for j in old:
            JOBS.pop(j, None)
    for j in old:
        shutil.rmtree(ROOT / j, ignore_errors=True)


# ---------------------------------------------------------------- serverless (Vercel)
# One call does the whole job: fetch the clips the browser uploaded to Vercel Blob, render in /tmp,
# put the outputs back in Blob, return the finished job. No server-side job state survives the call;
# the browser keeps its own job list.
BLOB_URL = re.compile(r"^https://[a-z0-9]+\.(public|private)\.blob\.vercel-storage\.com/[^\s]+$")


class RenderRequest(BaseModel):
    params: dict = {}
    video_url: str
    echo_url: str | None = None
    ir_url: str | None = None          # a previous job's ir.json in Blob
    ir: dict | None = None             # an uploaded trajectory, inline
    moth: dict | None = None           # present = measure on Moth with the visitor's key


def _fetch_blob(url: str, dest: pathlib.Path, limit_mb: int = MAX_UPLOAD_MB) -> pathlib.Path:
    if not BLOB_URL.match(url or ""):                   # never fetch arbitrary URLs (SSRF)
        raise EngineError("invalid_video", "files must come from this app's Blob store")
    size = 0
    with httpx.stream("GET", url, timeout=60, follow_redirects=False) as r:
        if r.status_code != 200:
            raise EngineError("invalid_video", f"could not fetch {pathlib.Path(url).name} ({r.status_code})")
        with open(dest, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                size += len(chunk)
                if size > limit_mb << 20:
                    raise EngineError("invalid_video", f"file exceeds {limit_mb} MB", 413)
                fh.write(chunk)
    return dest


def _blob_put(path: str, data: bytes, content_type: str) -> str:
    from vercel.blob import put                        # Vercel runtime only (needs Python >= 3.10)
    return put(path, data, access="public", content_type=content_type, add_random_suffix=True,
               multipart=len(data) > (50 << 20)).url


@app.post("/render")
def render_now(req: RenderRequest, x_moth_key: str | None = Header(None)):
    p = _parse_params(req.params)
    jid = uuid.uuid4().hex
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"rcv-{jid[:8]}-"))
    try:
        ir_obj = None
        if req.ir is not None:
            ir_obj = _resolve_ir(json.dumps(req.ir).encode(), None)
        elif req.ir_url:
            ir_obj = _resolve_ir(_fetch_blob(req.ir_url, d / "ir-in.json", limit_mb=20).read_bytes(), None)
        mo = req.moth if req.moth is not None and ir_obj is None else None
        key = moth.resolve(x_moth_key) if mo is not None else None
        if mo is not None and not key:
            raise EngineError("moth_unauthorized", "No Moth key applied; add one with the key button.", 401)
        src, echo_src = _fetch_blob(req.video_url, d / "input.mp4"), None
        if req.echo_url:
            echo_src = _fetch_blob(req.echo_url, d / "echo.mp4")
        for f in filter(None, (src, echo_src)):
            try:
                probe(str(f))
            except Exception:
                raise EngineError("invalid_video", "could not read a video stream in an uploaded clip")
        with LOCK:
            JOBS[jid] = {"id": jid, "status": "running", "stage": "queued", "progress": 0.0, "created": time.time(),
                         "ir_source": "reference" if req.ir_url else ("upload" if req.ir is not None else ("moth" if mo is not None else "measured")),
                         "cross_echo": echo_src is not None}
        _run(jid, p, src, ir_obj, mo, echo_src, key, d=d)
        with LOCK:
            j = JOBS.pop(jid)
        if j["status"] == "succeeded":
            urls = {slot: _blob_put(f"renders/{jid}/{name}", (d / name).read_bytes(), mime) for slot, (name, mime) in FILES.items()}
            j["result"]["data"]["files"] = urls
            j["result"]["data"]["ir_ref"] = urls["ir"]
        return j
    finally:
        with LOCK:
            JOBS.pop(jid, None)
        shutil.rmtree(d, ignore_errors=True)


@app.get("/jobs")
def jobs(x_client_id: str | None = Header(None)):
    """Local dev lists every job; in public mode a visitor only sees their own (X-Client-Id)."""
    with LOCK:
        mine = [j for j in JOBS.values() if not moth.public() or (x_client_id and j.get("client") == x_client_id)]
        return sorted(({k: v for k, v in j.items() if k not in PRIVATE} for j in mine), key=lambda j: -j["created"])


@app.get("/jobs/{jid}")
def job(jid: str):
    with LOCK:
        if jid not in JOBS:
            raise HTTPException(404, "unknown job")
        return {k: v for k, v in JOBS[jid].items() if k != "client"}


FILES = {"video": ("video.mp4", "video/mp4"), "ir": ("ir.json", "application/json"),
         "taps": ("taps.json", "application/json"), "spacetime": ("spacetime.png", "image/png")}


@app.get("/jobs/{jid}/files/{slot}")
def job_file(jid: str, slot: str, download: bool = False):
    if slot not in FILES or not JID.match(jid):
        raise HTTPException(404, "unknown slot")
    name, mime = FILES[slot]
    path = ROOT / jid / name
    if not path.exists():
        raise HTTPException(404, "not ready")
    return FileResponse(path, media_type=mime, filename=f"{jid}-{name}",
                        content_disposition_type="attachment" if download else "inline")


@app.get("/", response_class=HTMLResponse)
def ui():
    return (STATIC / "index.html").read_text()
