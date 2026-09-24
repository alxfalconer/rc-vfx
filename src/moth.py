"""Bring-your-own-key Moth client: measure F on the platform instead of the local sim.

The key lives only in this process (set from the UI, or MOTH_API_KEY / .env at start-up).
It is never returned to the browser; the browser talks to this server, this server talks to
the Moth API with `Authorization: Bearer moth_...`.

Flow (moth-api): POST /engines/{id}/process -> poll /jobs/{id}/status -> GET /jobs/{id}/result,
whose body is either an inline `result` or presigned `outputs`. Either way we expect a
`trajectory` envelope, which EchoIR.from_trajectory already reads.

The otoc-echo-v1 params schema is not pinned here: it is fetched from GET /engines/{id} and
only the circuit fields it declares are sent.
"""
from __future__ import annotations
import os, pathlib, threading, time
from typing import Callable, Optional

import httpx

from echo_ir import EchoIR

BASE = os.environ.get("MOTH_API_URL", "https://api.mothquantum.com").rstrip("/") + "/api/v1"
ECHO_ENGINE = os.environ.get("MOTH_ECHO_ENGINE", "otoc-echo-v1")
POLL_S, TIMEOUT_S = 3.0, float(os.environ.get("MOTH_TIMEOUT_S", 45 * 60))
TRANSPORT: Optional[httpx.BaseTransport] = None        # tests swap in httpx.MockTransport

_lock = threading.Lock()
_key: Optional[str] = None
_state = {"state": "off", "account": None, "source": None, "message": None}
_schema: Optional[dict] = None


class MothError(Exception):
    def __init__(self, code: str, message: str, status: int = 502):
        super().__init__(message); self.code, self.message, self.status = code, message, status


def load_env_file(path) -> None:
    """KEY=VALUE lines into os.environ without overriding what the shell already set."""
    p = pathlib.Path(path) if path else None
    if not p or not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        os.environ.setdefault(k, v.strip().strip("'\""))


def _client(key: Optional[str] = None) -> httpx.Client:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.Client(base_url=BASE, headers=headers, timeout=30, transport=TRANSPORT)


def _detail(r: httpx.Response) -> str:
    try:
        j = r.json()
        return j.get("detail") or j.get("title") or r.reason_phrase
    except Exception:
        return r.reason_phrase or str(r.status_code)


def _check(r: httpx.Response) -> httpx.Response:
    if r.status_code == 401:
        with _lock:
            _state.update(state="bad", message="key rejected by moth")
        raise MothError("moth_unauthorized", "Moth rejected the API key (invalid, disabled or revoked).", 401)
    if r.status_code == 429:
        raise MothError("moth_unavailable", "Moth rate limit reached; try again in a minute.", 429)
    if r.status_code >= 500:
        raise MothError("moth_unavailable", f"Moth API unavailable ({r.status_code}: {_detail(r)}).", 502)
    if r.status_code >= 400:
        raise MothError("moth_failed", f"Moth API {r.status_code}: {_detail(r)}", 422)
    return r


# ---------------------------------------------------------------- key
def status() -> dict:
    with _lock:
        return {**_state, "engine": ECHO_ENGINE, "api": BASE}


def current_key() -> Optional[str]:
    with _lock:
        return _key if _state["state"] == "on" else None


def set_key(key: str, source: str = "ui") -> dict:
    global _key, _schema
    key = (key or "").strip()
    if not key.startswith("moth_"):
        raise MothError("moth_unauthorized", "Moth keys start with moth_.", 422)
    try:
        with _client(key) as c:
            r = c.get("/me")
    except httpx.HTTPError as e:
        raise MothError("moth_unavailable", f"could not reach {BASE}: {type(e).__name__}", 502)
    if r.status_code == 401:
        with _lock:
            _key, _schema = None, None
            _state.update(state="bad", account=None, source=source, message="key rejected by moth")
        raise MothError("moth_unauthorized", "Moth rejected this key (invalid, disabled or revoked).", 401)
    _check(r)
    me = r.json()
    with _lock:
        _key, _schema = key, None
        _state.update(state="on", account=me.get("email") or me.get("id"), source=source, message=None)
    return status()


def clear() -> dict:
    global _key, _schema
    with _lock:
        _key, _schema = None, None
        _state.update(state="off", account=None, source=None, message=None)
    return status()


def init_from_env() -> None:
    k = os.environ.get("MOTH_API_KEY")
    if not k:
        return
    try:
        set_key(k, source="env")
    except MothError as e:
        with _lock:
            _state.update(message=e.message)


# ---------------------------------------------------------------- engine
def engine_schema() -> dict:
    global _schema
    key = current_key()
    if not key:
        raise MothError("moth_unauthorized", "No Moth key applied.", 401)
    if _schema is None:
        with _client(key) as c:
            rec = _check(c.get(f"/engines/{ECHO_ENGINE}")).json()
        _schema = rec.get("params_schema") or {}
    return _schema


def options() -> dict:
    """What the UI needs to draw machine / shots selectors, straight from the schema."""
    props = (engine_schema().get("properties") or {})
    out = {"engine": ECHO_ENGINE, "fields": sorted(props),
           "describe": {k: {x: v[x] for x in ("type", "default", "enum", "minimum", "maximum", "description") if x in v}
                        for k, v in props.items() if isinstance(v, dict)}}
    for name in ("machine", "backend_name", "shots", "mode"):
        if name in props:
            f = props[name]
            out[name] = {k: f[k] for k in ("enum", "default", "minimum", "maximum", "description") if k in f}
    return out


CIRCUIT = ("n_sites", "depth", "theta_zz", "theta_x", "theta_z", "kick", "kick_site",
           "lattice", "shape", "disorder", "seed")


def build_params(sim_kwargs: dict, opts: dict, schema: dict) -> dict:
    props = schema.get("properties")
    wanted = {k: v for k, v in sim_kwargs.items() if k in CIRCUIT and v is not None}
    wanted.update({k: v for k, v in (opts or {}).items() if k in ("machine", "backend_name", "shots", "mode") and v not in (None, "")})
    shape = wanted.pop("shape", None)
    if wanted.get("lattice") == "square" and shape:
        wanted["width"], wanted["height"] = shape                  # otoc-echo-v1 spells [nx, ny] as width/height
        wanted.pop("n_sites", None)
    machine = wanted.get("machine") or wanted.get("backend_name")
    if machine and machine != "aer" and "via" not in wanted:
        wanted["via"] = "mothbackend"                              # Moth's credentials; `direct` would need an IBM token
    return {k: v for k, v in wanted.items() if props is None or k in props}


# ---------------------------------------------------------------- measure
def unwrap(body) -> dict:
    """Find the trajectory envelope: moth-api nests it as {result: {output: {data, extras, provenance}}}."""
    env = body
    for _ in range(4):
        if not isinstance(env, dict) or "data" in env:
            break
        env = env.get("output") or env.get("result") or env
        if env is body:
            break
    if isinstance(env, dict) and "data" not in env and "series" in env:
        env = {"data": env}                                      # a bare data block
    if not isinstance(env, dict) or "data" not in env:
        raise ValueError(f"no trajectory in result (top-level keys: {sorted(body)[:8] if isinstance(body, dict) else type(body).__name__})")
    ex = dict(env.get("extras") or {})
    if ex.get("lattice") == "square" and "shape" not in ex and ex.get("width") and ex.get("height"):
        ex["shape"] = [ex["width"], ex["height"]]
    for k in ("taps", "params", "spec"):                         # bulky; the engine's own copy stays on moth
        ex.pop(k, None)
    return {**env, "extras": {**ex, **({"provenance": env["provenance"]} if "provenance" in env else {})}}


def _fetch_result(c: httpx.Client, job_id: str) -> dict:
    body = _check(c.get(f"/jobs/{job_id}/result")).json()
    if body.get("result") is not None:
        return body["result"]
    for o in body.get("outputs") or []:
        if "json" in (o.get("content_type") or "") or (o.get("filename") or "").endswith(".json"):
            with httpx.Client(timeout=60, transport=TRANSPORT) as raw:      # presigned: no auth header
                r = raw.get(o["url"]); r.raise_for_status()
                return r.json()
    raise MothError("moth_failed", "Moth job finished without a trajectory result.")


def measure(sim_kwargs: dict, opts: dict, on_status: Callable[[str, float], None] = lambda s, p: None) -> EchoIR:
    key = current_key()
    if not key:
        raise MothError("moth_unauthorized", "No Moth key applied; add one with the key button.", 401)
    params = build_params(sim_kwargs, opts, engine_schema())
    with _client(key) as c:
        sub = _check(c.post(f"/engines/{ECHO_ENGINE}/process", json={"params": params})).json()
        jid = sub["job_id"]
        on_status("queued", 0.0)
        t0 = time.time()
        while True:
            s = _check(c.get(f"/jobs/{jid}/status")).json()
            st = s.get("status", "processing")
            if st == "completed":
                try:
                    env = unwrap(s.get("result"))
                except ValueError:
                    env = _fetch_result(c, jid)
                break
            if st in ("failed", "cancelled"):
                err = s.get("error") or {}
                raise MothError("moth_failed", f"Moth job {st}: {err.get('message') or err.get('type') or 'no detail'}")
            on_status(st, float(s.get("progress") or 0.0))
            if time.time() - t0 > TIMEOUT_S:
                raise MothError("moth_failed", f"Moth job {jid} still {st} after {int(TIMEOUT_S // 60)} min.")
            time.sleep(POLL_S)
    try:
        ir = EchoIR.from_trajectory(unwrap(env))
    except Exception as e:
        raise MothError("moth_failed", f"Moth result is not a readable trajectory: {e}")
    return EchoIR(F=ir.F, lattice=ir.lattice, shape=ir.shape, kick_site=ir.kick_site,
                  meta={**ir.meta, "machine": params.get("machine") or params.get("backend_name") or "moth",
                        "moth_job": jid, "moth_engine": ECHO_ENGINE})
