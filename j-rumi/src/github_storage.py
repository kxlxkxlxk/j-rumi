"""
Uses the GitHub repo itself as the foundation-shade "database" -- no
separate DB service. Reads/writes data/foundation_db.json in the repo via
GitHub's REST API, using a fine-grained personal access token with
Contents: Read and write on this one repo.

Config comes from Streamlit secrets (set in the Streamlit Cloud app
settings, never committed to the repo):

    [github]
    token = "github_pat_..."
    repo = "kxlxkxlxk/j-rumi"
    branch = "main"
    data_path = "data/foundation_db.json"
"""
import base64
import json
import requests

API_ROOT = "https://api.github.com"


class GitHubStorageError(Exception):
    pass


def _headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_config():
    import streamlit as st

    cfg = st.secrets.get("github", {})
    required = ["token", "repo"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise GitHubStorageError(f"Streamlit secrets에 github.{missing} 설정이 없어요")
    cfg = dict(cfg)
    cfg.setdefault("branch", "main")
    cfg.setdefault("data_path", "data/foundation_db.json")
    return cfg


def read_shades():
    """Returns (shades_list, sha) -- sha is needed to write back safely."""
    cfg = get_config()
    url = f"{API_ROOT}/repos/{cfg['repo']}/contents/{cfg['data_path']}"
    resp = requests.get(url, headers=_headers(cfg["token"]), params={"ref": cfg["branch"]}, timeout=15)
    if resp.status_code == 404:
        return [], None
    if resp.status_code != 200:
        raise GitHubStorageError(f"DB 파일을 읽지 못했어요 ({resp.status_code}): {resp.text[:200]}")
    payload = resp.json()
    content = base64.b64decode(payload["content"]).decode("utf-8")
    data = json.loads(content)
    return data.get("shades", []), payload["sha"]


def write_shades(shades: list, sha: str, commit_message: str):
    cfg = get_config()
    url = f"{API_ROOT}/repos/{cfg['repo']}/contents/{cfg['data_path']}"
    body_str = json.dumps({"shades": shades}, ensure_ascii=False, indent=2)
    body_b64 = base64.b64encode(body_str.encode("utf-8")).decode("ascii")
    payload = {
        "message": commit_message,
        "content": body_b64,
        "branch": cfg["branch"],
    }
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=_headers(cfg["token"]), json=payload, timeout=15)
    if resp.status_code not in (200, 201):
        raise GitHubStorageError(f"DB 저장 실패 ({resp.status_code}): {resp.text[:300]}")
    return resp.json()["content"]["sha"]


def add_shade(new_shade: dict, commit_message: str = None):
    shades, sha = read_shades()
    shades.append(new_shade)
    msg = commit_message or f"Add shade: {new_shade.get('brand')} {new_shade.get('name')}"
    write_shades(shades, sha, msg)
    return shades


def delete_shade(shade_id: str, commit_message: str = None):
    shades, sha = read_shades()
    shades = [s for s in shades if s.get("id") != shade_id]
    msg = commit_message or f"Delete shade: {shade_id}"
    write_shades(shades, sha, msg)
    return shades


def update_shade(shade_id: str, updates: dict, commit_message: str = None):
    shades, sha = read_shades()
    for s in shades:
        if s.get("id") == shade_id:
            s.update(updates)
    msg = commit_message or f"Update shade: {shade_id}"
    write_shades(shades, sha, msg)
    return shades
