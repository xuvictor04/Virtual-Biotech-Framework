"""Local serving operations (vbt.local): serving profiles, harness config profiles, command
rendering, docker tags, GPU auto-pick, deploy files and the capability check / bench against a
fake OpenAI-compatible server (offline)."""

import argparse
import asyncio
import copy
import json
import re
import shlex
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import yaml

from vbt.config import CONFIG_DIR, PROJECT_ROOT, load_config
from vbt.local import add_local_parsers, load_local_profiles
from vbt.local import check as chk
from vbt.local import serve as srv
from vbt.local.profiles import (
    LOCAL_MODELS_FILE,
    LocalProfileError,
    docker_run_argv,
    parse_nvidia_smi,
    pick_profile,
    resolve_serve,
    select_docker_tag,
    shell_command,
    validate_local_models,
)
from vbt.providers.base import (
    Message,
    ModelResponse,
    ProviderError,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    Usage,
    system_text,
)

PROFILES = load_local_profiles()
RAW = yaml.safe_load(LOCAL_MODELS_FILE.read_text())
COMPOSE = PROJECT_ROOT / "deploy" / "local" / "docker-compose.yml"
SERVE_SCRIPT = PROJECT_ROOT / "deploy" / "local" / "serve_vllm.sh"
QWEN_HARNESS = ("local-h100", "local-h200", "local-rtxpro6000", "local-5090", "local-dp")


# ============================================================== serving-profile catalogue

def test_catalogue_validates_and_lists_every_digest_profile():
    assert validate_local_models(RAW) == []
    assert list(PROFILES) == ["h100", "h200", "rtxpro6000", "b200", "5090", "dp", "deepseek-v4"]
    digest = {p["digest_profile"].split(" ")[0] for p in PROFILES.values()}
    assert digest == {"P1-hopper-1gpu-primary", "P2-blackwell-1gpu-primary", "P3-consumer-32gb",
                      "P4-multi-gpu-scale-out", "P5-multi-gpu-max-quality"}
    for name, p in PROFILES.items():
        assert p["vision"] is False, name
        assert p["verification"] in ("recipe-verified", "supported, smoke-test"), name
        assert p["caveats"], name
        assert p["engine"]["min_version"] == "0.31.0", name
        assert p["env"]["VLLM_ENFORCE_STRICT_TOOL_CALLING"] == "1"
        assert "bulk" in p["variants"] or name == "deepseek-v4", name


def _problems(mutate):
    doc = copy.deepcopy(RAW)
    mutate(doc)
    return "\n".join(validate_local_models(doc))


def test_validation_catches_inconsistent_profiles():
    def ctx(doc):
        doc["profiles"]["h100"]["context_tokens"] = 131072
    assert "context_tokens=131072 but --max-model-len 262144" in _problems(ctx)

    def no_details(doc):
        doc["profiles"]["h100"]["vllm_args"].remove("--enable-prompt-tokens-details")
    assert "missing required argument --enable-prompt-tokens-details" in _problems(no_details)

    def family(doc):
        doc["profiles"]["h200"]["family"] = "qwen9"
    assert "unknown family 'qwen9'" in _problems(family)

    def verification(doc):
        doc["profiles"]["5090"]["verification"] = "verified"
    assert "verification must be one of" in _problems(verification)

    def port(doc):
        doc["profiles"]["h100"]["vllm_args"].append({"--port": 9000})
    assert "--port must not be in vllm_args" in _problems(port)

    def bad_variant(doc):
        doc["profiles"]["h100"]["variants"]["bulk"]["remove"] = ["--does-not-exist"]
    assert "removes --does-not-exist" in _problems(bad_variant)

    def variant_summary(doc):
        doc["profiles"]["h100"]["variants"]["bulk"]["max_num_seqs"] = 64
    assert "variants.bulk: max_num_seqs=64 but --max-num-seqs 128" in _problems(variant_summary)

    def mtp(doc):
        doc["profiles"]["h200"]["mtp"] = None
    assert "mtp=None but --speculative-config" in _problems(mtp)

    def vision(doc):
        doc["profiles"]["h100"]["vllm_args"].remove("--language-model-only")
    assert "--language-model-only is not passed" in _problems(vision)

    def dup(doc):
        doc["profiles"]["h100"]["vllm_args"].append({"--max-num-seqs": 48})
    assert "duplicate arguments ['--max-num-seqs']" in _problems(dup)

    def tags(doc):
        doc["defaults"]["docker"]["tags"] = []
    assert "docker.tags must be a non-empty list" in _problems(tags)

    assert validate_local_models([]) == ["the catalogue must be a mapping"]


def test_profile_families_are_known_to_the_adapter():
    fams = pytest.importorskip("vbt.providers.families")
    for name, p in PROFILES.items():
        names = {p["family"]} | {v.get("family") for v in p["variants"].values() if v.get("family")}
        for fam in names:
            assert fam in fams.FAMILIES, (name, fam)
            assert fams.resolve_family(p["served_model_name"], fam).name == fam


# ============================================================== harness config profiles

def _harness(name, monkeypatch=None):
    return load_config([name])


@pytest.mark.parametrize("serving", list(PROFILES))
def test_harness_profile_matches_its_serving_profile(serving, monkeypatch):
    monkeypatch.delenv("VBT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    p = PROFILES[serving]
    hp = p["harness"]["profile"]
    assert (CONFIG_DIR / "profiles" / f"{hp}.yaml").is_file()
    cfg = load_config([hp])
    prov = cfg["provider"]
    opts = prov["options"]
    assert prov["name"] == "vllm"
    assert opts["base_url"] == "http://localhost:8000/v1"
    assert opts["served_model_name"] == p["served_model_name"]
    assert opts["family"] == p["family"]
    assert opts["pricing"] is None and opts["read_timeout_s"] == 900
    assert opts["data_parallel_size"] == p["data_parallel_size"]
    seqs = (p["max_num_seqs"] or 256) * p["data_parallel_size"]
    assert opts["max_concurrency"] <= seqs
    if hp != "local-h200" or serving == "h200":   # b200 shares local-h200 (a smaller client cap)
        assert opts["max_concurrency"] == p["harness"]["max_concurrency"]
    for tier, t in cfg["models"].items():
        assert t["model"] == p["served_model_name"], (tier, t)
        assert t["context_window_tokens"] == p["context_tokens"], (tier, t)
        if t.get("thinking_budget"):
            assert t["thinking"] and t["max_tokens"] > t["thinking_budget"], (tier, t)
    assert cfg["context"]["default_window_tokens"] == p["context_tokens"]
    assert 0 < cfg["context"]["soft_ratio"] < cfg["context"]["hard_ratio"] < 1
    assert cfg["web"]["search"] == {"backend": "auto", "searxng_url": "http://localhost:8888"}
    assert cfg["limits"]["max_turn_tokens"] > cfg["limits"]["max_item_tokens"] > 0
    assert cfg["limits"]["max_turn_cost_usd"] == 150            # kept; 0-cost locally
    assert 0 < cfg["bulk"]["default_concurrency"] <= opts["max_concurrency"]
    assert cfg["limits"]["max_parallel_agents"] <= opts["max_concurrency"]


@pytest.mark.parametrize("hp", QWEN_HARNESS)
def test_qwen_tiers_follow_the_tier_mapping(hp):
    m = load_config([hp])["models"]
    for tier, t in m.items():
        # never 'high'/'max'/'minimal' for Qwen3.8 (the template raises -> HTTP 400)
        assert t["effort"] in ("xhigh", "medium", "low", None), (hp, tier)
        assert bool(t["thinking"]) == (t["effort"] is not None), (hp, tier)
    assert m["support"] == {**m["support"], "effort": None, "thinking": False}
    if hp != "local-5090":
        assert (m["orchestrator"]["effort"], m["orchestrator"]["thinking_budget"], m["orchestrator"]["max_tokens"]) \
            == ("xhigh", 24576, 40960)
        assert (m["bulk"]["effort"], m["bulk"]["thinking_budget"], m["bulk"]["max_tokens"]) == ("medium", 3072, 8192)
        assert m["support"]["max_tokens"] == 16000
    if hp in ("local-h100", "local-h200", "local-rtxpro6000"):
        assert (m["scientist"]["effort"], m["scientist"]["thinking_budget"], m["scientist"]["max_tokens"]) \
            == ("medium", 8192, 32768)
    if hp == "local-5090":
        assert m["bulk"]["thinking"] is False and m["scientist"]["thinking_budget"] == 4096
        ctx = load_config([hp])["context"]
        assert (ctx["soft_ratio"], ctx["hard_ratio"]) == (0.5, 0.62)


def test_deepseek_profile_uses_its_own_effort_vocabulary():
    cfg = load_config(["local-deepseek-v4"])
    assert cfg["provider"]["options"]["family"] == "deepseek_v4"
    assert cfg["provider"]["options"]["data_parallel_size"] == 4
    assert {t["effort"] for t in cfg["models"].values()} <= {"high", "low", None}
    assert not any(t.get("thinking_budget") for t in cfg["models"].values())


def test_harness_profile_env_overrides(monkeypatch):
    monkeypatch.setenv("VBT_LLM_BASE_URL", "http://gpu-box:9000/v1")
    monkeypatch.setenv("SEARXNG_URL", "http://search:8080")
    cfg = load_config(["local-h100"])
    assert cfg["provider"]["options"]["base_url"] == "http://gpu-box:9000/v1"
    assert cfg["web"]["search"]["searxng_url"] == "http://search:8080"


CLAUDE_TIERS = {
    "orchestrator": {"model": "claude-opus-5", "effort": "high", "max_tokens": 64000, "thinking": True},
    "scientist": {"model": "claude-opus-5", "effort": "high", "max_tokens": 64000, "thinking": True},
    "support": {"model": "claude-haiku-4-5", "effort": None, "max_tokens": 16000, "thinking": False},
    "bulk": {"model": "claude-opus-5", "effort": "medium", "max_tokens": 32000, "thinking": True},
}
CLAUDE_OPTIONS = {"refusal_fallback": True, "web_search_model": "claude-sonnet-5", "prompt_caching": True,
                  "max_retries": 4}


@pytest.mark.parametrize("layers", [["claude"], ["local-h100", "claude"], ["local-dp", "claude"]])
def test_claude_profile_restores_the_anthropic_defaults(layers):
    # ["local-h100", "claude"] stands for "default.yaml switched to local, then --profile claude".
    cfg = load_config(layers)
    prov = cfg["provider"]
    assert prov["name"] == "anthropic"
    live = {k: v for k, v in prov["options"].items() if v is not None}
    assert live == CLAUDE_OPTIONS          # no base_url / served_model_name / ... leak into the Anthropic SDK
    assert prov.get("model_pattern") is None
    for tier, want in CLAUDE_TIERS.items():
        got = {k: v for k, v in cfg["models"][tier].items() if v is not None or k == "effort"}
        assert got == want, tier
    assert cfg["model_aliases"]["opus"] == "claude-opus-5"
    assert cfg["context"]["default_window_tokens"] == 200000
    assert (cfg["context"]["soft_ratio"], cfg["context"]["hard_ratio"]) == (0.7, 0.85)
    assert cfg["limits"]["max_turn_tokens"] is None and cfg["limits"]["max_turn_cost_usd"] == 150


def test_claude_profile_tiers_equal_the_pre_switch_defaults():
    """claude.yaml copies the Anthropic provider/model blocks of the original default.yaml."""
    claude = yaml.safe_load((CONFIG_DIR / "profiles" / "claude.yaml").read_text())
    default = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text())
    if default["provider"]["name"] == "anthropic":   # before the default switch: compare with it directly
        assert {k: v for k, v in claude["provider"]["options"].items() if v is not None} == \
            default["provider"]["options"]
        for tier, t in default["models"].items():
            assert {k: v for k, v in claude["models"][tier].items() if v is not None or k == "effort"} == t
        assert claude["model_aliases"] == default["model_aliases"]


# ============================================================== command rendering

ABSENT = object()
EXPECTED = {
    "h100": {"hf": "Qwen/Qwen3.8-27B-FP8", "--kv-cache-dtype": "fp8", "--max-num-seqs": "48",
             "--max-model-len": "262144", "--max-num-batched-tokens": "16384",
             "--speculative-config": '{"method":"mtp","num_speculative_tokens":3}',
             "--language-model-only": None, "--tool-strict-level": "function", "--tool-call-parser": "qwen3_coder",
             "--enable-chunked-prefill": ABSENT},
    "h200": {"hf": "Qwen/Qwen3.8-27B-FP8", "--kv-cache-dtype": "auto", "--max-num-seqs": "96",
             "--max-model-len": "262144", "--speculative-config": '{"method":"mtp","num_speculative_tokens":3}',
             "--language-model-only": None, "--tool-strict-level": "function"},
    "rtxpro6000": {"hf": "RedHatAI/Qwen3.8-27B-NVFP4", "--kv-cache-dtype": "fp8", "--max-num-seqs": "48",
                   "--max-num-batched-tokens": "8192", "--max-model-len": "262144",
                   "--speculative-config": '{"method":"mtp","num_speculative_tokens":3}',
                   "--language-model-only": None, "--tool-strict-level": "function"},
    "b200": {"hf": "RedHatAI/Qwen3.8-27B-NVFP4", "--kv-cache-dtype": "fp8", "--max-num-seqs": "128",
             "--language-model-only": None},
    "5090": {"hf": "RedHatAI/Qwen3.8-27B-INT4", "--tokenizer": "Qwen/Qwen3.8-27B", "--kv-cache-dtype": "fp8",
             "--max-num-seqs": "4", "--max-model-len": "131072", "--max-num-batched-tokens": "4096",
             "--speculative-config": ABSENT, "--language-model-only": None, "--tool-call-parser": "qwen3_xml",
             "--tool-strict-level": "function", "--enable-chunked-prefill": None},
    "dp": {"hf": "Qwen/Qwen3.8-27B", "--data-parallel-size": "4", "--kv-cache-dtype": "auto",
           "--max-num-seqs": "64", "--speculative-config": '{"method":"mtp","num_speculative_tokens":3}',
           "--language-model-only": None},
    "deepseek-v4": {"hf": "deepseek-ai/DeepSeek-V4-Flash-0731", "--served-model-name": "dsv4-flash",
                    "--trust-remote-code": None, "--data-parallel-size": "4", "--enable-expert-parallel": None,
                    "--kv-cache-dtype": "fp8", "--block-size": "256", "--max-model-len": "393216",
                    "--tokenizer-mode": "deepseek_v4", "--tool-call-parser": "deepseek_v4",
                    "--reasoning-parser": "deepseek_v4", "--tool-strict-level": "function",
                    "--default-chat-template-kwargs": '{"reasoning_effort":"high"}',
                    "--override-generation-config": '{"temperature":1.0,"top_p":0.95}',
                    "--language-model-only": ABSENT, "--speculative-config": ABSENT},
}
QWEN_COMMON = {"--served-model-name": "qwen3.8-27b", "--reasoning-parser": "qwen3", "--enable-auto-tool-choice": None,
               "--enable-prefix-caching": None, "--enable-prompt-tokens-details": None,
               "--default-chat-template-kwargs": '{"reasoning_effort":"medium"}'}


def _flags(argv):
    """{flag: value-or-None} of a rendered argv (after the model)."""
    out, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[a] = argv[i + 1]
                i += 2
                continue
            out[a] = None
        i += 1
    return out


@pytest.mark.parametrize("name", list(EXPECTED))
def test_rendered_command_matches_the_corrected_profile(name):
    spec = resolve_serve(PROFILES, name)
    argv = spec.vllm_argv()
    exp = dict(EXPECTED[name])
    assert argv[:3] == ["vllm", "serve", exp.pop("hf")]
    assert argv[-4:] == ["--host", "127.0.0.1", "--port", "8000"]
    flags = _flags(argv[3:])
    if name != "deepseek-v4":
        exp = {**QWEN_COMMON, **exp}
    for flag, want in exp.items():
        if want is ABSENT:
            assert flag not in flags, (name, flag)
        else:
            assert flag in flags and flags[flag] == want, (name, flag, flags.get(flag))
    assert "--model" not in flags


@pytest.mark.parametrize("name", list(PROFILES))
def test_tool_strict_level_only_for_vllm_031_and_later(name):
    assert "--tool-strict-level" in resolve_serve(PROFILES, name).vllm_argv()
    assert "--tool-strict-level" in resolve_serve(PROFILES, name, engine_version="0.32.1").vllm_argv()
    old = resolve_serve(PROFILES, name, engine_version="0.30.0")
    assert "--tool-strict-level" not in old.vllm_argv()
    assert any("dropped --tool-strict-level" in n for n in old.notes)
    assert old.engine_version == "0.30.0"


def test_bulk_variants_drop_mtp_and_raise_sequences():
    h100 = resolve_serve(PROFILES, "h100", bulk=True)
    f = _flags(h100.vllm_argv())
    assert "--speculative-config" not in f and f["--max-model-len"] == "65536" and f["--max-num-seqs"] == "128"
    assert (h100.context_tokens, h100.max_num_seqs, h100.mtp, h100.variants) == (65536, 128, None, ["bulk"])
    assert "--enable-prefix-caching" in f            # bulk keeps prefix caching (shared system prompt)
    assert _flags(resolve_serve(PROFILES, "h200", bulk=True).vllm_argv())["--max-num-seqs"] == "256"
    rtx = resolve_serve(PROFILES, "rtxpro6000", bulk=True)
    assert rtx.context_tokens == 262144 and _flags(rtx.vllm_argv())["--max-num-seqs"] == "128"
    assert "--speculative-config" not in _flags(resolve_serve(PROFILES, "dp", bulk=True).vllm_argv())
    # --bulk and --variant bulk are the same thing
    assert resolve_serve(PROFILES, "h100", variants=["bulk"], bulk=True).variants == ["bulk"]


def test_variants_and_overrides():
    eager = resolve_serve(PROFILES, "5090", variants=["eager"])
    assert "--enforce-eager" in eager.vllm_argv()
    moe = resolve_serve(PROFILES, "5090", variants=["bulk-moe"])
    f = _flags(moe.vllm_argv())
    assert moe.hf_id == "nvidia/Qwen3.6-35B-A3B-NVFP4" and moe.family == "qwen3_6"
    assert moe.served_model_name == f["--served-model-name"] == "qwen3.6-35b-a3b"
    assert "--tokenizer" not in f and f["--quantization"] == "modelopt_fp4"
    assert f["--default-chat-template-kwargs"] == '{"preserve_thinking":true}'
    assert moe.env["VLLM_HAS_FLASHINFER_CUBIN"] == "1"
    fp8 = resolve_serve(PROFILES, "h200", variants=["fp8kv"])
    assert (fp8.kv_cache_dtype, _flags(fp8.vllm_argv())["--kv-cache-dtype"]) == ("fp8", "fp8")
    dp_h100 = resolve_serve(PROFILES, "dp", variants=["h100"], data_parallel=2)
    f = _flags(dp_h100.vllm_argv())
    assert dp_h100.hf_id == "Qwen/Qwen3.8-27B-FP8" and f["--data-parallel-size"] == "2" and f["--kv-cache-dtype"] == "fp8"
    assert dp_h100.data_parallel_size == 2
    single = resolve_serve(PROFILES, "dp", data_parallel=1)
    assert "--data-parallel-size" not in single.vllm_argv() and single.data_parallel_size == 1
    with pytest.raises(LocalProfileError, match="no variant 'turbo'.*available: bulk"):
        resolve_serve(PROFILES, "h100", variants=["turbo"])
    with pytest.raises(LocalProfileError, match="unknown serving profile 'a100'"):
        resolve_serve(PROFILES, "a100")
    alt = resolve_serve(PROFILES, "rtxpro6000", hf_id="nvidia/Qwen3.8-27B-NVFP4")
    assert alt.vllm_argv()[2] == "nvidia/Qwen3.8-27B-NVFP4"
    assert any("recipe-verified" in n for n in alt.notes)
    assert any("not listed" in n for n in resolve_serve(PROFILES, "h100", hf_id="someone/fork").notes)
    extra = resolve_serve(PROFILES, "h100", extra_args=["--enforce-eager"], host="0.0.0.0", port=8001)
    assert extra.vllm_argv()[-5:] == ["--enforce-eager", "--host", "0.0.0.0", "--port", "8001"]
    assert extra.base_url == "http://localhost:8001/v1"
    tep = resolve_serve(PROFILES, "deepseek-v4", variants=["tep"])
    assert "--data-parallel-size" not in tep.vllm_argv() and _flags(tep.vllm_argv())["--tensor-parallel-size"] == "4"
    assert tep.data_parallel_size == 1


def test_shell_rendering_quotes_json_and_round_trips():
    spec = resolve_serve(PROFILES, "h100")
    text = shell_command(spec.vllm_argv(), spec.env)
    assert text.startswith("VLLM_ENFORCE_STRICT_TOOL_CALLING=1 vllm serve Qwen/Qwen3.8-27B-FP8 \\\n")
    assert "  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \\\n" in text
    one = shell_command(spec.vllm_argv(), multiline=False)
    assert shlex.split(one) == spec.vllm_argv()
    assert shlex.split(text.replace("\\\n", " "))[1:] == spec.vllm_argv()


# ============================================================== docker

@pytest.mark.parametrize("driver,tag", [("580.65.06", "v0.31.0"), ("590.12", "v0.31.0"),
                                        ("575.57.08", "v0.31.0-cu129"), ("579.99", "v0.31.0-cu129")])
def test_docker_tag_follows_the_driver(driver, tag):
    p = PROFILES["h100"]
    got, note = select_docker_tag(p["docker"], driver, engine=p["engine"])
    assert got == tag and driver in note


def test_docker_tag_edge_cases():
    p = PROFILES["h100"]
    with pytest.raises(LocalProfileError, match="driver 570.86.15 is too old.*>= 575"):
        select_docker_tag(p["docker"], "570.86.15", engine=p["engine"])
    tag, note = select_docker_tag(p["docker"], None, engine=p["engine"])
    assert tag == "v0.31.0" and "v0.31.0-cu129 (driver >= 575)" in note
    assert select_docker_tag(p["docker"], "576.1", engine=p["engine"], version="0.30.0")[0] == "v0.30.0-cu129"
    assert select_docker_tag(p["docker"], "581.1", engine=p["engine"], version="0.30.0")[0] == "v0.30.0"
    assert select_docker_tag(p["docker"], "581.1", engine=p["engine"], version="0.31.1")[0] == "v0.31.1"
    ds = PROFILES["deepseek-v4"]
    assert select_docker_tag(ds["docker"], "581.1", engine=ds["engine"], version="0.28.0")[0] == "v0.28.0"


def test_docker_run_line(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HOME", raising=False)
    spec = resolve_serve(PROFILES, "h100")
    argv, notes = docker_run_argv(spec, driver="577.1", hf_cache=str(tmp_path / "hf"))
    assert argv[:3] == ["docker", "run", "--rm"]
    for piece in (["--gpus", "all"], ["-p", "127.0.0.1:8000:8000"], ["-v", f"{tmp_path / 'hf'}:/root/.cache/huggingface"],
                  ["-e", "VLLM_ENFORCE_STRICT_TOOL_CALLING=1"], ["-e", "HF_TOKEN"]):
        assert any(argv[i:i + 2] == piece for i in range(len(argv))), piece
    assert "--ipc=host" in argv
    i = argv.index("vllm/vllm-openai:v0.31.0-cu129")
    assert argv[i + 1:] == [*spec.server_argv, "--host", "0.0.0.0", "--port", "8000"]
    assert "cu129" in notes[0]
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home-hf"))
    argv2, _ = docker_run_argv(resolve_serve(PROFILES, "h100", port=8100), driver="580.1", bind="0.0.0.0", detach=True)
    assert f"{tmp_path / 'home-hf'}:/root/.cache/huggingface" in argv2 and "-d" in argv2
    assert "0.0.0.0:8100:8000" in argv2


def test_compose_services_match_the_serving_profiles():
    doc = yaml.safe_load(COMPOSE.read_text())
    services = doc["services"]
    assert set(services) == {"vllm-h100", "vllm-h200", "vllm-rtxpro6000", "vllm-5090", "searxng"}
    for name in ("h100", "h200", "rtxpro6000", "5090"):
        s = services[f"vllm-{name}"]
        spec = resolve_serve(PROFILES, name)
        assert s["profiles"] == [name]
        assert s["command"] == [*spec.server_argv, "--host", "0.0.0.0", "--port", str(spec.container_port)], name
        assert s["image"] == "vllm/vllm-openai:${VLLM_TAG:-v0.31.0}"
        assert s["ipc"] == "host"
        dev = s["deploy"]["resources"]["reservations"]["devices"][0]
        assert dev["driver"] == "nvidia" and dev["capabilities"] == ["gpu"]
        assert any(v.endswith(":/root/.cache/huggingface") for v in s["volumes"])
        assert s["environment"]["VLLM_ENFORCE_STRICT_TOOL_CALLING"] == "1"
        assert s["ports"] == ["${VLLM_BIND:-127.0.0.1}:${VLLM_PORT:-8000}:8000"]
    sx = services["searxng"]
    assert "profiles" not in sx                     # starts with any profile, and alone: `up -d searxng`
    assert sx["image"].startswith("${SEARXNG_IMAGE:-searxng/searxng")
    assert "./searxng/settings.yml:/etc/searxng/settings.yml:ro" in sx["volumes"]
    assert sx["ports"] == ["${SEARXNG_BIND:-127.0.0.1}:${SEARXNG_PORT:-8888}:8080"]


def test_compose_file_is_valid_for_docker_compose():
    docker = shutil.which("docker")
    if not docker:
        pytest.skip("docker not installed")
    ok = subprocess.run([docker, "compose", "version"], capture_output=True, text=True, timeout=30)
    if ok.returncode != 0:
        pytest.skip("docker compose not available")
    for prof in ("h100", "5090"):
        res = subprocess.run([docker, "compose", "-f", str(COMPOSE), "--profile", prof, "config", "--services"],
                             capture_output=True, text=True, timeout=60)
        assert res.returncode == 0, res.stderr
        assert set(res.stdout.split()) == {f"vllm-{prof}", "searxng"}


def test_serve_script_renders_without_the_harness_dependencies():
    if not shutil.which("bash"):
        pytest.skip("bash not available")
    env = {"PATH": "/usr/bin:/bin", "PYTHON": sys.executable, "HOME": str(Path.home())}
    res = subprocess.run(["bash", str(SERVE_SCRIPT), "--profile", "h100", "--bulk", "--dry-run"],
                         capture_output=True, text=True, timeout=60, env=env)
    assert res.returncode == 0, res.stderr
    assert "vllm serve Qwen/Qwen3.8-27B-FP8" in res.stdout and "--max-num-seqs 128" in res.stdout
    assert "--speculative-config" not in res.stdout
    res = subprocess.run(["bash", str(SERVE_SCRIPT), "--profile", "nope", "--dry-run"], capture_output=True,
                         text=True, timeout=60, env=env)
    assert res.returncode == 2 and "unknown serving profile 'nope'" in res.stderr


# ============================================================== nvidia-smi parsing and auto-pick

H100_CSV = "0, NVIDIA H100 80GB HBM3, 81559, 580.65.06\n"
H200_TABLE = """\
Mon Oct  5 12:00:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H200                    On  |   00000000:19:00.0 Off |                    0 |
| N/A   31C    P0             76W /  700W |       1MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
"""
RTXPRO_CSV_HEADER = ("index, name, memory.total [MiB], driver_version\n"
                     "0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 575.57.08\n")
RTX5090_CSV = "0, NVIDIA GeForce RTX 5090, 32607, 580.76.05\n"
RTX5090_LIST = "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-12345678-1234-1234-1234-123456789abc)\n"


def test_parse_nvidia_smi_formats():
    info = parse_nvidia_smi(H200_TABLE)
    assert (info.driver_version, info.cuda_version) == ("580.65.06", "13.0")
    assert [(g.index, g.name, g.memory_mib) for g in info.gpus] == [(0, "NVIDIA H200", 143771)]
    info = parse_nvidia_smi(RTXPRO_CSV_HEADER)
    assert info.driver_version == "575.57.08"
    assert [(g.index, g.name, g.memory_mib) for g in info.gpus] == \
        [(0, "NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887)]
    info = parse_nvidia_smi(H100_CSV)
    assert (info.gpus[0].name, info.gpus[0].memory_mib, info.driver_version) == ("NVIDIA H100 80GB HBM3", 81559,
                                                                                  "580.65.06")
    info = parse_nvidia_smi(RTX5090_LIST)
    assert info.gpus[0].name == "NVIDIA GeForce RTX 5090" and info.gpus[0].memory_mib is None
    assert parse_nvidia_smi("").gpus == []


@pytest.mark.parametrize("text,profile,tag", [
    (H100_CSV, "h100", "v0.31.0"),
    (H200_TABLE, "h200", "v0.31.0"),
    (RTXPRO_CSV_HEADER, "rtxpro6000", "v0.31.0-cu129"),
    (RTX5090_CSV, "5090", "v0.31.0"),
    (RTX5090_LIST, "5090", "v0.31.0"),
    ("0, NVIDIA GH200 480GB, 97871, 580.1\n", "h100", "v0.31.0"),
    ("0, NVIDIA GH200 144G HBM3e, 146831, 580.1\n", "h200", "v0.31.0"),
    ("0, NVIDIA B200, 183359, 580.95.05\n", "b200", "v0.31.0"),
    ("0, NVIDIA H100 NVL, 95830, 578.2\n", "h100", "v0.31.0-cu129"),
])
def test_pick_profile_single_gpu(text, profile, tag):
    pick = pick_profile(text, PROFILES)
    assert pick.profile == profile, pick
    assert pick.docker_tag == tag and pick.data_parallel_size == 1 and pick.variants == []
    assert pick.serve_args() == ["--profile", profile]


@pytest.mark.parametrize("text,reason,alternative", [
    ("0, Tesla T4, 15360, 535.104.05\n", "unrecognised GPU 'Tesla T4'", None),
    ("0, NVIDIA A100-SXM4-80GB, 81920, 550.54.15\n", "unrecognised GPU", "5090"),
    ("0, NVIDIA RTX 6000 Ada Generation, 49140, 580.1\n", "unrecognised GPU", "5090"),
    ("0, NVIDIA GeForce RTX 5090 Laptop GPU, 24463, 580.1\n", "need >= 30 GiB", None),
    ("", "no NVIDIA GPU found", None),
])
def test_pick_profile_unknown_or_too_small(text, reason, alternative):
    pick = pick_profile(text, PROFILES)
    assert pick.profile is None and reason in pick.reason
    assert pick.serve_args() == []
    if alternative:
        assert alternative in pick.alternatives and pick.warnings
    else:
        assert not pick.alternatives


def _csv_gpus(name, mib, n, driver="580.65.06"):
    return "".join(f"{i}, {name}, {mib}, {driver}\n" for i in range(n))


def test_pick_profile_multi_gpu():
    pick = pick_profile(_csv_gpus("NVIDIA H200", 143771, 4), PROFILES)
    assert (pick.profile, pick.variants, pick.data_parallel_size) == ("dp", [], 4)
    assert any(a.startswith("deepseek-v4") for a in pick.alternatives)
    assert pick.serve_args() == ["--profile", "dp", "--data-parallel", "4"]
    pick = pick_profile(_csv_gpus("NVIDIA H100 80GB HBM3", 81559, 8), PROFILES)
    assert (pick.profile, pick.variants, pick.data_parallel_size) == ("dp", ["h100"], 8)
    assert not any(a.startswith("deepseek") for a in pick.alternatives)
    pick = pick_profile(_csv_gpus("NVIDIA B200", 183359, 2), PROFILES)
    assert (pick.profile, pick.variants, pick.data_parallel_size) == ("dp", ["b200"], 2)
    pick = pick_profile(_csv_gpus("NVIDIA GeForce RTX 5090", 32607, 2), PROFILES)
    assert (pick.profile, pick.data_parallel_size) == ("5090", 2)
    assert pick.serve_args() == ["--profile", "5090", "--data-parallel", "2"]
    pick = pick_profile(_csv_gpus("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887, 8), PROFILES)
    assert (pick.profile, pick.variants) == ("dp", ["rtxpro6000"])
    assert "deepseek-v4 --variant rtxpro6000x8" in pick.alternatives
    mixed = H100_CSV + "1, Tesla T4, 15360, 580.65.06\n"
    pick = pick_profile(mixed, PROFILES)
    assert pick.profile == "h100" and any("mixed GPUs" in w for w in pick.warnings)


def test_pick_profile_old_driver_warns():
    pick = pick_profile("0, NVIDIA H100 80GB HBM3, 81559, 570.86.15\n", PROFILES)
    assert pick.profile == "h100" and pick.docker_tag is None
    assert any("too old" in w for w in pick.warnings) and any("570.86.15 < 575" in w for w in pick.warnings)


# ============================================================== CLI wiring

def _parser():
    p = argparse.ArgumentParser(prog="vbt")
    p.add_argument("--profile", action="append", default=[])     # the global harness --profile
    sub = p.add_subparsers(dest="cmd", required=True)
    add_local_parsers(sub)
    return p


def test_local_subcommands_register_handlers_without_clobbering_global_profile():
    p = _parser()
    a = p.parse_args(["--profile", "local-h100", "local", "serve", "--profile", "h100", "--bulk", "--dry-run"])
    assert a.profile == ["local-h100"] and a.serving_profile == "h100" and a.bulk and a.dry_run
    assert a.handler is srv.cmd_serve
    assert p.parse_args(["local", "profiles"]).handler is srv.cmd_profiles
    assert p.parse_args(["local", "check", "--only", "health"]).handler is chk.cmd_check
    assert p.parse_args(["local", "bench", "--levels", "1,8"]).handler is chk.cmd_bench
    for argv in (["local", "check"], ["local", "bench"], ["local", "serve"]):
        assert not hasattr(p.parse_args(argv), "model")     # never shadows the global --model


def test_cmd_serve_dry_run_bare_metal_and_docker(capsys):
    p = _parser()
    a = p.parse_args(["local", "serve", "--profile", "h100", "--dry-run"])
    assert a.handler(a, {}) == 0
    out = capsys.readouterr().out
    assert "vllm serve Qwen/Qwen3.8-27B-FP8" in out and "--tool-strict-level function" in out
    assert "vbt --profile local-h100" in out
    a = p.parse_args(["local", "serve", "--profile", "h100", "--docker", "--driver", "577.1", "--dry-run",
                      "--engine-version", "0.30.0"])
    assert a.handler(a, {}) == 0
    out = capsys.readouterr().out
    assert "vllm/vllm-openai:v0.30.0-cu129 Qwen/Qwen3.8-27B-FP8" in out and "--tool-strict-level function" not in out
    assert "# note: vLLM 0.30.0 < 0.31.0: dropped --tool-strict-level" in out
    a = p.parse_args(["local", "serve", "--profile", "h100", "--json"])
    assert a.handler(a, {}) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["argv"][:3] == ["vllm", "serve", "Qwen/Qwen3.8-27B-FP8"] and data["context_tokens"] == 262144
    a = p.parse_args(["local", "serve", "--profile", "h100", "--variant", "nope", "--dry-run"])
    assert a.handler(a, {}) == 2
    assert "no variant 'nope'" in capsys.readouterr().err


def test_cmd_serve_executes_the_rendered_command(monkeypatch):
    calls = []
    monkeypatch.setattr(srv, "_exec", lambda argv, env: calls.append((list(argv), dict(env))) or 0)
    monkeypatch.setattr(srv.shutil, "which", lambda exe: f"/usr/bin/{exe}")
    p = _parser()
    a = p.parse_args(["local", "serve", "--profile", "5090", "--port", "8001", "--vllm-arg=--enforce-eager"])
    assert a.handler(a, {}) == 0
    argv, env = calls[0]
    assert argv[:3] == ["vllm", "serve", "RedHatAI/Qwen3.8-27B-INT4"]
    assert argv[-5:] == ["--enforce-eager", "--host", "127.0.0.1", "--port", "8001"]
    assert env["VLLM_ENFORCE_STRICT_TOOL_CALLING"] == "1"
    monkeypatch.setattr(srv.shutil, "which", lambda exe: None)
    assert a.handler(a, {}) == 127


def test_cmd_serve_auto_detects_the_profile(tmp_path, capsys):
    smi = tmp_path / "smi.txt"
    smi.write_text(_csv_gpus("NVIDIA H100 80GB HBM3", 81559, 4, driver="576.2"))
    p = _parser()
    a = p.parse_args(["local", "serve", "--nvidia-smi-file", str(smi), "--docker", "--dry-run"])
    assert a.handler(a, {}) == 0
    out = capsys.readouterr().out
    assert "auto-detected" in out and "vllm/vllm-openai:v0.31.0-cu129 Qwen/Qwen3.8-27B-FP8" in out
    assert "--data-parallel-size 4" in out and "--kv-cache-dtype fp8" in out
    smi.write_text("0, Tesla T4, 15360, 535.1\n")
    assert a.handler(a, {}) == 2
    assert "unrecognised GPU" in capsys.readouterr().err


def test_cmd_profiles_list_show_detect(tmp_path, capsys):
    p = _parser()
    a = p.parse_args(["local", "profiles"])
    assert a.handler(a, {}) == 0
    out = capsys.readouterr().out
    assert "deepseek-v4" in out and "local-rtxpro6000" in out and "recipe-verified" in out
    a = p.parse_args(["local", "profiles", "--show", "rtxpro6000"])
    assert a.handler(a, {}) == 0
    out = capsys.readouterr().out
    assert "alternative: nvidia/Qwen3.8-27B-NVFP4 [recipe-verified]" in out and "variant bulk:" in out
    assert "vllm serve RedHatAI/Qwen3.8-27B-NVFP4" in out
    smi = tmp_path / "smi.txt"
    smi.write_text(H200_TABLE)
    a = p.parse_args(["local", "profiles", "--nvidia-smi-file", str(smi)])
    assert a.handler(a, {}) == 0
    assert "Recommended: 1 x NVIDIA H200: profile h200" in capsys.readouterr().out
    a = p.parse_args(["local", "profiles", "--nvidia-smi-file", str(smi), "--json"])
    assert a.handler(a, {}) == 0
    assert json.loads(capsys.readouterr().out)["profile"] == "h200"
    smi.write_text("0, Tesla T4, 15360, 535.1\n")
    a = p.parse_args(["local", "profiles", "--nvidia-smi-file", str(smi)])
    assert a.handler(a, {}) == 1
    a = p.parse_args(["local", "profiles", "--show", "a100"])
    assert a.handler(a, {}) == 2


# ============================================================== fake OpenAI-compatible server

TRIAL_OK = {
    "nct_id": "NCT01234567", "overall_status": "Terminated", "primary_endpoint_result": "NEGATIVE",
    "primary_endpoint_rationale": "PFS HR 0.97, p = 0.74: not met.", "secondary_endpoint_result": "NEGATIVE",
    "secondary_endpoint_rationale": "OS p = 0.52.", "stop_reason_categories": ["Interim analysis", "Negative"],
    "stop_reason_text": "futility at interim", "serious_ae_pct": {"infections": 6.1, "gastrointestinal": 4.0,
                                                                  "cardiac": 1.2},
    "total_serious_ae_pct": 31.2, "results_source": "ClinicalTrials.gov", "ae_source": "ClinicalTrials.gov",
    "pubmed_ids": [], "source_urls": [], "tiers_consulted": ["ClinicalTrials.gov"], "confidence": "high",
    "notes": None,
}
FINDINGS_OK = {"gene": "TP53", "findings": [{"claim": "Tumour suppressor", "evidence_level": "strong"},
                                            {"claim": "Mutated in many cancers", "evidence_level": "moderate"}],
               "confidence": 0.8}
GENE_RE = re.compile(r"\b(" + "|".join(chk.GENES) + r")\b")


def _tok(text):
    return len(text) // 4 + 1 if text else 0


class FakeModel:
    """Scripted 'capable model' behind a fake vLLM; ``fail`` switches individual capabilities off."""

    def __init__(self):
        self.fail = set()
        self.requests = []
        self.seen_prompts = []
        self.lock = threading.Lock()
        self.version = "0.31.0"
        self.served = "qwen3.8-27b"
        self.max_model_len = 262144

    def respond(self, body):
        msgs = body.get("messages") or []
        texts = []
        for m in msgs:
            c = m.get("content")
            texts.append(c if isinstance(c, str) else json.dumps(c))
        prompt = "\n".join(texts)
        user = next((t for m, t in zip(reversed(msgs), reversed(texts)) if m.get("role") == "user"), "")
        tools = [t["function"]["name"] for t in body.get("tools") or []]
        choice = body.get("tool_choice")
        forced = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
        effort = body.get("reasoning_effort")
        if effort in ("high", "max", "minimal"):
            return 400, {"error": {"message": f"Unexpected reasoning effort {effort}", "type": "BadRequestError"}}
        off = effort == "none" or (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False
        thinking = not off or "reasoning_ignores_off" in self.fail
        max_tokens = int(body.get("max_tokens") or 1024)
        budget = body.get("thinking_token_budget")
        reasoning, content, calls, finish = "", "", [], "stop"
        if thinking:
            reasoning = "Let me think about this carefully. "
        if "prime number" in user and thinking:
            if "budget" in self.fail or not budget:
                reasoning = "x" * (max_tokens * 4)
                finish = "length"
            else:
                reasoning = "y" * (int(budget) * 4 - 4)
                content = "78"
        elif forced and "forced" not in self.fail:
            calls = [(forced, self.args_for(forced, user))]
        elif tools and "submit_result" in tools:
            calls = [("submit_result", self.args_for("submit_result", user))]
        elif tools and "record_findings" in tools:
            calls = [("record_findings", self.args_for("record_findings", user))]
        elif tools and "get_gene_info" in tools and "submit_answer" not in tools:
            genes = GENE_RE.findall(user)
            if "parallel" in self.fail:
                genes = genes[:1]
            calls = [("get_gene_info", json.dumps({"symbol": g})) for g in genes]
        else:
            m = re.search(r"access code for the archive vault is ([A-Z]+-\d{4})", prompt)
            if m and "access code" in user[-200:]:
                content = "UNKNOWN" if "needle" in self.fail else m.group(1)
            elif body.get("ignore_eos"):
                content = "lorem " * max_tokens
                finish = "length"
            elif "17 * 23" in user:
                content = "391"
            elif "one word" in prompt.lower():
                content = "Autumn"
            else:
                content = "Hello! I can look up genes."
        if calls:
            finish = "tool_calls"
        with self.lock:
            cached = 0
            for prev in self.seen_prompts:
                n = len(prompt) if prev == prompt else next(
                    (i for i, (a, b) in enumerate(zip(prev, prompt)) if a != b), min(len(prev), len(prompt)))
                cached = max(cached, (n // 4) // 16 * 16)
            self.seen_prompts.append(prompt)
        if "cache" in self.fail or cached < 64:
            cached = 0
        prompt_tokens = _tok(prompt) + 8
        cached = min(cached, prompt_tokens - 1)
        completion = (max_tokens if finish == "length" else
                      _tok(reasoning) + _tok(content) + sum(_tok(a) for _, a in calls))
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                 "total_tokens": prompt_tokens + completion, "prompt_tokens_details": {"cached_tokens": cached}}
        return 200, {"reasoning": reasoning, "content": content, "calls": calls, "finish": finish, "usage": usage}

    def args_for(self, name, user):
        if name == "submit_result":
            if "strict" in self.fail:
                return json.dumps({**TRIAL_OK, "nct_id": "NCT123", "confidence": "very high"})
            return json.dumps(TRIAL_OK)
        if name == "record_findings":
            if "malformed" in self.fail:
                return '{"gene": "TP53", "findings": [{"claim": "x", '
            gene = (GENE_RE.findall(user) or ["TP53"])[0]
            return json.dumps({**FINDINGS_OK, "gene": gene})
        if name == "submit_answer":
            return json.dumps({"answer": "Hello", "confidence": "high"})
        if name == "get_gene_info":
            return json.dumps({"symbol": "TP53"})
        return "{}"


def _handler_for(model):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, status, payload):
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                if "health" in model.fail:
                    return self._json(503, {"error": "loading"})
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [
                    {"id": model.served, "object": "model", "root": "Qwen/Qwen3.8-27B-FP8",
                     "max_model_len": model.max_model_len, "owned_by": "vllm"}]})
            if self.path == "/version":
                return self._json(200, {"version": model.version})
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            with model.lock:
                model.requests.append(body)
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": {"message": "not found"}})
            if body.get("model") != model.served:
                return self._json(404, {"error": {"message": f"The model `{body.get('model')}` does not exist.",
                                                  "type": "NotFoundError", "code": 404}})
            status, r = model.respond(body)
            if status != 200:
                return self._json(status, r)
            tool_calls = [{"id": f"chatcmpl-tool-{i}", "type": "function",
                           "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(r["calls"])]
            if not body.get("stream"):
                msg = {"role": "assistant", "content": r["content"] or None, "reasoning": r["reasoning"] or None,
                       "tool_calls": tool_calls}
                return self._json(200, {"id": "chatcmpl-1", "object": "chat.completion", "model": model.served,
                                        "choices": [{"index": 0, "message": msg, "finish_reason": r["finish"]}],
                                        "usage": r["usage"]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def send(delta, finish=None, usage=None):
                chunk = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": model.served,
                         "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage:
                    chunk["usage"] = usage
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

            send({"role": "assistant", "content": ""})
            if r["reasoning"]:
                send({"reasoning": r["reasoning"]})
            if r["content"]:
                half = len(r["content"]) // 2
                send({"content": r["content"][:half]})
                send({"content": r["content"][half:]})
            for i, (n, a) in enumerate(r["calls"]):
                send({"tool_calls": [{"index": i, "id": f"chatcmpl-tool-{i}", "type": "function",
                                      "function": {"name": n, "arguments": ""}}]})
                send({"tool_calls": [{"index": i, "function": {"arguments": a}}]})
            send({}, finish=r["finish"])
            send({}, usage=r["usage"])
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


@pytest.fixture
def fake_server():
    model = FakeModel()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(model))
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    model.base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    yield model
    httpd.shutdown()
    httpd.server_close()


class StubProvider:
    """A minimal non-streaming OpenAI client with the adapter's interface (prepare / complete /
    aclose), so the check logic is tested independently of vbt.providers.openai_compat."""

    def __init__(self, opts, extra_body=None):
        self.api, self.root = chk.split_base_url(opts.base_url)
        self.model = opts.model
        self.name = opts.provider_name
        self.extra_body = dict(extra_body or {})

    async def prepare(self):
        async with httpx.AsyncClient(trust_env=False, timeout=30) as c:
            if (await c.get(self.root + "/health")).status_code != 200:
                raise ProviderError("server not ready")
            ids = [d["id"] for d in (await c.get(self.api + "/models")).json()["data"]]
        if self.model not in ids:
            raise ProviderError(f"model {self.model!r} is not served; served: {ids}")

    async def complete(self, *, settings, system, messages, tools, on_text=None, on_thinking=None):
        off = not settings.thinking or settings.effort is None
        effort = "none" if off else {"high": "xhigh", "max": "xhigh"}.get(settings.effort, settings.effort)
        extra = settings.extra or {}
        body = {"model": self.model or settings.model, "max_tokens": settings.max_tokens, "stream": False,
                "reasoning_effort": effort,
                "messages": [{"role": "system", "content": system_text(system)}]
                + [{"role": m.role, "content": m.text} for m in messages]}
        if not off and extra.get("thinking_budget"):
            body["thinking_token_budget"] = extra["thinking_budget"]
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.input_schema,
                **({"strict": True} if getattr(t, "strict", False) else {})}} for t in tools]
            tc = extra.get("tool_choice")
            body["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}} if isinstance(tc, dict) \
                else (tc or "auto")
        body.update(self.extra_body)
        async with httpx.AsyncClient(trust_env=False, timeout=60) as c:
            r = await c.post(self.api + "/chat/completions", json=body)
        if r.status_code >= 400:
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        msg, finish = data["choices"][0]["message"], data["choices"][0]["finish_reason"]
        blocks = []
        if msg.get("reasoning"):
            blocks.append(ThinkingBlock(msg["reasoning"], provider=self.name))
            if on_thinking:
                on_thinking(msg["reasoning"])
        if msg.get("content"):
            blocks.append(TextBlock(msg["content"]))
            if on_text:
                on_text(msg["content"])
        for tc in msg.get("tool_calls") or []:
            raw = tc["function"]["arguments"]
            try:
                args, native = json.loads(raw), None
                if not isinstance(args, dict):
                    raise ValueError("not an object")
            except ValueError as exc:
                args, native = {}, {"invalid_arguments": raw, "error": str(exc)}
            blocks.append(ToolCall(tc["id"], tc["function"]["name"], args, native=native))
        u = data.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        usage = Usage(input_tokens=u.get("prompt_tokens", 0) - cached, output_tokens=u.get("completion_tokens", 0),
                      cache_read_tokens=cached)
        stop = StopReason.TOOL_USE if msg.get("tool_calls") else \
            {"stop": StopReason.END_TURN, "length": StopReason.MAX_TOKENS}.get(finish, StopReason.OTHER)
        return ModelResponse(Message("assistant", blocks), stop, usage, data.get("model", ""))

    async def aclose(self):
        return None


def _stub_factory(opts, extra_body=None):
    return StubProvider(opts, extra_body)


@pytest.fixture(params=["stub", "adapter"])
def factory(request):
    """The check runs through a stub client and, when installed, the real OpenAI-compatible adapter."""
    if request.param == "adapter":
        pytest.importorskip("vbt.providers.openai_compat")
        return None   # run_checks' default: vbt.providers.openai_compat.OpenAICompatProvider
    return _stub_factory


def _opts(server, **kw):
    base = dict(base_url=server.base_url, model="qwen3.8-27b", family="qwen3_8", needle_tokens=(1500,),
                prefix_tokens=1200, thinking_budget=64, concurrency=3, output_tokens=16, trials=4,
                min_tokens_per_s=1.0, check_timeout_s=60.0, read_timeout_s=60.0, seed=7,
                min_context_tokens=262144)
    base.update(kw)
    return chk.CheckOptions(**base)


def _statuses(report):
    return {c.name: c.status for c in report.checks}


def test_check_passes_against_a_capable_server(fake_server, factory):
    report = asyncio.run(chk.run_checks(_opts(fake_server), factory))
    st = _statuses(report)
    assert st == {n: "pass" for n in ["health", "models", "version", "prepare", "tool_call", "parallel_tool_calls",
                                      "strict_schema", "forced_tool_choice", "reasoning_effort", "thinking_budget",
                                      "needle_1k", "prefix_cache", "throughput", "malformed_calls"]}, \
        [(c.name, c.status, c.detail) for c in report.checks if c.status != "pass"]
    assert report.ok and report.model == "qwen3.8-27b" and report.server["version"] == "0.31.0"
    assert report.server["max_model_len"] == 262144
    strict = report.result("strict_schema")
    assert strict.metrics["schema"] == {"anyOf": 13, "$ref": 7, "pattern": 2}
    assert strict.metrics["forced_retry"] is False
    assert report.result("parallel_tool_calls").metrics["symbols"] == ["TP53", "BRCA1", "EGFR"]
    assert report.result("prefix_cache").metrics["second"]["cached_tokens"] > 0
    chats = [b for b in fake_server.requests if "messages" in b]
    # what reached the server: strict submit_result, forced tool_choice, effort tiers, budgets, ignore_eos
    submit = [t for b in chats for t in b.get("tools") or [] if t["function"]["name"] == "submit_result"]
    assert submit and all(t["function"].get("strict") is True for t in submit)
    assert any(b.get("tool_choice") == {"type": "function", "function": {"name": "submit_answer"}} for b in chats)
    efforts = {b.get("reasoning_effort") for b in chats}
    assert {"xhigh", "medium", "none"} <= efforts and not efforts & {"high", "max", "minimal"}
    assert any(b.get("thinking_token_budget") == 64 for b in chats)
    assert sum(1 for b in chats if b.get("ignore_eos") is True) == 3


@pytest.mark.parametrize("fail,check", [
    ("parallel", "parallel_tool_calls"),
    ("budget", "thinking_budget"),
    ("cache", "prefix_cache"),
    ("strict", "strict_schema"),
    ("forced", "forced_tool_choice"),
    ("reasoning_ignores_off", "reasoning_effort"),
    ("needle", "needle_1k"),
    ("malformed", "malformed_calls"),
])
def test_check_reports_the_failing_capability(fake_server, factory, fail, check):
    fake_server.fail = {fail}
    report = asyncio.run(chk.run_checks(_opts(fake_server), factory))
    st = _statuses(report)
    assert st[check] == "fail", report.result(check)
    others = {n: s for n, s in st.items() if n != check and s == "fail"}
    assert not others, [(n, report.result(n).detail) for n in others]
    assert not report.ok


def test_check_failure_details(fake_server):
    fake_server.fail = {"strict", "malformed", "budget", "cache"}
    report = asyncio.run(chk.run_checks(_opts(fake_server), _stub_factory))
    strict = report.result("strict_schema")
    assert "nct_id" in strict.detail and "confidence" in " ".join(strict.metrics["errors"])
    mal = report.result("malformed_calls")
    assert mal.metrics["breakdown"] == {"invalid_json": 4} and mal.metrics["rate"] == 1.0
    assert "thinking_token_budget is not enforced" in report.result("thinking_budget").detail
    assert "--enable-prompt-tokens-details" in report.result("prefix_cache").detail


def test_check_server_level_problems(fake_server):
    fake_server.version = "0.30.0"
    report = asyncio.run(chk.run_checks(_opts(fake_server, only=("health", "models", "version")), _stub_factory))
    assert _statuses(report) == {"health": "pass", "models": "pass", "version": "warn"}
    assert report.ok and "--tool-strict-level is unavailable" in report.result("version").detail
    report = asyncio.run(chk.run_checks(_opts(fake_server, only=("models",), min_context_tokens=300000),
                                        _stub_factory))
    assert _statuses(report)["models"] == "fail" and "lower models.<tier>.context_window_tokens" in \
        report.result("models").detail
    fake_server.fail = {"health"}
    report = asyncio.run(chk.run_checks(_opts(fake_server, only=("health",)), _stub_factory))
    assert _statuses(report) == {"health": "fail"} and "HTTP 503" in report.result("health").detail


def test_check_model_not_served_skips_model_checks(fake_server, factory):
    report = asyncio.run(chk.run_checks(_opts(fake_server, model="qwen-other"), factory))
    st = _statuses(report)
    assert st["models"] == "fail" and "is not served; served: ['qwen3.8-27b']" in report.result("models").detail
    assert (st["prepare"], st["tool_call"], st["throughput"], st["version"]) == ("skip", "skip", "skip", "pass")
    assert "qwen-other" in report.result("tool_call").detail


def test_check_discovers_the_served_model(fake_server):
    report = asyncio.run(chk.run_checks(_opts(fake_server, model=None, only=("tool_call",)), _stub_factory))
    assert report.model == "qwen3.8-27b" and _statuses(report) == {"tool_call": "pass"}


def test_check_unreachable_server_skips_everything():
    report = asyncio.run(chk.run_checks(chk.CheckOptions(base_url="http://127.0.0.1:9/v1", model="m",
                                                         check_timeout_s=10, timeout_s=2), _stub_factory))
    st = _statuses(report)
    assert st["health"] == "fail" and "vbt local serve" in report.result("health").detail
    assert set(st.values()) == {"fail", "skip"} and st["tool_call"] == "skip"


def test_check_throughput_threshold_and_long_context(fake_server):
    report = asyncio.run(chk.run_checks(_opts(fake_server, only=("throughput",), min_tokens_per_s=1e12),
                                        _stub_factory))
    tp = report.result("throughput")
    assert tp.status == "fail" and tp.metrics["output_tokens"] == 3 * 16 and tp.metrics["errors"] == 0
    fake_server.max_model_len = 4096
    report = asyncio.run(chk.run_checks(_opts(fake_server, only=("needle",), needle_tokens=(1500, 200000),
                                              min_context_tokens=None), _stub_factory))
    assert _statuses(report) == {"needle_1k": "pass", "needle_195k": "skip"}


def test_check_report_files_and_markdown(fake_server, tmp_path):
    report = asyncio.run(chk.run_checks(_opts(fake_server, only=("health", "tool_call")), _stub_factory))
    jp, mp = chk.write_report(tmp_path, report.as_dict(), report.markdown(), "check")
    data = json.loads(jp.read_text())
    assert data["ok"] is True and data["counts"]["pass"] == 2 and "api_key" not in data["options"]
    md = mp.read_text()
    assert "| tool_call | pass |" in md and "Result: **PASS**" in md and "## Metrics" in md


def test_options_from_config(monkeypatch):
    monkeypatch.delenv("VBT_LLM_BASE_URL", raising=False)
    cfg = load_config(["local-h100"])
    o = chk.options_from_config(cfg)
    assert (o.base_url, o.model, o.family, o.provider_name, o.min_context_tokens, o.read_timeout_s) == \
        ("http://localhost:8000/v1", "qwen3.8-27b", "qwen3_8", "vllm", 262144, 900.0)
    p = _parser()
    a = p.parse_args(["local", "check", "--base-url", "http://h:1/v1", "--served-model", "x", "--only",
                      "tool_call,needle", "--long-context", "--needle-tokens", "8192", "--min-tps", "5"])
    o = chk.options_from_config(cfg, a)
    assert (o.base_url, o.model, o.only, o.needle_tokens, o.min_tokens_per_s) == \
        ("http://h:1/v1", "x", ("tool_call", "needle"), (8192, 200000), 5.0)
    monkeypatch.setenv("VBT_LLM_BASE_URL", "http://env:8000/v1")
    o = chk.options_from_config({"provider": {"name": "anthropic", "options": {"base_url": "https://api"}}})
    assert (o.base_url, o.model, o.min_context_tokens) == ("http://env:8000/v1", None, None)
    a = p.parse_args(["local", "check", "--only", "nonsense"])
    with pytest.raises(ValueError, match="unknown check"):
        chk.options_from_config(cfg, a)
    assert chk.split_base_url("http://h:8000") == ("http://h:8000/v1", "http://h:8000")
    assert chk.split_base_url("http://h:8000/v1/chat/completions/") == ("http://h:8000/v1", "http://h:8000")


def test_cmd_check_and_bench_cli(fake_server, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(chk, "default_provider_factory", _stub_factory)
    cfg = load_config(["local-h100"], overrides={"provider": {"options": {"base_url": fake_server.base_url}}})
    p = _parser()
    a = p.parse_args(["local", "check", "--only", "health,models,tool_call", "--out", str(tmp_path / "c")])
    assert a.handler(a, cfg) == 0
    out = capsys.readouterr().out
    assert "PASS  tool_call" in out and "PASS: 3 passed" in out
    assert (tmp_path / "c" / "check.json").is_file() and (tmp_path / "c" / "check.md").is_file()
    fake_server.fail = {"parallel"}
    a = p.parse_args(["local", "check", "--only", "parallel_tool_calls", "--out", str(tmp_path / "c2"), "--json"])
    assert a.handler(a, cfg) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    a = p.parse_args(["local", "check", "--only", "bogus"])
    assert a.handler(a, cfg) == 2
    capsys.readouterr()
    a = p.parse_args(["local", "bench", "--levels", "1,4", "--output-tokens", "8", "--prompt-tokens", "64",
                      "--out", str(tmp_path / "b")])
    assert a.handler(a, cfg) == 0
    out = capsys.readouterr().out
    assert "| Concurrency |" in out and "\n| 4 | 0 | 32 |" in out
    bench = json.loads((tmp_path / "b" / "bench.json").read_text())
    assert [r["concurrency"] for r in bench["levels"]] == [1, 4] and bench["ok"] is True
    assert all(r["tokens_per_s"] > 0 for r in bench["levels"])
    # default output directory: <runs_dir>/local/<kind>-<time>
    cfg2 = {**cfg, "paths": {"runs_dir": str(tmp_path / "runs")}}
    a = p.parse_args(["local", "check", "--only", "health"])
    assert a.handler(a, cfg2) == 0
    assert list((tmp_path / "runs" / "local").glob("check-*/check.md"))


def test_bench_against_the_adapter(fake_server):
    pytest.importorskip("vbt.providers.openai_compat")
    res = asyncio.run(chk.run_bench(_opts(fake_server), [1, 2], output_tokens=8, prompt_tokens=64))
    assert res["ok"] and [r["output_tokens"] for r in res["levels"]] == [8, 16]
    assert res["levels"][1]["p50_ttft_s"] is not None      # streamed: time to first token measured


def test_haystack_is_deterministic_and_sized():
    import random
    a = chk.make_haystack(1000, random.Random(1), needle="NEEDLE", depth=0.5)
    b = chk.make_haystack(1000, random.Random(1), needle="NEEDLE", depth=0.5)
    assert a == b and 3800 <= len(a) <= 4600 and 0.3 < a.index("NEEDLE") / len(a) < 0.7
    assert "access code" not in chk.make_haystack(5000, random.Random(2))
