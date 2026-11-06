"""LLM settings injected by the caller (T1-28 S2c, D25): LLMConfig, the provider table and the Settings shim.

The module reads no .env, environment variable or configs/model.yaml for the model. A provider other than mock
without a key fails when an LLM step first needs the client (LLMKeyMissing), so rule-only answers need no key.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from eval import run_eval
from kosis_chat import api
from kosis_chat.config import Settings, load_settings, settings_from_llm
from kosis_chat.data.loader import load_table
from kosis_chat.llm.model import HCXClient, LLMClient, LLMResult, MockClient
from kosis_chat.llm.model import build_client as build_from_settings
from kosis_chat.llm.providers import PROVIDERS, build_client, register_provider
from kosis_chat.pipeline import StatPipeline
from kosis_chat.schemas.resources import DEFAULT_MODEL, MOCK_RESPONSE, LLMConfig, LLMKeyMissing

from kosis_ai import paths  # isort: skip

ROOT = Path(__file__).resolve().parents[1]
KEY = "nv-test-secret-1234"
ANSWERED = "2023년 서울특별시 1인 가구 수는?"  # needs the model (compose)
REFUSED = "2030년 서울 1인 가구 수는?"  # the rules refuse it (no such period)
CLARIFY = "2023년 중구 1인 가구는?"  # the rules ask back (서울특별시 중구 / 부산광역시 중구)

# configs/model.yaml as it was before T1-28 S2c removed it (its `model` block and the other keys).
OLD_MODEL_YAML = {
    "backend": "hcx",
    "prompt_version": "baseline-v1",
    "serialization": "long",
    "model": {"name": "HCX-DASH-002", "thinking_effort": "none", "temperature": 0, "max_tokens": 1024,
              "max_input_tokens": None, "timeout_seconds": 60, "base_url": "https://clovastudio.stream.ntruss.com",
              "api_key_env": "HCX_API_KEY"},
    "mock": {"response": "[mock] 표에 있는 값만으로 답변합니다. 출처: KOSIS"},
    "eval": {"runs": 1},
}
# load_settings() as it returned from that file (printed 2026-11-05, before the file was removed).
OLD_LOAD_SETTINGS = {
    "backend": "hcx",
    "prompt_version": "baseline-v1",
    "serialization": "long",
    "model_name": "HCX-DASH-002",
    "thinking_level": "none",
    "temperature": 0.0,
    "max_tokens": 1024,
    "max_input_tokens": None,
    "timeout_seconds": 60.0,
    "base_url": "https://clovastudio.stream.ntruss.com",
    "api_key_env": "HCX_API_KEY",
    "mock_response": "[mock] 표에 있는 값만으로 답변합니다. 출처: KOSIS",
    "eval_runs": 1,
}
# The manifest "model" block of a mock run (eval/run_eval.py, --backend mock), as in the S0 parity base run.
OLD_MOCK_MODEL_PARAMS = {
    "backend": "mock", "model_name": "HCX-DASH-002", "thinking_level": "none", "temperature": 0.0, "max_tokens": 1024,
    "max_input_tokens": None, "timeout_seconds": 60.0, "base_url": "https://clovastudio.stream.ntruss.com",
}


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("HCX_API_KEY", raising=False)


@pytest.fixture
def households(household_path):
    return load_table(household_path)


@pytest.fixture
def workspace(data_dir, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(data_dir.parent))
    api._cached.cache_clear()
    api._cached_llm.cache_clear()
    yield data_dir.parent
    api._cached.cache_clear()
    api._cached_llm.cache_clear()


def reply(answer):
    return json.dumps({"answer": answer}, ensure_ascii=False)


# ---------------------------------------------------------------- defaults = the removed configs/model.yaml


def test_the_settings_file_is_gone():
    assert not (ROOT / "configs" / "model.yaml").exists() and not (ROOT / "configs").exists()


@pytest.mark.parametrize("field", list(OLD_LOAD_SETTINGS))
def test_load_settings_returns_what_the_file_gave(field):
    assert getattr(load_settings(), field) == OLD_LOAD_SETTINGS[field]
    assert type(getattr(load_settings(), field)) is type(OLD_LOAD_SETTINGS[field])


def test_load_settings_has_no_other_fields_and_keeps_overrides():
    assert dataclasses.asdict(load_settings()) == OLD_LOAD_SETTINGS
    assert load_settings(backend="mock").model_params() == OLD_MOCK_MODEL_PARAMS  # the mock run manifest
    assert load_settings(model_name="HCX-007", max_tokens=None).model_name == "HCX-007"
    assert load_settings(max_tokens=None).max_tokens == 1024  # None means no override, as before


def test_llm_config_defaults_are_the_model_block():
    block = OLD_MODEL_YAML["model"]
    config = LLMConfig(model=DEFAULT_MODEL)
    assert (config.model, config.provider, config.api_key) == (block["name"], OLD_MODEL_YAML["backend"], None)
    assert config.thinking_effort == block["thinking_effort"]
    assert config.temperature == block["temperature"] and config.max_tokens == block["max_tokens"]
    assert config.max_input_tokens == block["max_input_tokens"] and config.timeout_seconds == block["timeout_seconds"]
    assert config.base_url == block["base_url"]
    assert MOCK_RESPONSE == OLD_MODEL_YAML["mock"]["response"]


# ---------------------------------------------------------------- the key is checked on first use, not when made


def test_no_key_rule_only_answers(households):
    bot = StatPipeline(households, llm=LLMConfig(model=DEFAULT_MODEL), org_id="101")  # no key, provider hcx
    refused = bot.ask(REFUSED)
    assert refused.status == "ok" and refused.stat_answer.outcome == "refused"
    asked = bot.ask(CLARIFY)
    assert asked.stat_answer.outcome == "clarify" and len(asked.stat_answer.clarify_options) == 2
    assert bot._client is None  # no client was built


def test_no_key_llm_step_raises(households):
    bot = StatPipeline(households, llm=LLMConfig(model=DEFAULT_MODEL), org_id="101")
    with pytest.raises(LLMKeyMissing, match="api_key is required for provider='hcx'"):
        bot.ask(ANSWERED)
    assert issubclass(LLMKeyMissing, RuntimeError) and issubclass(LLMKeyMissing, ValueError)


def test_no_key_settings_path_is_lazy_too(households):
    """StatPipeline(table) with the default Settings (backend hcx) no longer fails when made."""
    bot = StatPipeline(households, org_id="101")
    assert bot.ask(REFUSED).stat_answer.outcome == "refused"
    with pytest.raises(LLMKeyMissing, match="HCX_API_KEY"):
        bot.ask(ANSWERED)


def test_no_key_summarize_stat(workspace):
    config = LLMConfig(model=DEFAULT_MODEL)
    assert api.summarize_stat(REFUSED, "101_FAKE_TBL_002", llm=config).outcome == "refused"
    with pytest.raises(RuntimeError):  # LLMKeyMissing: summarize_stat's "failed model call" contract
        api.summarize_stat(ANSWERED, "101_FAKE_TBL_002", llm=config)


def test_explicit_mock_needs_no_key(households, mock_llm):
    bot = StatPipeline(households, llm=mock_llm, org_id="101")
    answer = bot.ask(ANSWERED)
    assert answer.status == "ok" and answer.stat_answer.outcome == "answered"
    assert isinstance(bot.client, MockClient) and bot.client.calls  # the mock reply is not JSON: template
    assert answer.raw_responses[0] == MOCK_RESPONSE
    assert isinstance(build_client(mock_llm), MockClient)


def test_a_given_client_wins(households, mock_llm):
    client = MockClient([reply("서울은 {f1}입니다.")])
    bot = StatPipeline(households, client=client, llm=LLMConfig(model=DEFAULT_MODEL), org_id="101")
    assert bot.ask(ANSWERED).decision_metadata["compose"] == "model" and bot.client is client


# ---------------------------------------------------------------- old Settings callers


def test_summarize_stat_with_settings(workspace, mock_settings):
    stat = api.summarize_stat(ANSWERED, "101_FAKE_TBL_002", settings=mock_settings)
    assert stat.outcome == "answered" and stat.source.tbl_id == "FAKE_TBL_002"


def test_hcx_client_from_settings_and_from_config_send_the_same_request(mock_settings, server_factory, monkeypatch):
    body = {"status": {"code": "20000"}, "result": {"message": {"role": "assistant", "content": "ok"}}}
    srv = server_factory(body=body)
    settings = dataclasses.replace(mock_settings, backend="hcx", base_url=srv.base_url, model_name="HCX-007",
                                   thinking_level="low", max_tokens=5120)
    monkeypatch.setenv("HCX_API_KEY", KEY)  # the Settings path reads the shell environment variable, as before
    old = HCXClient(settings)
    assert old.settings is settings and old.config == LLMConfig.from_settings(settings, api_key=KEY)
    assert old.generate("s", "u").text == "ok"
    monkeypatch.delenv("HCX_API_KEY")
    new = HCXClient(LLMConfig.from_settings(settings, api_key=KEY))  # the key is passed, the env is not read
    assert new.generate("s", "u").text == "ok"
    assert srv.requests[0] == srv.requests[1] and new.url == old.url
    assert [h["Authorization"] for h in srv.request_headers] == [f"Bearer {KEY}"] * 2
    assert new.settings == settings_from_llm(new.config)  # Phase 2 reads client.settings


def test_hcx_client_without_a_key(mock_settings):
    with pytest.raises(LLMKeyMissing, match="HCX_API_KEY"):
        HCXClient(dataclasses.replace(mock_settings, backend="hcx"))
    with pytest.raises(LLMKeyMissing):
        HCXClient(LLMConfig(model=DEFAULT_MODEL))
    with pytest.raises(LLMKeyMissing):
        build_client(LLMConfig(model=DEFAULT_MODEL, api_key=""))


def test_build_client_takes_settings_or_config(mock_settings, monkeypatch):
    assert build_from_settings(mock_settings)._response == mock_settings.mock_response
    assert isinstance(build_client(mock_settings), MockClient)
    monkeypatch.setenv("HCX_API_KEY", KEY)
    assert isinstance(build_from_settings(LLMConfig(model=DEFAULT_MODEL, api_key=KEY)), HCXClient)


# ---------------------------------------------------------------- provider table


def test_provider_table(monkeypatch):
    assert PROVIDERS["hcx"] is HCXClient and PROVIDERS["mock"] is MockClient

    class Echo(LLMClient):
        name = "echo"

        def __init__(self, config: LLMConfig) -> None:
            self.config = config

        def generate(self, system: str, user: str) -> LLMResult:
            return LLMResult(text=user)

    monkeypatch.setattr("kosis_chat.llm.providers.PROVIDERS", dict(PROVIDERS))
    from kosis_chat.llm import providers

    assert register_provider("echo", Echo) is Echo and providers.PROVIDERS["echo"] is Echo
    client = build_client(LLMConfig(provider="echo", model="m", api_key=KEY))
    assert isinstance(client, Echo) and client.generate("s", "u").text == "u"
    with pytest.raises(LLMKeyMissing, match="provider='echo'"):
        build_client(LLMConfig(provider="echo", model="m"))
    with pytest.raises(ValueError, match="unknown provider"):
        build_client(LLMConfig(provider="nope", model="m", api_key=KEY))
    with pytest.raises(TypeError):
        register_provider("bad", object)
    assert "echo" not in PROVIDERS  # the module table was swapped for the test


# ---------------------------------------------------------------- the key never shows


def test_secret_is_not_in_repr_logs_or_dumps(caplog):
    config = LLMConfig(api_key=KEY, model=DEFAULT_MODEL)
    assert isinstance(config.api_key, SecretStr) and config.require_key() == KEY
    with caplog.at_level(logging.INFO, logger="kosis.test"):
        logging.getLogger("kosis.test").info("config %s %r %s", config, config, config.model_dump())
    for text in (repr(config), str(config), str(config.model_dump()), config.model_dump_json(),
                 json.dumps(config.model_dump(mode="json")), caplog.text):
        assert KEY not in text
    assert config.model_dump(mode="json")["api_key"] == "**********"


def test_config_is_frozen_and_strict_about_names():
    config = LLMConfig(model=DEFAULT_MODEL)
    with pytest.raises(ValidationError):
        config.model = "HCX-007"
    with pytest.raises(ValidationError):
        LLMConfig(model=DEFAULT_MODEL, api_key_env="HCX_API_KEY")
    hash(LLMConfig(api_key=KEY, model=DEFAULT_MODEL))  # api._cached_llm keys on it


# ---------------------------------------------------------------- from_settings


def test_from_settings_matches_field_by_field():
    for settings in (load_settings(), dataclasses.replace(load_settings(), model_name="HCX-007", thinking_level="low",
                                                          max_tokens=5120, max_input_tokens=9000, temperature=0.5,
                                                          timeout_seconds=30.0, base_url="http://x", backend="mock")):
        config = LLMConfig.from_settings(settings)
        assert config.api_key is None  # never read from the environment here
        assert (config.provider, config.model, config.thinking_effort) == (settings.backend, settings.model_name,
                                                                             settings.thinking_level)
        assert (config.temperature, config.max_tokens, config.max_input_tokens, config.timeout_seconds,
                config.base_url) == (settings.temperature, settings.max_tokens, settings.max_input_tokens,
                                     settings.timeout_seconds, settings.base_url)
        assert settings_from_llm(config) == settings  # and back
    assert LLMConfig.from_settings(load_settings(), api_key=KEY).require_key() == KEY
    assert isinstance(settings_from_llm(LLMConfig(model=DEFAULT_MODEL)), Settings)


# ---------------------------------------------------------------- run_eval reads the key from the shell at start


def test_run_eval_hcx_needs_the_shell_key(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        run_eval.main(["--stage", "x", "--backend", "hcx", "--golden", str(tmp_path / "none.jsonl"),
                       "--out", str(tmp_path / "r"), "--log-dir", str(tmp_path / "logs")])
    assert exc.value.code == 2 and "HCX_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "logs").exists()  # stopped before anything ran


def test_run_eval_llm_config(monkeypatch):
    assert run_eval.llm_config(load_settings(backend="mock")).provider == "mock"
    with pytest.raises(KeyError):
        run_eval.llm_config(load_settings())
    monkeypatch.setenv("HCX_API_KEY", KEY)
    assert run_eval.llm_config(load_settings()).require_key() == KEY


# --- S2c review fixes --------------------------------------------------------------------------------------------


def test_a_missing_settings_file_gives_the_defaults_not_mock():
    from dataclasses import asdict

    from kosis_chat import config

    with pytest.deprecated_call():
        old_path = config.load_settings("configs/model.yaml")  # the deleted file
    assert asdict(old_path) == asdict(config.load_settings())


@pytest.mark.parametrize("key", ["", "   "])
def test_an_empty_key_is_no_key(key):
    from kosis_chat.schemas.resources import LLMConfig, LLMKeyMissing

    with pytest.raises(LLMKeyMissing):
        LLMConfig(model="HCX-DASH-002", api_key=key).require_key()


def test_cached_cache_clear_also_clears_the_llm_pipelines():
    from kosis_chat import api

    api._cached_llm.cache_clear()
    before = api._cached_llm.cache_info()
    api._cached.cache_clear()
    assert api._cached_llm.cache_info().currsize == 0
    assert api._cached.cache_info().currsize == 0
    assert before.maxsize == api._cached_llm.cache_info().maxsize
