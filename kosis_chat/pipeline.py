"""C pipeline: numbers by code, sentences by the model (docs/plans/20261024-c-compute-pipeline/PLAN.md).

judge (rules) -> analyze (rules, the model fills what they leave open) -> compute -> compose (model, {f1} citations)
-> verify (one retry, then a code template) -> render. StatPipeline.ask returns the same Answer as Chatbot.ask, with
the finished StatAnswer in Answer.stat_answer. A question neither the rules nor the fill can place is planned by the
model as v3 blocks (analyze_plan, S5b; code still computes), and only when that fails too goes to the compact-v2
chatbot. A plan that names something several codes could be asks back (outcome clarify).
S6: the answer also records its caveats (shown as '참고:' lines), the facts shown beside it (context_fact_ids: the
other scenarios of a projection the question did not choose, a newer period), a table of targets by periods, and an
ask-back's options with their codes (clarify_choices).
T1-28 S3a/S3b: every step is a stage node (judge/ analyze/ compute/ compose/ chart/ render/ node.py, legacy/fallback.py:
step_<name>(In, StatResources) -> Out). The model steps get a ModelAccess (llm=) and return their calls as LLMCall
records; the Answer and its spans are made from the records (graph/legacy.py, graph/spans.py).
T1-28 S4: ask runs the steps as a LangGraph graph (graph/build.py: node table, conditional edges, the chart branch
beside compose, the compose <-> verify loop); the manual router it replaced is kept only as the reference of
tests/test_graph_parity.py.
"""

from __future__ import annotations

import time
from dataclasses import replace

import kosis_chat.shared.patterns  # noqa: F401  first: puts src/ on the path when kosis_ai is not installed (S6 removes)
from kosis_chat.analyze.rules import build_plan
from kosis_chat.config import Settings, load_settings, settings_from_llm
from kosis_chat.graph.build import StatContext, build_graph, run
from kosis_chat.graph.legacy import FALLBACK_PROMPT, Chatbot, add_usage, answer_of  # noqa: F401  add_usage: old name
from kosis_chat.graph.spans import llm_attributes
from kosis_chat.llm import compose  # the old path kept: pipeline.VERSION is kosis_chat.llm.compose.VERSION (Phase 2)
from kosis_chat.llm.calls import ModelAccess
from kosis_chat.llm.model import STATUS_API_ERROR, STATUS_INPUT_TOO_LONG, STATUS_TIMEOUT, LLMClient
from kosis_chat.llm.providers import build_client
from kosis_chat.prepare.schemas import TableInput
from kosis_chat.render.node import causal_note
from kosis_chat.schemas.answer import Answer
from kosis_chat.schemas.resources import LLMConfig, MultiResources, StatResources, load_resources

VERSION = compose.VERSION
# Kept by name for callers of the old pipeline module (T1-28 S3b): FALLBACK_PROMPT is legacy/fallback.py's,
# add_usage graph/legacy.py's, _llm_attributes graph/spans.py llm_attributes.
_llm_attributes = llm_attributes
_causal_note = causal_note  # T1-28 S3a: moved to render/node.py (the same function)


class StatPipeline:
    """Answers questions about one table; same attributes run_eval reads from a Chatbot.

    Deprecated as an entry point (T1-28 S6): callers outside kosis_chat use kosis_chat.api.StatModule (a loaded
    table) or summarize_stat / summarize_stat_run (a tables.csv key), which build this class. It stays, with the same
    arguments and attributes, for eval/run_eval.py (decision_metadata, attempts) and api._build (Phase 2); no
    warning is raised. pipeline.VERSION stays."""

    def __init__(
        self,
        table: TableInput,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        org_id: str | None = None,
        retry_interval: float = 0.0,
        fallback_scope: str = "auto",
        llm: LLMConfig | None = None,
    ) -> None:
        # llm (T1-28 S2c): the injected model; a Settings alone is the Phase 2 shim (its key from the shell env).
        self.settings = settings or (settings_from_llm(llm) if llm is not None else load_settings())
        self.llm = llm
        # Built on first use by an LLM step (LLMKeyMissing then, not here): rule-only answers need no key.
        self._client: LLMClient | None = client
        # The cleaned table, its index and lookups (T1-28 S2): the index is the one compute's checks use too.
        self.resources: StatResources = load_resources(table, org_id)
        self.table = self.resources.table
        self.index = self.resources.index
        self.org_id = org_id
        self.retry_interval = retry_interval
        self.prompt_version = VERSION
        self.system_prompt = compose.SYSTEM_PROMPT
        self.answer_model = compose.ComposeOutput
        self.decision_metadata = {"pipeline": "stat", "outcome_origin": "rules", "numbers_origin": "code",
                                  "defaults_origin": "rules"}
        self.fallback_scope = fallback_scope
        self._fallback: Chatbot | None = None
        self._graph = None

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = build_client(self.llm if self.llm is not None else self.settings)
        return self._client

    @client.setter
    def client(self, value: LLMClient) -> None:  # run_eval swaps in its recording client
        self._client = value
        if self._fallback is not None:
            self._fallback.client = value

    @property
    def fallback(self) -> Chatbot:
        """compact-v2 with the same rules, for the questions the analysis cannot place (built on first use)."""
        if self._fallback is None:
            self._fallback = Chatbot(
                self.table, client=self.client, settings=replace(self.settings, prompt_version=FALLBACK_PROMPT),
                prompt_version=FALLBACK_PROMPT, scope_mode=self.fallback_scope, answer_format="json",
                retry_interval=self.retry_interval, judge_mode="rules",
            )
        return self._fallback

    # ------------------------------------------------------------ steps

    @property
    def model_name(self) -> str:
        return self.settings.model_name if self.client.name == "hcx" else self.client.name

    def _model_access(self) -> ModelAccess:
        """How this question's model steps reach the model: the client is read at the first call (built then)."""
        return ModelAccess(client=lambda: self.client, model_name=lambda: self.model_name,
                           retry_interval=self.retry_interval, max_input_tokens=self.settings.max_input_tokens)

    # ------------------------------------------------------------ entry point

    @property
    def graph(self):
        """The compiled stat graph for this table (graph/build.py, built on first use; one per pipeline)."""
        if self._graph is None:
            self._graph = build_graph(self.resources)
        return self._graph

    def ask(self, question: str, golden_scope: dict | None = None) -> Answer:
        """One question through the stat graph (T1-28 S4; the steps and their order are graph/build.py's)."""
        llm = self._model_access()
        started = time.perf_counter()
        context = StatContext(llm=llm, chatbot=lambda: self.fallback, build=build_plan)  # this module's: tests swap it
        state = run(self.graph, question, context=context, meta={**self.decision_metadata, "route": "rules"},
                    started=started, golden_scope=golden_scope)
        return answer_of(state, table_meta=self.table.meta, prompt_version=self.prompt_version,
                         fallback_prompt=FALLBACK_PROMPT)


class MultiStatPipeline:
    """Answers questions over two or three tables (T1-30): the multi-table graph (graph/multi.py).

    Built from prepared TableInputs (the first table is the answer's `source`) or from the tables' StatResources, so
    a module that has prepared a table for one-table questions reuses its cleaning and index. The model is reached as
    StatPipeline reaches it (client=, llm= or settings=; built on first use, so a question the rules refuse needs no
    key). A plan the model could not write after one retry is answered as a refusal; a failed model call raises
    RuntimeError, as summarize_stat does (a reply in the wrong format is not a failed call: it is refused)."""

    def __init__(self, tables: list[TableInput] | list[StatResources], client: LLMClient | None = None,
                 settings: Settings | None = None, org_ids: list[str | None] | None = None,
                 retry_interval: float = 0.0, llm: LLMConfig | None = None) -> None:
        self.settings = settings or (settings_from_llm(llm) if llm is not None else load_settings())
        self.llm = llm
        self._client: LLMClient | None = client
        org_ids = org_ids or [None] * len(tables)
        prepared = [t if isinstance(t, StatResources) else load_resources(t, o) for t, o in zip(tables, org_ids, strict=True)]
        self.resources = MultiResources(tuple(prepared))
        self.retry_interval = retry_interval
        self.prompt_version = VERSION
        self.decision_metadata = {"pipeline": "stat_multi", "outcome_origin": "rules", "numbers_origin": "code",
                                  "defaults_origin": "rules"}
        self._graph = None

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = build_client(self.llm if self.llm is not None else self.settings)
        return self._client

    @client.setter
    def client(self, value: LLMClient) -> None:
        self._client = value

    @property
    def model_name(self) -> str:
        return self.settings.model_name if self.client.name == "hcx" else self.client.name

    def _model_access(self) -> ModelAccess:
        return ModelAccess(client=lambda: self.client, model_name=lambda: self.model_name,
                           retry_interval=self.retry_interval, max_input_tokens=self.settings.max_input_tokens)

    @property
    def graph(self):
        if self._graph is None:
            from kosis_chat.graph.multi import build_multi_graph

            self._graph = build_multi_graph(self.resources)
        return self._graph

    def ask(self, question: str) -> Answer:
        """One question through the multi-table graph; RuntimeError when the plan's model call failed."""
        from kosis_chat.graph.multi import run_multi

        started = time.perf_counter()
        context = StatContext(llm=self._model_access())
        state = run_multi(self.graph, question, context=context,
                          meta={**self.decision_metadata, "route": "rules"}, started=started)
        # the model could not be reached (not a reply in the wrong format, which is answered as a refusal)
        failed = [state.meta.get(k) for k in ("plan_status", "plan_retry_status")
                  if state.meta.get(k) in (STATUS_TIMEOUT, STATUS_API_ERROR, STATUS_INPUT_TOO_LONG)]
        if failed:
            raise RuntimeError(f"the multi-table plan's model call failed: {failed[-1]}")
        return answer_of(state, table_meta=self.resources.table.meta, prompt_version=self.prompt_version,
                         fallback_prompt=FALLBACK_PROMPT)

