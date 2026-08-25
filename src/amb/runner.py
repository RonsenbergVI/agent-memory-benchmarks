# MIT License
#
# Copyright (c) 2026 René-Jean Corneille
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from amb.base import Benchmark, Callback, Callbacks, Memory
from amb.callbacks import TimingTracker
from amb.constants import DEFAULT_DATA_DIR, DEFAULT_RUNS_DIR, RunType
from amb.contracts import IngestionRecord, QAPair, Run, Sample, Session


class RunConfig(BaseModel):
    """Everything one benchmark run needs, all of it from the CLI.

    These are the run's arguments, not the system's: memory-system
    parameters (`--param`) belong to the Benchmark that is being run.
    """

    dataset: str
    variant: str | None = None
    mode: RunType = RunType.DIRECT  # "direct": the harness drives ingest and search;
    # "agentic": a model drives both through the system's own tools.
    # In either mode, answers are generated (the qa phase's second half)
    # iff a model is available — optional in direct, required in agentic.
    k: int = 10
    limit: int | None = None  # max conversations
    sample_seed: int | None = None  # random-sample `limit` conversations
    workers: int = 1  # samples run in parallel; sessions stay ordered
    keep: bool = False  # skip teardown so the store can be reused
    reuse: bool = False  # skip ingestion, search what the store holds
    max_turns: int | None = None  # turn budget ingested per conversation
    max_questions: int | None = None  # max questions per conversation
    model: str | None = None  # answer model, pydantic-ai id; enables answer
    # generation in direct mode, mandatory in agentic mode
    judge: bool = False  # CLI convenience: judge right after the run (the
    # judge itself is an evaluation step, not part of the run loop)
    judge_model: str | None = None  # judge model, pydantic-ai id
    data_dir: Path = DEFAULT_DATA_DIR  # dataset cache
    out: Path = DEFAULT_RUNS_DIR  # where run data is written


def draw_samples[T](samples: list[T], limit: int | None, seed: int) -> list[T]:
    """Reproducibly draw `limit` samples, keeping dataset order."""
    if limit is None or limit >= len(samples):
        return samples
    chosen = random.Random(seed).sample(samples, limit)
    order = {id(sample): i for i, sample in enumerate(samples)}
    return sorted(chosen, key=lambda sample: order[id(sample)])


class Runner:
    """Executes one benchmark under one run config.

    Subclass it only to change the *experiment* — how samples are drawn,
    how a question is asked. Changing how one memory system behaves belongs
    on that system's Benchmark instead.
    """

    # measurements no run may be missing: latency is the benchmark's own
    # headline result, so it is the harness's to attach, not an
    # integration's to opt into
    core_callback_classes: ClassVar[tuple[type[Callback], ...]] = (TimingTracker,)

    def __init__(
        self,
        benchmark: Benchmark,
        config: RunConfig,
        samples: list[Sample] | None = None,
    ) -> None:
        """Bind a benchmark to a run config, optionally with fixed samples.

        `samples` bypasses the dataset loader entirely — the escape hatch
        for tests and notebooks that already hold the conversations.
        """
        self.benchmark = benchmark
        self.config = config
        self.samples = samples
        self.callbacks: Callbacks = Callbacks()

    @property
    def name(self) -> str:
        """The benchmark's name, as it is recorded and logged."""
        return self.benchmark.name

    def load_samples(self) -> list[Sample]:
        """Load the dataset's samples, honouring `--limit` and `--seed`.

        Without a seed, `--limit` takes the first N samples (stable, cheap:
        loaders stop reading early). With one, the whole dataset is loaded
        and N samples are drawn reproducibly — the same seed always picks
        the same conversations, so runs stay comparable.
        """
        if self.samples is not None:
            return self.samples
        # imported here: datasets import the base package, so a module-level
        # import would be circular
        from amb.datasets import get_loader

        loader = get_loader(self.config.dataset, self.config.data_dir)
        if self.config.sample_seed is None:
            return loader.load(self.config.variant, limit=self.config.limit)
        return draw_samples(
            loader.load(self.config.variant),
            self.config.limit,
            self.config.sample_seed,
        )

    def select_sessions(self, sample: Sample) -> list[Session]:
        """Sessions to ingest, honouring the `--turns` budget.

        Sessions are taken in chronological order until the budget runs out;
        the one that straddles the boundary is truncated, so the result is
        always a valid prefix of the conversation. Without a budget the whole
        conversation is returned.
        """
        budget = self.config.max_turns
        if budget is None:
            return list(sample.conversation.sessions)
        chosen: list[Session] = []
        for session in sample.conversation.sessions:
            if budget <= 0:
                break
            if len(session.turns) <= budget:
                chosen.append(session)
                budget -= len(session.turns)
            else:
                chosen.append(
                    session.model_copy(update={"turns": session.turns[:budget]})
                )
                budget = 0
        return chosen

    def select_questions(self, sample: Sample, sessions: list[Session]) -> list[QAPair]:
        """Questions to ask: the answerable ones, capped by `--questions`.

        The unanswerable are removed *before* the cap, so `--questions 10`
        means ten questions that can actually be answered rather than ten
        candidates of which some are dead.
        """
        return self.answerable_questions(sample, sessions)[: self.config.max_questions]

    def answerable_questions(
        self, sample: Sample, sessions: list[Session]
    ) -> list[QAPair]:
        """The sample's questions whose evidence survived the turn budget.

        When a turn budget drops the evidence a question depends on, that
        question cannot be answered from what was ingested; keeping it would
        report a retrieval failure that the data, not the system, caused.
        Questions without evidence labels are kept, since there is no way
        to tell whether their evidence survived.
        """
        questions = sample.qa
        if self.config.max_turns is None:
            return list(questions)
        ingested_turns = {t.turn_id for s in sessions for t in s.turns}
        original = {s.session_id: len(s.turns) for s in sample.conversation.sessions}
        # a truncated session may have lost the evidence turn, so only
        # sessions ingested whole can vouch for session-level labels
        whole_sessions = {
            s.session_id for s in sessions if len(s.turns) == original[s.session_id]
        }
        answerable = []
        for qa in questions:
            if qa.evidence_turn_ids:
                if set(qa.evidence_turn_ids) <= ingested_turns:
                    answerable.append(qa)
            elif qa.evidence_session_ids:
                if set(qa.evidence_session_ids) <= whole_sessions:
                    answerable.append(qa)
            else:
                answerable.append(qa)
        return answerable

    # -- the run loop -----------------------------------------------------

    def run(self) -> Run:
        """Run every sample and return the raw observations.

        Computes no scores and writes no files — that is the report's job.

        Raises:
            ValueError: in agentic mode, when the benchmark declares no
                toolsets or the config names no model — an agentic run
                never falls back to a default model or to direct behavior.
        """
        cfg = self.config
        if cfg.mode == "agentic":
            if not self.benchmark.supports_agentic():
                raise ValueError(
                    f"{self.name} does not support agentic mode: its Benchmark "
                    "must set search_toolset_class and ingest_toolset_class "
                    "to the system's own tool surface"
                )
            if not cfg.model:
                raise ValueError(
                    "agentic mode needs --model: an agentic run never falls "
                    "back to a default model or to direct behavior"
                )
        # one probe instance answers what will be measured: the system's
        # version and the models it will call internally
        probe = self.benchmark.create_system()
        internal = probe.models()
        data = Run(
            # the timestamp alone collides across parallel CI cells (two
            # cells starting the same second produced the same run_id and
            # fought over one directory); the random suffix keeps every
            # run's directory and identity unique
            run_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:6]}",
            system=self.name,
            dataset=cfg.dataset,
            variant=cfg.variant,
            mode=cfg.mode,
            k=cfg.k,
            ingestion_model=internal.get("ingestion_model"),
            embedding_model=internal.get("embedding_model"),
            tracks_usage=type(probe).tracks_usage,
            # explicit overrides join the run's identity; the two model
            # params are already lifted into their own fields above
            system_params={
                key: value
                for key, value in self.benchmark.params.items()
                if key not in ("model", "embedding_model")
            },
            # str() so tests can pass a pydantic-ai Model instance through
            # the config; real runs always carry the id string already
            model=None if cfg.model is None else str(cfg.model),
            max_turns=cfg.max_turns,
            sample_seed=cfg.sample_seed,
            workers=cfg.workers,
            system_version=probe.version(),
        )
        # core callbacks first, so no benchmark override can drop the
        # harness's own measurements
        self.callbacks = Callbacks(
            [cls() for cls in self.core_callback_classes]
            + self.benchmark.create_callbacks().callbacks
        )
        self.callbacks.on_run_begin(cfg, data)
        samples = self.load_samples()
        if cfg.workers <= 1:
            for sample in samples:
                self._run_sample(sample, data)
        else:
            # samples are independent (each isolates its own conversation),
            # so they parallelize; sessions within one stay strictly ordered.
            # Each worker collects into its own Run, merged in dataset
            # order so output is deterministic regardless of scheduling.
            with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                shards = [
                    data.model_copy(
                        update={"question_records": [], "ingestion_records": []}
                    )
                    for _ in samples
                ]
                futures = [
                    pool.submit(self._run_sample, sample, shard)
                    for sample, shard in zip(samples, shards, strict=True)
                ]
                for future in futures:
                    future.result()
            for shard in shards:
                data.question_records.extend(shard.question_records)
                data.ingestion_records.extend(shard.ingestion_records)
        self.callbacks.on_run_end(data)
        return data

    def _run_sample(self, sample: Sample, data: Run) -> None:
        system = self.benchmark.create_system()
        try:
            system.setup()
            self.benchmark.before_sample(system, sample)
            self.callbacks.on_sample_begin(sample, system)
            stats = self.ingestion(system, sample, data)
            t0 = time.perf_counter()
            self.conversation(system, sample, data)
            stats.conversation_s = time.perf_counter() - t0
            self.benchmark.after_sample(system, sample)
        finally:
            try:
                self.callbacks.on_sample_end(sample, system)
            except Exception:
                traceback.print_exc()
            try:
                if self.config.keep:
                    logger.bind(scope=self.name).info(
                        "{}: keeping ingested memories (--keep)", sample.sample_id
                    )
                else:
                    system.teardown()
            except Exception:
                traceback.print_exc()

    def ingestion(self, system: Memory, sample: Sample, data: Run) -> IngestionRecord:
        """Feed the sample's sessions into the memory system, timed.

        `max_turns` truncates the conversation for smoke runs. Returns the
        sample's stats record, which the caller extends with the conversation
        time.
        """
        sessions = self.select_sessions(sample)
        log = logger.bind(scope=self.name)
        if self.config.reuse:
            # the store already holds this corpus (a prior --keep run);
            # charge zero ingestion and go straight to the questions
            log.info("{}: reusing existing store, skipping ingestion", sample.sample_id)
            stats = {
                "sample_id": sample.sample_id,
                "ingest_s": 0.0,
                "num_sessions": len(sessions),
                "num_turns": sum(len(s.turns) for s in sessions),
                "reused": True,
                "system_stats": system.stats(),
            }
            dropped = len(sample.qa) - len(self.answerable_questions(sample, sessions))
            if dropped:
                stats["questions_dropped"] = dropped
            self.callbacks.on_ingest_end(sample, system, stats)
            return data.add_ingestion(stats)
        log.debug(
            "{}: ingesting {} sessions / {} turns (of {} / {})",
            sample.sample_id,
            len(sessions),
            sum(len(s.turns) for s in sessions),
            len(sample.conversation.sessions),
            sample.conversation.num_turns,
        )
        # the empty-ingest early return above deliberately skips this, so
        # the timer charges a truthful zero there; here the clock must start
        # or TimingTracker.on_ingest_end overwrites ingest_s with 0.0
        self.callbacks.on_ingest_begin(sample, system)
        t0 = time.perf_counter()
        agent_stats: dict = {}
        if self.config.mode == "agentic":
            agent_stats = self._agent_ingestion(system, sample, sessions)
        else:
            for session in tqdm(
                sessions,
                desc=f"ingest {sample.sample_id}",
                unit="session",
                leave=False,
            ):
                system.ingest_session(sample.conversation.conversation_id, session)
        stats = {
            "sample_id": sample.sample_id,
            "ingest_s": time.perf_counter() - t0,
            # what was actually ingested, not what the sample holds
            "num_sessions": len(sessions),
            "num_turns": sum(len(s.turns) for s in sessions),
            "system_stats": system.stats(),
            **agent_stats,
        }
        dropped = len(sample.qa) - len(self.answerable_questions(sample, sessions))
        if dropped:
            # questions whose evidence fell outside the turn budget: they are
            # unanswerable by construction, so scoring them would punish the
            # memory system for data it was never given
            stats["questions_dropped"] = dropped
        log.debug(
            "{}: ingested in {:.1f}s; {} of {} questions answerable",
            sample.sample_id,
            stats["ingest_s"],
            len(sample.qa) - dropped,
            len(sample.qa),
        )
        self.callbacks.on_ingest_end(sample, system, stats)
        return data.add_ingestion(stats)

    def _agent_ingestion(
        self, system: Memory, sample: Sample, sessions: list[Session]
    ) -> dict:
        """Agentic ingestion: a model writes each session via the system's tools.

        Returns the extra stats fields the agent's involvement adds — write
        counts and the agent's own token spend (its LLM traffic is part of
        what this mode costs, distinct from the memory system's).
        """
        # imported lazily so retrieval-only runs need no LLM deps/keys
        from amb.agent import ingest_with_agent

        totals = {
            "num_writes": 0,
            "write_s": 0.0,
            "agent_input_tokens": 0,
            "agent_output_tokens": 0,
        }
        assert self.config.model is not None  # guaranteed by run()
        for session in tqdm(
            sessions,
            desc=f"ingest {sample.sample_id}",
            unit="session",
            leave=False,
        ):
            toolset = self.benchmark.create_ingest_toolset(
                system, sample.conversation.conversation_id, session
            )
            generation = ingest_with_agent(session, toolset, model=self.config.model)
            totals["num_writes"] += toolset.num_writes
            totals["write_s"] += toolset.write_s
            totals["agent_input_tokens"] += generation.input_tokens
            totals["agent_output_tokens"] += generation.output_tokens
        return totals

    def conversation(self, system: Memory, sample: Sample, data: Run) -> None:
        """Ask every benchmark question against the ingested memory.

        Produces one raw data row per question.
        """
        for qa in self.select_questions(sample, self.select_sessions(sample)):
            self.callbacks.on_question_begin(sample, qa)
            record = self._run_question(system, sample, qa)
            self.callbacks.on_question_end(sample, qa, record)
            data.add_question(record)

    def _log_retrieval(self, qa: QAPair, hits: list, row: dict) -> None:
        """Trace one question's retrieval at debug level."""
        logger.bind(scope=self.name).debug(
            "{}: {} hits, sessions {} vs {}, turns {} vs {}",
            qa.question_id,
            len(hits),
            len(row.get("retrieved_session_ids") or []),
            row.get("evidence_session_ids"),
            len(row.get("retrieved_turn_ids") or []),
            row.get("evidence_turn_ids"),
        )

    def _run_question(self, system: Memory, sample: Sample, qa: QAPair) -> dict:
        cfg = self.config
        row = {
            "sample_id": sample.sample_id,
            "question_id": qa.question_id,
            "category": qa.category,
            "question": qa.question,
            "gold_answer": qa.answer,
        }
        try:
            generation = None
            if cfg.mode == "agentic":
                # imported lazily so model-free runs need no LLM deps/keys
                from amb.agent import answer_with_memory

                toolset = self.benchmark.create_search_toolset(
                    system, sample.conversation.conversation_id, k=cfg.k
                )
                # the agent's searches are the harness's only view of
                # retrieval here: they must report through the same hooks
                # as the searches the harness drives itself
                toolset.observe(self.callbacks, sample, qa)
                assert cfg.model is not None  # guaranteed by run()
                t0 = time.perf_counter()
                generation = answer_with_memory(
                    qa.question,
                    toolset,
                    cfg.model,
                    question_date=qa.question_date,
                )
                row["answer_s"] = time.perf_counter() - t0
                # retrieval is scored on everything the agent chose to look at
                hits = toolset.retrieved
                row["search_s"] = toolset.search_s
                row["num_searches"] = toolset.num_searches
                row["num_hits"] = len(hits)
            else:
                self.callbacks.on_search_begin(sample, qa)
                t0 = time.perf_counter()
                hits = system.search(
                    sample.conversation.conversation_id, qa.question, k=cfg.k
                )
                elapsed = time.perf_counter() - t0
                row["search_s"] = elapsed
                row["num_hits"] = len(hits)
                self.callbacks.on_search(sample, qa, hits, elapsed)

            # raw retrieval data at every level available. Session ids are
            # the comparable headline (every system can attest a memory's
            # session); turn ids are the stricter bonus for systems that
            # store verbatim turns.
            scored = False
            if qa.evidence_session_ids and (
                not hits or any(h.session_ids for h in hits)
            ):
                row["retrieved_session_ids"] = [s for h in hits for s in h.session_ids]
                row["evidence_session_ids"] = qa.evidence_session_ids
                scored = True
            if qa.evidence_turn_ids and (not hits or any(h.turn_ids for h in hits)):
                row["retrieved_turn_ids"] = [t for h in hits for t in h.turn_ids]
                row["evidence_turn_ids"] = qa.evidence_turn_ids
                scored = True
            if scored:
                self._log_retrieval(qa, hits, row)
            elif hits and (qa.evidence_turn_ids or qa.evidence_session_ids):
                logger.bind(scope=self.name).warning(
                    "{} returned {} hits for {} with no usable provenance "
                    "(populate MemoryHit.turn_ids or session_ids); the "
                    "question is left out of retrieval metrics instead of "
                    "scoring a false 0.0",
                    self.name,
                    len(hits),
                    qa.question_id,
                )

            if cfg.mode != "agentic" and cfg.model:
                # direct mode with a model: the qa phase's second half,
                # answering from the retrieved context
                from amb.agent import answer_question

                t0 = time.perf_counter()
                generation = answer_question(
                    qa.question, hits, cfg.model, question_date=qa.question_date
                )
                row["answer_s"] = time.perf_counter() - t0

            if generation is not None:
                row["predicted_answer"] = generation.text
                row["input_tokens"] = generation.input_tokens
                row["output_tokens"] = generation.output_tokens
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        return row
