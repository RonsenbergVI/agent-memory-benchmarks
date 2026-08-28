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

"""agentmemory (rohitg00/agentmemory) — observation memory over a REST API.

A Node service on port 3111 with no Python SDK, so the adapter is a small
httpx client against the endpoints its own README documents (the PyPI
``agentmemory`` is a different project — see pyproject.toml).
AGENTMEMORY_BASE_URL names the server (compose stands one up);
AGENTMEMORY_SECRET is sent as a bearer when the server requires one.

Ingestion is agentmemory's own hook path: each dataset session opens an
agentmemory *session* and every turn is posted as one ``prompt_submit``
observation — the only hook whose content is the utterance itself.
Keyless (the server's default) it stores the turn verbatim and retrieves
with BM25 plus on-device embeddings; a provider key turns on LLM
compression instead — a different system and a different row.

Two non-obvious properties of the service shape this adapter:

* **Isolation is the agent id, not the project.** ``mem::smart-search``
  never passes ``project`` to the searcher; it filters on ``agentId``,
  which observations inherit from the session row set at
  ``session/start``. So the conversation id travels as the agent id, at
  session start and on every search.
* **That filter runs after retrieval, not inside the index.** The server
  over-fetches 3x the requested limit and trims, so with several
  conversations resident a conversation's true top-k can be crowded out
  of the window before the filter runs. Over-asking widens the window
  but cannot close it: ``--workers 1`` is the only exact-isolation
  setting. See benchmarks/agentmemory/README.md.

Search is two round trips by construction — a compact search returns ids
and scores, a second call expands them into content — counted as one
search, what the system charges to answer one question.
"""

from typing import ClassVar

from amb.base import Memory
from amb.contracts import MemoryHit, Session
from amb.logs import logger, quiet_frameworks
from src.settings import Settings

# `cwd` is required by the hook contract; meaningless for a conversation corpus
DEFAULT_CWD = "/amb"
# the only hook whose payload is the utterance itself
HOOK_TYPE = "prompt_submit"
# `expandIds` is capped at 20 server-side; every k the harness sweeps fits
MAX_EXPAND = 20
# over-ask so the server's post-retrieval agent filter trims from a wider
# window; its own over-fetch is 3x the limit capped at 300, so 100 asks
# for the most the API will honour
FETCH_MULTIPLIER = 10
MAX_LIMIT = 100
DEFAULT_TIMEOUT_S = 120.0


class AgentMemoryMemory(Memory):
    """agentmemory: observation memory over a REST API."""

    name: ClassVar[str] = "agentmemory"
    description: ClassVar[str] = "agentmemory — observation memory over REST"
    # no Python distribution: the server reports its own version
    sdk_dist: ClassVar[str | None] = None
    # Token spend happens inside the Node server; the harness cannot observe it.
    usage_coverage: ClassVar[str] = "none"

    def __init__(
        self,
        base_url: str | None = None,
        secret: str | None = None,
        include_lessons: bool = False,
        timeout: float | str = DEFAULT_TIMEOUT_S,
        **params: object,
    ) -> None:
        """Point the adapter at an agentmemory server."""
        super().__init__(**params)
        settings = Settings()
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self._secret = secret or settings.secret
        # lessons are LLM-derived and never produced keyless; asking for
        # them costs a lookup per search for nothing
        self.include_lessons = include_lessons
        # --param values arrive as strings
        self.timeout = float(timeout)
        # the models are the server's business and there are none keyless
        self.model: str | None = None
        self.embedding_model: str | None = None
        # observation id -> turn id: restores the turn behind a hit
        self._turns: dict[str, str] = {}
        # the agentmemory sessions this instance opened, for teardown
        self._sessions: list[str] = []
        self._conversation_id: str | None = None
        self._observations = 0

    def version(self) -> str | None:
        """The server's version — the server is the system under test."""
        try:
            return str(self._get("/agentmemory/health").get("version") or "") or None
        except Exception:  # noqa: BLE001 - any failure leaves the version unknown
            return None

    @staticmethod
    def _agent_id(conversation_id: str) -> str:
        """The agent id that isolates this conversation on every search."""
        return f"conv-{conversation_id}"

    def setup(self) -> None:
        """Open the HTTP client and confirm the server is answering."""
        import httpx

        headers = {"Content-Type": "application/json"}
        if self._secret:
            headers["Authorization"] = f"Bearer {self._secret}"
        self.client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=self.timeout
        )
        self._get("/agentmemory/health")
        quiet_frameworks("httpx")

    def _get(self, path: str) -> dict:
        """One GET, raising on anything but success."""
        response = self.client.get(path)
        response.raise_for_status()
        return response.json() or {}

    def _post(self, path: str, body: dict) -> dict:
        """One POST, raising on anything but success.

        The body is logged on failure — the benchmark log outlives the
        container that answered the 400.
        """
        try:
            response = self.client.post(path, json=body)
            response.raise_for_status()
        except Exception:
            logger.bind(scope="agentmemory").error(
                "{} failed: body={!r}", path, {k: body[k] for k in list(body)[:6]}
            )
            raise
        return response.json() or {}

    def _ensure_session(self, conversation_id: str, session_id: str) -> str:
        """Open this session on the server once, and return its scoped id.

        `observe` has no agent-id field — observations inherit it from the
        session row — so this call is what makes them filterable back to
        the conversation. Lazy because the agentic write toolset never
        announces a session first.
        """
        scoped = self._scoped_session_id(conversation_id, session_id)
        if scoped in self._sessions:
            return scoped
        self._post(
            "/agentmemory/session/start",
            {
                "sessionId": scoped,
                "project": self._agent_id(conversation_id),
                "cwd": DEFAULT_CWD,
                "agentId": self._agent_id(conversation_id),
            },
        )
        self._sessions.append(scoped)
        return scoped

    def _scoped_session_id(self, conversation_id: str, session_id: str) -> str:
        """A session id unique across conversations.

        Server session ids are global and dataset ids repeat ("1", "2",
        ...), so side-by-side conversations would otherwise write into
        each other's session row.
        """
        return f"{conversation_id}:{session_id}"

    def observe(
        self,
        conversation_id: str,
        content: str,
        *,
        session_id: str,
        timestamp: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Post one observation into the conversation's session."""
        self._conversation_id = conversation_id
        scoped = self._ensure_session(conversation_id, session_id)
        response = self._post(
            "/agentmemory/observe",
            {
                "hookType": HOOK_TYPE,
                "sessionId": scoped,
                "project": self._agent_id(conversation_id),
                "cwd": DEFAULT_CWD,
                "timestamp": timestamp or "1970-01-01T00:00:00Z",
                "data": {"prompt": content},
            },
        )
        # a repeated utterance dedups server-side (no id back); claiming
        # provenance for it would claim a hit the store cannot return
        if observation_id := self._observation_id(response):
            self._observations += 1
            if turn_id:
                self._turns[observation_id] = turn_id

    @staticmethod
    def _observation_id(response: dict) -> str | None:
        """The id agentmemory filed an observation under, if it filed one."""
        for key in ("obsId", "observationId", "id"):
            if value := response.get(key):
                return str(value)
        observation = response.get("observation")
        if isinstance(observation, dict) and (value := observation.get("id")):
            return str(value)
        return None

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Open an agentmemory session and post every turn into it."""
        if not session.turns:
            return
        self._conversation_id = conversation_id
        self._ensure_session(conversation_id, session.session_id)
        for turn in session.turns:
            self.observe(
                conversation_id,
                f"{turn.speaker}: {turn.text}",
                session_id=session.session_id,
                timestamp=session.timestamp,
                turn_id=turn.turn_id,
            )

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return up to k observations, content included.

        Two calls because the API is built that way: the compact search
        ranks and returns ids, a second call expands the chosen ids into
        their observations.
        """
        if not query.strip():
            return []
        agent_id = self._agent_id(conversation_id)
        compact = self._post(
            "/agentmemory/smart-search",
            {
                "query": query,
                "limit": min(max(k * FETCH_MULTIPLIER, k), MAX_LIMIT),
                "agentId": agent_id,
                "includeLessons": self.include_lessons,
            },
        )
        ranked = [r for r in (compact.get("results") or []) if r.get("obsId")][:k]
        if not ranked:
            return []
        expanded = self._post(
            "/agentmemory/smart-search",
            {
                "expandIds": [
                    {"obsId": r["obsId"], "sessionId": r.get("sessionId")}
                    for r in ranked[:MAX_EXPAND]
                ],
                "agentId": agent_id,
            },
        )
        contents = {
            str(item.get("obsId")): item.get("observation") or {}
            for item in (expanded.get("results") or [])
        }
        hits = []
        for result in ranked:
            observation = contents.get(str(result["obsId"]), {})
            hits.append(
                MemoryHit(
                    content=self._content(observation),
                    score=result.get("score"),
                    turn_ids=self._turn_ids(str(result["obsId"])),
                    session_ids=self._session_ids(result),
                    metadata={
                        "obs_id": result["obsId"],
                        "type": result.get("type"),
                    },
                )
            )
        return hits

    @staticmethod
    def _content(observation: dict) -> str:
        """The observation's text.

        Keyless, `narrative` is the utterance verbatim; with compression
        on, `facts` carries the extracted statements — joined rather than
        assuming which one the server was configured to produce.
        """
        parts = [str(observation.get("narrative") or "").strip()]
        parts += [str(f).strip() for f in (observation.get("facts") or [])]
        return "\n".join(p for p in parts if p)

    def _turn_ids(self, observation_id: str) -> list[str]:
        """The turn this observation was posted from, if it was one of ours."""
        turn_id = self._turns.get(observation_id)
        return [turn_id] if turn_id else []

    def _session_ids(self, result: dict) -> list[str]:
        """The dataset session behind a hit, unscoped back from its id."""
        scoped = str(result.get("sessionId") or "")
        if not scoped:
            return []
        # conversation ids carry no colon, so one partition is unambiguous
        _, _, session_id = scoped.partition(":")
        return [session_id or scoped]

    def teardown(self) -> None:
        """Forget the sessions this instance opened, and only those.

        Under `--workers N` other conversations' sessions are on the same
        server; anything wider would take them with it.
        """
        for session_id in self._sessions:
            try:
                self._post("/agentmemory/forget", {"sessionId": session_id})
            except Exception:  # noqa: BLE001 - teardown must not fail a scored run
                logger.bind(scope="agentmemory").warning(
                    "could not forget session {!r}", session_id
                )
        self._sessions.clear()
        self._turns.clear()
        if client := getattr(self, "client", None):
            client.close()

    def stats(self) -> dict:
        """Report what this run stored."""
        return {
            "observations": self._observations,
            "sessions": len(self._sessions),
            "include_lessons": self.include_lessons,
        }
