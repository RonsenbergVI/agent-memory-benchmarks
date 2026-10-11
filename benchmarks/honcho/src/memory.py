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

"""Honcho (plastic-labs/honcho) — memory reasoned from messages.

Needs a running Honcho server — its API and its deriver worker
(docker-compose.yaml here) — named by HONCHO_BASE_URL. Each conversation
is a Honcho *workspace*, each speaker a *peer*, each dataset session a
Honcho session, and each turn a message from its speaker's peer.

Honcho stores, then reasons asynchronously: the deriver reads every
message and writes conclusions about its speaker. `ingest_session`
waits for the session's queue to drain before returning, so that work
lands in ingest time and is never still in flight when the first
question is asked.

Recall is Honcho's workspace search — hybrid (semantic + full-text)
over the stored messages — so every hit is a verbatim turn.
"""

import json
import os
import re
import time
import urllib.request
from datetime import UTC, datetime
from typing import Any, ClassVar

from amb.base import Memory
from amb.contracts import MemoryHit, Session
from amb.logs import logger

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_INGESTION_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# seconds one session's reasoning may take before the run is failed
DEFAULT_QUEUE_TIMEOUT = 900
# the API enqueues reasoning after it answers, so an empty queue straight
# after a write can mean "not enqueued yet"; past this, it means "no work"
ENQUEUE_GRACE_S = 10.0
# the API's per-request ceilings
MESSAGE_BATCH = 100
SEARCH_LIMIT = 100
TIMESTAMP_FORMATS = ("%I:%M %p on %d %B, %Y", "%Y/%m/%d (%a) %H:%M", "%Y-%m-%d")


def _resource_id(text: str) -> str:
    """The text as a Honcho resource id, which admits ``[A-Za-z0-9_-]`` only."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", text)[:512]


class HonchoMemory(Memory):
    """Honcho: messages per peer, reasoned over server-side, searched hybrid."""

    name: ClassVar[str] = "honcho"
    description: ClassVar[str] = "Honcho — memory reasoned from messages"
    sdk_dist: ClassVar[str | None] = "honcho-ai"
    # every LLM and embedding call happens inside the server, and Honcho
    # reports no usage for any of them
    usage_coverage: ClassVar[str] = "none"

    def __init__(
        self,
        base_url: str | None = None,
        queue_timeout: float | str = DEFAULT_QUEUE_TIMEOUT,
        timeout: float | str = 60.0,
        **params: object,
    ) -> None:
        """Point the adapter at a Honcho server."""
        super().__init__(**params)
        self.base_url = base_url or os.environ.get("HONCHO_BASE_URL", DEFAULT_BASE_URL)
        # --param values arrive as strings
        self.queue_timeout = float(queue_timeout)
        self.timeout = float(timeout)
        # the server's configuration (docker-compose.yaml), recorded as identity
        self.model = DEFAULT_INGESTION_MODEL
        self.embedding_model = DEFAULT_EMBEDDING_MODEL
        self.client: Any = None
        self._conversation_id: str | None = None
        # speaker -> peer
        self._peers: dict[str, Any] = {}
        self._messages = 0

    def version(self) -> str | None:
        """The server's version — the server is the system under test.

        Read from the API's OpenAPI document, which carries the server's
        own version; falls back to the client SDK's when unreachable.
        """
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/openapi.json", timeout=5
            ) as r:
                server = json.load(r).get("info", {}).get("version")
            if server:
                return str(server)
        except OSError:
            pass
        return super().version()

    def _workspace(self, conversation_id: str) -> Any:
        """The conversation's workspace client, created on first use.

        Summaries and dreams are switched off for the workspace: neither
        feeds message search, and dreams fire on idle timers that would
        land at arbitrary points of a run.
        """
        if self._conversation_id == conversation_id:
            return self.client
        from honcho import Honcho
        from honcho.api_types import (
            DreamConfiguration,
            SummaryConfiguration,
            WorkspaceConfiguration,
        )

        self.client = Honcho(
            base_url=self.base_url,
            workspace_id=_resource_id(f"amb-{conversation_id}"),
            timeout=self.timeout,
        )
        # get-or-create, so the configuration below has a workspace to land on
        self.client.get_configuration()
        self.client.set_configuration(
            WorkspaceConfiguration(
                summary=SummaryConfiguration(enabled=False),
                dream=DreamConfiguration(enabled=False),
            )
        )
        self._conversation_id = conversation_id
        return self.client

    def _peer(self, speaker: str) -> Any:
        """The speaker's peer; its metadata keeps the name the id had to drop."""
        if speaker not in self._peers:
            self._peers[speaker] = self.client.peer(
                _resource_id(speaker), metadata={"speaker": speaker}
            )
        return self._peers[speaker]

    @staticmethod
    def _timestamp(session: Session) -> datetime | None:
        """The session's in-world date, so reasoning can place it in time."""
        for fmt in TIMESTAMP_FORMATS:
            try:
                return datetime.strptime(session.timestamp or "", fmt).replace(
                    tzinfo=UTC
                )
            except ValueError:
                continue
        return None

    def store(
        self,
        conversation_id: str,
        session: Session,
        writes: list[tuple[str, str, list[str]]],
    ) -> None:
        """Add (speaker, content, turn ids) messages to the session, then drain.

        The turn ids ride in the message metadata, beside the dataset
        session id, so a search hit names both back without a lookup.
        """
        self._workspace(conversation_id)
        honcho_session = self.client.session(
            _resource_id(session.session_id),
            metadata={"session_id": session.session_id},
        )
        created_at = self._timestamp(session)
        messages = [
            self._peer(speaker).message(
                content,
                metadata={"turn_ids": turn_ids, "session_id": session.session_id},
                created_at=created_at,
            )
            for speaker, content, turn_ids in writes
        ]
        for start in range(0, len(messages), MESSAGE_BATCH):
            honcho_session.add_messages(messages[start : start + MESSAGE_BATCH])
        self._messages += len(messages)
        self._drain(honcho_session.id)

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Add the session's turns as messages from their speakers' peers."""
        writes = [
            (turn.speaker, turn.text, [turn.turn_id])
            for turn in session.turns
            if turn.text.strip()
        ]
        if writes:
            self.store(conversation_id, session, writes)

    def _drain(self, session_id: str) -> None:
        """Block until the deriver has worked off this session's queue.

        Raises:
            TimeoutError: If the queue is still busy after `queue_timeout`,
                rather than let questions run against a half-processed store.
        """
        start = time.monotonic()
        delay = 0.2
        seen = False
        while True:
            status = self.client.queue_status(session=session_id)
            seen = seen or status.total_work_units > 0
            idle = status.pending_work_units == 0 and status.in_progress_work_units == 0
            elapsed = time.monotonic() - start
            if idle and (seen or elapsed > ENQUEUE_GRACE_S):
                return
            if elapsed > self.queue_timeout:
                raise TimeoutError(
                    f"honcho session {session_id!r} still reasoning after "
                    f"{self.queue_timeout:.0f}s ({status.pending_work_units} pending, "
                    f"{status.in_progress_work_units} in progress)"
                )
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return up to k messages from the workspace search, query verbatim."""
        if not query.strip():
            return []
        self._workspace(conversation_id)
        hits = []
        for message in self.client.search(query, limit=min(k, SEARCH_LIMIT)):
            metadata = message.metadata or {}
            hits.append(
                MemoryHit(
                    content=f"{self._speaker_of(message.peer_id)}: {message.content}",
                    turn_ids=list(metadata.get("turn_ids") or []),
                    session_ids=[metadata["session_id"]]
                    if "session_id" in metadata
                    else [],
                    metadata={"message_id": message.id},
                )
            )
        return hits

    def _speaker_of(self, peer_id: str) -> str:
        """A peer id's speaker name, as the dataset wrote it."""
        if not self._peers:
            # --reuse skipped ingestion; the server still has the peers
            for peer in self.client.peers():
                speaker = (peer.metadata or {}).get("speaker", peer.id)
                self._peers[str(speaker)] = peer
        for speaker, peer in self._peers.items():
            if peer.id == peer_id:
                return speaker
        return peer_id

    def teardown(self) -> None:
        """Delete this conversation's workspace, and only this conversation's.

        Honcho refuses to delete a workspace with live sessions, so they
        go first; the conversations running beside this one under
        `--workers N` live in workspaces of their own.
        """
        if self.client is None:
            return
        try:
            for session in list(self.client.sessions()):
                session.delete()
            self.client.delete_workspace(self.client.workspace_id)
        except Exception:  # noqa: BLE001 - teardown must not fail a scored run
            logger.bind(scope="honcho").warning(
                "could not delete workspace {!r}", self.client.workspace_id
            )

    def stats(self) -> dict:
        """Report what this run stored."""
        return {"messages": self._messages}
