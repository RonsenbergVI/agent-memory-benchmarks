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

"""LongMemEval (Wu et al., ICLR 2025).

Long-term interactive memory for chat assistants, distributed on HuggingFace.

Each instance is one question over its own haystack of chat sessions. Variants:
`s` (~115k-token haystacks, default), `m` (~500 sessions), `oracle` (evidence
sessions only), and two cuts of `s` made at load time: `xs` (each question's
evidence plus a few of its own fillers) and `xs-shared` (questions dealt into
a few synthetic users who each hold one long history).
"""

import json
import random
from collections.abc import Collection
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from amb.base.dataset import Dataset, DatasetLoader
from amb.contracts import (
    Conversation,
    QAPair,
    Sample,
    Session,
    Turn,
)

# upstream deprecated xiaowu0162/longmemeval; -cleaned removes noisy history sessions
DEFAULT_REPO = "xiaowu0162/longmemeval-cleaned"

# variants cut at load time from a published file
CUTS = {"xs": "s", "xs-shared": "s"}
XS_FILLERS = 4
SHARED_USERS = 10
SHARED_FILLERS = 20


class LongMemEvalLoader(DatasetLoader):
    """Loads LongMemEval questions with their haystack sessions."""

    name = Dataset.LONGMEMEVAL
    variants = ("s", "m", "oracle", "xs", "xs-shared")
    default_variant = "s"

    def __init__(self, cache_dir: Path | None = None, repo: str = DEFAULT_REPO) -> None:
        """Point the loader at its cache and the HuggingFace source repo."""
        super().__init__(cache_dir)
        self.repo = repo

    def pull(self, variant: str | None = None) -> Path:
        """Download the variant's file from HuggingFace into the cache.

        A cut variant downloads the published file it is cut from.

        Raises:
            FileNotFoundError: if the repo has no file for that variant.
        """
        variant = self.resolve_variant(variant)
        variant = CUTS.get(variant, variant)
        repo = self.repo
        # file names differ across repo revisions; match by variant token, not filename
        token = "oracle" if variant == "oracle" else f"_{variant}"
        files = HfApi().list_repo_files(repo, repo_type="dataset")
        candidates = [
            f
            for f in files
            if f.startswith("longmemeval") and token in f and not f.endswith(".md")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"no longmemeval file matching variant {variant!r} in {repo}: {files}"
            )
        return Path(
            hf_hub_download(
                repo,
                sorted(candidates)[0],
                repo_type="dataset",
                cache_dir=self.cache_dir,
            )
        )

    def load(
        self, variant: str | None = None, limit: int | None = None
    ) -> list[Sample]:
        """Return LongMemEval questions normalized into list[Sample].

        `limit` caps the number of samples, for smoke runs; on `xs-shared` a
        sample is a whole history, so it counts histories, not questions.
        """
        variant = self.resolve_variant(variant)
        raw = json.loads(self.pull(variant).read_text())
        if variant == "xs-shared":
            return self._shared([_parse(item) for item in raw])[:limit]
        samples = []
        for item in raw[:limit]:
            sessions, qa = _parse(item)
            if variant == "xs":
                sessions = _cut(
                    sessions, qa.evidence_session_ids, XS_FILLERS, qa.question_id
                )
            samples.append(self._sample(qa.question_id, sessions, [qa]))
        return samples

    def _shared(self, parsed: list[tuple[list[Session], QAPair]]) -> list[Sample]:
        """Deal the questions into synthetic users sharing one history each.

        Abstention questions keep a history of their own, cut as in `xs`:
        another question's evidence could answer what they ask about.

        Raises:
            ValueError: if more than `SHARED_USERS` questions share a wording.
        """
        users: list[list[tuple[list[Session], QAPair]]] = [
            [] for _ in range(SHARED_USERS)
        ]
        alone = []
        cursor = 0
        for sessions, qa in parsed:
            if qa.question_id.endswith("_abs"):
                history = _cut(
                    sessions, qa.evidence_session_ids, XS_FILLERS, qa.question_id
                )
                alone.append(self._sample(qa.question_id, history, [qa]))
                continue
            # one wording asked twice of one user would have two gold answers
            for step in range(SHARED_USERS):
                user = (cursor + step) % SHARED_USERS
                if all(other.question != qa.question for _, other in users[user]):
                    break
            else:
                raise ValueError(
                    f"more than {SHARED_USERS} questions share the wording "
                    f"{qa.question!r}"
                )
            users[user].append((sessions, qa))
            cursor = user + 1
        samples = []
        for n, members in enumerate(users):
            if not members:
                continue
            sample_id = f"user_{n}"
            pool: dict[str, Session] = {}
            evidence: set[str] = set()
            for sessions, qa in members:
                evidence.update(qa.evidence_session_ids)
                for session in sessions:
                    pool.setdefault(session.session_id, session)
            history = _cut(list(pool.values()), evidence, SHARED_FILLERS, sample_id)
            samples.append(self._sample(sample_id, history, [qa for _, qa in members]))
        return samples + alone

    def _sample(
        self, sample_id: str, sessions: list[Session], qa: list[QAPair]
    ) -> Sample:
        """Wrap one history and the questions asked of it."""
        return Sample(
            sample_id=sample_id,
            dataset=self.name.value,
            conversation=Conversation(
                conversation_id=sample_id,
                speakers=["user", "assistant"],
                sessions=sessions,
            ),
            qa=qa,
        )


def _cut(
    sessions: list[Session], evidence: Collection[str], fillers: int, seed: str
) -> list[Session]:
    """Keep the evidence sessions plus `fillers` others drawn by `seed`, by date."""
    rest = [s for s in sessions if s.session_id not in evidence]
    chosen = random.Random(seed).sample(rest, min(fillers, len(rest)))
    kept = [s for s in sessions if s.session_id in evidence] + chosen
    # "YYYY/MM/DD (Day) HH:MM" sorts chronologically as a string
    return sorted(kept, key=lambda s: s.timestamp or "")


def _parse(item: dict) -> tuple[list[Session], QAPair]:
    """Normalize one record into its haystack and its question."""
    qid = str(item["question_id"])
    session_ids = [str(s) for s in item.get("haystack_session_ids", [])]
    dates = item.get("haystack_dates", [])
    answer_ids = {str(s) for s in item.get("answer_session_ids", [])}
    sessions = []
    evidence_turns = []
    for k, sess in enumerate(item.get("haystack_sessions", [])):
        sid = session_ids[k] if k < len(session_ids) else str(k)
        turns = []
        for j, msg in enumerate(sess):
            turn_id = f"{sid}:{j}"
            turns.append(
                Turn(
                    turn_id=turn_id,
                    speaker=msg.get("role", "user"),
                    text=msg.get("content", ""),
                )
            )
            if msg.get("has_answer"):
                evidence_turns.append(turn_id)
        sessions.append(
            Session(
                session_id=sid,
                timestamp=dates[k] if k < len(dates) else None,
                turns=turns,
            )
        )
    qa = QAPair(
        question_id=qid,
        question=item.get("question", ""),
        answer=str(item.get("answer", "")),
        category=item.get("question_type"),
        evidence_turn_ids=evidence_turns,
        evidence_session_ids=sorted(answer_ids),
        question_date=item.get("question_date"),
    )
    return sessions, qa
