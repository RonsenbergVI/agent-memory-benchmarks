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

Each instance is one question over its own haystack of chat sessions, so each
becomes one Sample. Variants: `s` (~115k-token haystacks, default),
`m` (~500 sessions), `oracle` (evidence sessions only).
"""

import json
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

# The originally released xiaowu0162/longmemeval is deprecated in favor of
# the -cleaned repo (noisy history sessions removed).
DEFAULT_REPO = "xiaowu0162/longmemeval-cleaned"


class LongMemEvalLoader(DatasetLoader):
    """Loads LongMemEval questions with their haystack sessions."""

    name = Dataset.LONGMEMEVAL
    variants = ("s", "m", "oracle")
    default_variant = "s"

    def __init__(self, cache_dir: Path | None = None, repo: str = DEFAULT_REPO) -> None:
        """Point the loader at its cache and the HuggingFace source repo."""
        super().__init__(cache_dir)
        self.repo = repo

    def pull(self, variant: str | None = None) -> Path:
        """Download the variant's file from HuggingFace into the cache.

        Raises:
            FileNotFoundError: if the repo has no file for that variant.
        """
        variant = self.resolve_variant(variant)
        repo = self.repo
        # File naming differs across repo revisions (longmemeval_s,
        # longmemeval_s.json, ...), so match by variant token instead of
        # hardcoding a filename.
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

        `limit` caps the number of samples, for smoke runs.
        """
        raw = json.loads(self.pull(variant).read_text())
        samples = []
        for item in raw[:limit]:
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

            samples.append(
                Sample(
                    sample_id=qid,
                    dataset=self.name.value,
                    conversation=Conversation(
                        conversation_id=qid,
                        speakers=["user", "assistant"],
                        sessions=sessions,
                    ),
                    qa=[
                        QAPair(
                            question_id=qid,
                            question=item.get("question", ""),
                            answer=str(item.get("answer", "")),
                            category=item.get("question_type"),
                            evidence_turn_ids=evidence_turns,
                            evidence_session_ids=sorted(answer_ids),
                            question_date=item.get("question_date"),
                        )
                    ],
                )
            )
        return samples
