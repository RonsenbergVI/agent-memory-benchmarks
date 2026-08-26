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

"""LoCoMo (Maharana et al., ACL 2024): long multi-session conversations with QA.

No official HuggingFace release — the raw JSON is fetched from the
snap-research/locomo GitHub repository (override with AMB_LOCOMO_URL).
"""

import json
import urllib.request
from pathlib import Path

from amb.base import DatasetLoader
from amb.constants import Dataset
from amb.contracts import (
    Conversation,
    QAPair,
    Sample,
    Session,
    Turn,
)

DEFAULT_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

# Category codes from the LoCoMo paper's QA taxonomy.
CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


class LocomoLoader(DatasetLoader):
    """Loads the LoCoMo conversations and their QA annotations."""

    name = Dataset.LOCOMO
    variants = ("default",)
    default_variant = "default"

    def __init__(self, cache_dir: Path | None = None, url: str = DEFAULT_URL) -> None:
        """Point the loader at its cache and the raw JSON's source URL."""
        super().__init__(cache_dir)
        self.url = url

    def pull(self, variant: str | None = None) -> Path:
        """Download locomo10.json into the cache if it is not there yet."""
        self.resolve_variant(variant)
        path = self.cache_dir / "locomo10.json"
        if not path.exists():
            urllib.request.urlretrieve(self.url, path)
        return path

    def load(
        self, variant: str | None = None, limit: int | None = None
    ) -> list[Sample]:
        """Return LoCoMo conversations normalized into list[Sample].

        `limit` caps the number of samples, for smoke runs.
        """
        raw = json.loads(self.pull(variant).read_text())
        samples = []
        for i, item in enumerate(raw[:limit]):
            conv_raw = item["conversation"]
            sample_id = str(item.get("sample_id", i))
            speakers = [
                s for s in (conv_raw.get("speaker_a"), conv_raw.get("speaker_b")) if s
            ]
            sessions = []
            n = 1
            while f"session_{n}" in conv_raw:
                turns = [
                    Turn(
                        turn_id=t.get("dia_id", f"D{n}:{j}"),
                        speaker=t.get("speaker", "unknown"),
                        text=t.get("text", "")
                        + (
                            f" [shared image: {t['blip_caption']}]"
                            if t.get("blip_caption")
                            else ""
                        ),
                    )
                    for j, t in enumerate(conv_raw[f"session_{n}"])
                ]
                sessions.append(
                    Session(
                        session_id=str(n),
                        timestamp=conv_raw.get(f"session_{n}_date_time"),
                        turns=turns,
                    )
                )
                n += 1

            qa = []
            for j, q in enumerate(item.get("qa", [])):
                category = q.get("category")
                answer = q.get("answer", q.get("adv_answer"))
                evidence = q.get("evidence", [])
                qa.append(
                    QAPair(
                        question_id=f"{sample_id}:{j}",
                        question=q.get("question", ""),
                        answer=str(answer) if answer is not None else None,
                        category=CATEGORIES.get(category, str(category)),
                        evidence_turn_ids=[str(e) for e in evidence],
                        # LoCoMo evidence ids look like "D3:12" — session 3
                        evidence_session_ids=sorted(
                            {
                                str(e).split(":")[0].lstrip("D")
                                for e in evidence
                                if ":" in str(e)
                            }
                        ),
                    )
                )

            samples.append(
                Sample(
                    sample_id=sample_id,
                    dataset=self.name.value,
                    conversation=Conversation(
                        conversation_id=sample_id, speakers=speakers, sessions=sessions
                    ),
                    qa=qa,
                )
            )
        return samples
