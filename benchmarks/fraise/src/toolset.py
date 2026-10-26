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

"""Fraise's native tool surface for `--mode agentic`: one raw FQL `query` tool.

The answering agent writes Fraise Query Language recalls itself, from the
spec in the toolset's instructions, and reads the server's errors and
warnings back to correct them. Ingestion stays the adapter's tagged
per-message `remember`; the graph, `top` and the vector are always the
adapter's, so isolation and the k budget never rest
on the agent.
"""

import time
from typing import Any

from fraise_sdk.errors import FraiseAPIError

from amb.agent.toolset import SearchToolset
from amb.base import Memory
from src.memory import FraiseMemory

# condensed from the fraise v0.3.5 FQL docs (docs/query-language), recall only
FQL_SPEC = """\
You search this conversation's memory with the `query` tool, which runs one \
Fraise Query Language (FQL) recall. Each stored fact is one conversation \
message, verbatim, as "<speaker>: <text>", filed under topics and entities.

Grammar (EBNF; the tool's own clauses are marked):

    recall_query = 'recall' recall_body ;
    recall_body  = ( term+ field* ) | field+ ;  (* needs one seed *)
    term         = bare_word | phrase ;         (* soft ranking seed *)
    field        = anchor | modifier ;
    anchor       = ( 'topic' | 'entity' ) ':' anchor_value ;  (* lift; repeatable *)
    anchor_value = bare_word | phrase ;         (* folded to lower case *)
    modifier     = since_field | until_field | depth_field ;  (* never a seed *)
    since_field  = 'since' ':' time_value ;     (* lower time bound *)
    until_field  = 'until' ':' time_value ;     (* upper time bound *)
    depth_field  = 'depth' ':' integer ;        (* retrieval lane, 0-2 *)
    (* set by the tool, never write them: the graph selector recall@N, *)
    (* 'top:' (result limit), 'vec:' (semantic vector of your terms)   *)
    bare_word    = letters and digits only ;
    phrase       = "'" { any char | "''" } "'" ; (* literal; '' is one apostrophe *)
    time_value   = integer ('s'|'m'|'h'|'d'|'w') | "'" YYYY-MM-DD "'" ;

Rules:
- Terms come first; after the first field a bare term is an error. Fields \
interleave in any order; only anchors may repeat (a repeated modifier is a \
duplicate-clause error).
- Several terms are a union: any match seeds the search. There is no AND, OR \
or NOT. An anchor beside terms filters nothing: it lifts the facts filed \
under it through the graph, and a match filed elsewhere still counts. Alone, \
anchors seed the search with the facts filed under them. An entity is a \
person or thing a message mentions, e.g. a speaker's name.
- Reserved words are syntax wherever a term can stand: recall, remember, \
forget, update, explain, describe, topic, entity, since, until, top, depth, \
vec. Quote one to search for it ('top'). After a field's ':' a reserved \
word is ordinary data (entity:top). Syntax is lower case.
- A bare word is letters and digits only; anything else (spaces, hyphens, \
punctuation, apostrophes) needs quotes: 'new york', 'machine-learning'. A \
phrase is the best way to ask a whole natural-language question, and its \
text is also embedded for semantic search.
- English stop words never match as bare terms (warning); a recall made only \
of stop words is an error. Inside a phrase they are fine.
- depth: 0 searches the text and vector indices only; 1 lets topics and \
entities that concentrate the matches pull in facts filed under them; 2 \
admits them at their fair share, for maximum recall. An anchor you name \
always lifts, so a recall with terms and an anchor runs at depth 1 at least \
(a written depth:0 warns). Omit it for the default.
- since/until filter by when a fact was stored. Every fact here was stored \
at ingestion, not when it was said, so they cannot narrow by conversation \
date; search for dates as terms instead.

Examples:
    recall 'when did Anna start her pottery class'
    recall pottery 'art class' entity:anna
    recall 'what does Ben think of his new job' depth:2

The tool returns the ranked hits as {content, score}. A rejected query \
returns its `error`, positioned by column in the `query` as sent; a query \
that ran but carries something it should not returns `warnings` beside its \
hits. Read them, fix the query, and search again.
"""


class FraiseSearchToolset(SearchToolset):
    """Fraise's raw FQL recall, exposed to the answering agent."""

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation and expose the query tool."""
        super().__init__(memory, conversation_id, k=k, instructions=FQL_SPEC, **kwargs)
        self.add_function(self.query, name="query")

    @property
    def fraise(self) -> FraiseMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, FraiseMemory)
        return self.memory

    def query(self, fql: str) -> dict:
        """Run one FQL recall over the conversation's memory.

        Args:
            fql: An FQL recall, e.g. `recall 'what did Anna paint' entity:anna`.

        Returns:
            The ranked `hits` and any `warnings`; or, for a rejected query,
            the `error` and the `query` as sent.

        Raises:
            FraiseAPIError: the server failed (5xx), not the query.
        """
        t0 = time.perf_counter()
        text, parameters = self.fraise.scope_query(self.conversation_id, fql, k=self.k)
        try:
            hits, warnings = self.fraise.query_hits(
                self.conversation_id, text, parameters
            )
        except FraiseAPIError as exc:
            # a server fault is the run's error, not the agent's to fix
            if exc.status_code >= 500:
                raise
            return {"error": exc.message, "query": text}
        result: dict = {"hits": self.record(hits, time.perf_counter() - t0)}
        if warnings:
            result["warnings"] = warnings
            result["query"] = text
        return result
