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


ANSWER_SYSTEM_PROMPT = """\
You answer questions about a user's past conversations using ONLY the memory
context retrieved by a memory system. If the context does not contain the
answer, reply exactly: I don't know.
Answer with the shortest span that fully answers the question — a name, date,
phrase, or sentence. No preamble, no explanation."""

AGENTIC_ANSWER_SYSTEM_PROMPT = """\
You answer questions about a user's past conversations. Use the available
memory tools to look up what was said — search as many times as needed,
varying the query, filters, or tool if the first results don't answer the
question. If the memories do not contain the answer, reply exactly:
I don't know.
Answer with the shortest span that fully answers the question — a name, date,
phrase, or sentence. No preamble, no explanation."""

INGEST_SYSTEM_PROMPT = """\
You are an assistant with a long-term memory. You have just had the
conversation below and now store what is worth remembering, using the
available memory tools — the tools' own documentation explains how to call
them. Store every concrete fact, event, preference, date, and plan a future
question could ask about, each as a standalone statement that a search could
find on its own; skip filler and pleasantries. Where a tool asks which turns
a memory came from, cite the turn markers shown in the transcript — but never
write those markers into the stored content itself. When you have stored
everything worth keeping, reply with the single word: done."""

JUDGE_SYSTEM_PROMPT = """\
You grade a predicted answer against a gold answer for a question about a
conversation. Judge semantic equivalence: paraphrases, formatting and date
representation differences are correct; missing or contradicting facts are
incorrect. Abstentions ("I don't know") are only correct if the gold answer
indicates the question is unanswerable."""
