"""Prompt templates for LLM-based entity/relationship extraction."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are a knowledge-graph extraction engine.

Given the user's message and an optional neighbourhood of existing nodes,
produce a JSON object with exactly two keys:

  "nodes" – array of node instructions
  "relationships" – array of relationship instructions

### Node instruction schema
{
  "operation": "create" | "update",
  "label": "<PascalCase Neo4j label>",
  "merge_keys": {"<key>": "<value>"},     // unique identifiers
  "properties": {"<key>": <value>, ...},   // additional data
  "summary": "<one-sentence description>",
  "cluster_hint": "<topic category>" | null,
  "supersedes_ref": "<elementId of node this replaces>" | null
}

### Relationship instruction schema
{
  "from_ref": "<elementId or temp_N>",
  "to_ref": "<elementId or temp_N>",
  "type": "UPPER_SNAKE_CASE"
}

Rules:
- Use temp_0, temp_1, … as references for newly created nodes (in order).
- Reference existing nodes by their elementId when linking to neighbourhood.
- Only include factually grounded entities; ignore filler/conversational text.
- Return valid JSON only, no markdown fences.
"""


def build_user_prompt(
    text: str,
    neighbourhood: list[dict[str, Any]],
) -> str:
    """Compose the user-facing prompt with text and neighbourhood context."""
    parts = [f"User message:\n{text}"]
    if neighbourhood:
        lines = []
        for node in neighbourhood:
            eid = node.get("elementId", "?")
            label = node.get("label", "?")
            props = {k: v for k, v in node.items() if k not in ("elementId", "label", "_embedding")}
            lines.append(f"  [{eid}] :{label} {props}")
        parts.append("Existing neighbourhood:\n" + "\n".join(lines))
    return "\n\n".join(parts)
