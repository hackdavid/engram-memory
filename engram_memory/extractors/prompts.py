"""Prompt templates for LLM-based entity/relationship extraction."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are a knowledge-graph extraction engine.

Given the user's message and an optional context of the top-5 most semantically
similar existing nodes (with their relationship types), produce a JSON object
with exactly two keys:

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
- Use temp_0, temp_1, ... as references for newly created nodes (in order).
- Reference existing nodes by their elementId when linking to context nodes.
- If the message refers to an entity already present in the context nodes
  (same name or identifier), emit "operation": "update" with that node's
  existing merge_keys so the MERGE matches the existing node. Only use
  "create" for genuinely new entities not found in the context.
- When updating, include only the properties that changed or are new;
  omit unchanged properties to keep the payload small.
- Only include factually grounded entities; ignore filler/conversational text.
- Return valid JSON only, no markdown fences.
"""


def build_user_prompt(
    text: str,
    neighbourhood: list[dict[str, Any]],
) -> str:
    """Compose the user-facing prompt with text and slim neighbourhood context.

    Each neighbourhood entry is expected to have:
        elementId, label, summary, rel_types (list of relationship type strings).
    Only summary and relationship types are included — no raw properties or
    embedding vectors — to keep token usage minimal.
    """
    parts = [f"User message:\n{text}"]
    if neighbourhood:
        lines = []
        for node in neighbourhood:
            eid = node.get("elementId", "?")
            label = node.get("label", "?")
            summary = (node.get("summary") or "").strip()
            rel_types: list[str] = node.get("rel_types") or []
            rel_str = ", ".join(rel_types) if rel_types else "none"
            lines.append(f'  [{eid}] :{label} "{summary}" -- {rel_str}')
        parts.append("Existing context (top-5 similar nodes):\n" + "\n".join(lines))
    return "\n\n".join(parts)
