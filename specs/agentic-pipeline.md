# Agentic TDD Pipeline

Automates: **spec (human) → tests → test review → implement → code review → done**.

One command. Walk away. Come back to green tests and reviewed code.

## Architecture

```
Bash orchestrator (run_pipeline.sh)
  ├── claude -p --agent test-writer    # sonnet — writes tests from spec
  ├── claude -p --agent reviewer       # opus  — reviews tests, returns JSON
  ├── claude -p --agent implementer    # sonnet — implements against frozen tests
  └── claude -p --agent reviewer       # opus  — reviews code, returns JSON
```

- **External script** owns the state machine, loop caps, and stage transitions
- **Subagents** own role-specific behavior, tools, and model choice
- **Hooks** enforce guardrails (run tests after edits, block protected files)
- **AGENTS.md** encodes repo-wide rules all agents inherit
- Spec approval stays **human-in-the-loop** — everything after is automated

## State Machine

```
SPEC_APPROVED (human)
  → TESTS_GENERATED
  → TESTS_REVIEWED ←─╮
  → TESTS_REVISED ────╯  (max 2 iterations)
  → TESTS_FROZEN
  → CODE_IMPLEMENTED
  → CODE_REVIEWED  ←─╮
  → CODE_REVISED  ────╯  (max 2 iterations)
  → DONE
```

## Subagents (`.claude/agents/`)

### test-writer.md

```md
---
name: test-writer
description: Writes tests from an approved spec. Never touches production code.
tools: Read, Write, Edit, Grep, Glob, Bash
model: sonnet
---

Read the approved spec. Generate unit tests covering all acceptance criteria.
Run `uv run pytest` to confirm tests fail (red) for unimplemented code.
Do not write production code. Output a coverage summary.
```

### implementer.md

```md
---
name: implementer
description: Implements production code against frozen tests and approved spec.
tools: Read, Write, Edit, Grep, Glob, Bash
model: sonnet
---

Read the spec and frozen tests. Implement the minimal production code to make
all tests pass. Run `uv run pytest` after changes. Do not modify frozen tests.
Prefer the smallest correct diff. If blocked, explain the blocker precisely.
```

### reviewer.md

```md
---
name: reviewer
description: Reviews tests or code for correctness, coverage, and spec alignment.
tools: Read, Grep, Glob, Bash
model: opus
---

Review the provided artifacts against the approved spec.
Do not edit any files. Produce findings grouped by severity:
- blocking: must fix before proceeding
- important: should fix
- optional: suggestions

State whether the artifact is acceptable or needs revision.
```

## Hooks (`.claude/settings.json`)

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "uv run ruff format --quiet ."
          }
        ]
      }
    ]
  }
}
```

## Review Stage Contract

Review stages use `--output-format json --json-schema` for machine-readable decisions:

```json
{
  "type": "object",
  "properties": {
    "decision": { "type": "string", "enum": ["approve", "revise"] },
    "summary": { "type": "string" },
    "blocking": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["decision", "summary"]
}
```

## Orchestrator (`scripts/run_pipeline.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?Usage: ./scripts/run_pipeline.sh <task-id>}"
SPEC="specs/${TASK}-spec.md"
MAX_REVISIONS=2

REVIEW_SCHEMA='{"type":"object","properties":{"decision":{"type":"string","enum":["approve","revise"]},"summary":{"type":"string"},"blocking":{"type":"array","items":{"type":"string"}}},"required":["decision","summary"]}'

review() {
  local prompt="$1"
  claude -p "$prompt" --agent reviewer \
    --output-format json --json-schema "$REVIEW_SCHEMA"
}

decision_of() {
  echo "$1" | jq -r '.result // . | .decision'
}

# --- Pre-check ---
[[ -f "$SPEC" ]] || { echo "Spec not found: $SPEC"; exit 1; }

# --- 1. Generate tests ---
echo "=== Stage: test generation ==="
claude -p "Read ${SPEC}. Write tests for task ${TASK}. Run uv run pytest to confirm red." \
  --agent test-writer

# --- 2. Review tests (bounded loop) ---
for ((i=0; i<=MAX_REVISIONS; i++)); do
  echo "=== Stage: test review (iteration $i) ==="
  RESULT=$(review "Review tests for ${TASK} against ${SPEC}.")
  echo "$RESULT" | jq .

  [[ "$(decision_of "$RESULT")" == "approve" ]] && break

  if ((i == MAX_REVISIONS)); then
    echo "FAIL: test revision cap reached"; exit 2
  fi

  echo "=== Stage: test revision ==="
  claude -p "Revise tests for ${TASK} based on reviewer feedback. Do not touch production code." \
    --agent test-writer
done
echo "=== Tests frozen ==="

# --- 3. Implement ---
echo "=== Stage: implementation ==="
claude -p "Read ${SPEC}. Implement code for ${TASK}. Tests are frozen — do not modify them. Run uv run pytest." \
  --agent implementer

# --- 4. Review code (bounded loop) ---
for ((i=0; i<=MAX_REVISIONS; i++)); do
  echo "=== Stage: code review (iteration $i) ==="
  RESULT=$(review "Review implementation for ${TASK} against ${SPEC} and frozen tests.")
  echo "$RESULT" | jq .

  [[ "$(decision_of "$RESULT")" == "approve" ]] && break

  if ((i == MAX_REVISIONS)); then
    echo "FAIL: code revision cap reached"; exit 3
  fi

  echo "=== Stage: code revision ==="
  claude -p "Revise implementation for ${TASK} based on reviewer feedback. Do not modify frozen tests." \
    --agent implementer
done

echo "=== Pipeline complete: ${TASK} ==="
```

## Guardrails

- **Revision caps**: 2 per loop (tests and code). Prevents drift.
- **Frozen tests**: implementer prompt forbids test edits; verify with `git diff --name-only` post-implementation.
- **Reviewer is read-only**: no Write/Edit tools. Findings only.
- **Spec is human-approved**: pipeline assumes spec already exists and is signed off.

## When to Use This

- **Good fit**: C milestone (6 replications of A1-A3/B1-B3 at 16L scale), repetitive tasks with clear specs.
- **Bad fit**: One-off math-heavy implementations (B2, B3) where domain judgment at each gate matters more than automation speed.
