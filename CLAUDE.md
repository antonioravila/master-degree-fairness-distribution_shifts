
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Project context

**README.md is the source of truth** for the pipeline, baselines, CLI, output folders and tools. Read it before touching any of those, and don't duplicate its content here.

Rules that follow from how the project works:
- **Folder names are cache keys.** After changing a baseline's logic, existing result folders will be silently reused — tell the user; don't delete them yourself. Any new parameter that changes output must be encoded in the folder name.
- **Changing the output schema** (folder names or CSV columns) requires a one-off `src/tools/migrate_*.py` script and updating the regexes in `src/tools/concatenate_results.py`.
- **Preprocessing cache** is keyed only on the functions listed in the README. Preprocessing logic changed elsewhere does not invalidate it — use `use_cache=False` or warn the user.
- **`src/results/` is committed** and represents hours of compute. Confirm with the user before deleting any folder there. After changing results, rerun `concatenate_results.py` so `src/results/all/` stays in sync.

Verification (complements section 4): there is no test suite. "Verified" means `uv run ruff check src/` is clean, a run with `--test_run True --alpha 0.5` finishes without errors, and the generated CSV/chart has been checked. For chart-only changes, use `src/tools/regenerate_charts.py`.

Workflow and style:
- Large changes start with a design doc in `docs/*_plan.md` (in Portuguese) before any code.
- Changes to the CLI, outputs or pipeline update README.md in the same change.
- Ruff uses single quotes (`quote-style = "single"`).
