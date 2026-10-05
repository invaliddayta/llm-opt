"""Explicit next-cohort configuration; previous presets and target stay unchanged."""
PROFILES = ("original", "balanced-v1", "focused-v1", "pragmatic-v1", "direct-code-v1", "reasoned-code-v1")


def preset(name, profile, original_preset, sampling_body):
    if profile not in PROFILES:
        raise ValueError("Unknown explicit cohort profile")
    if profile == "original":
        return original_preset(name)
    if profile == "reasoned-code-v1":
        _, body = preset(name, "direct-code-v1", original_preset, sampling_body)
        code = name in ("python_tools_long", "python_tools")
        if code:
            body = body | {"chat_template_kwargs": body["chat_template_kwargs"] | {"enable_thinking": True}}
        return "reasoned-" + ("code-v1" if code else "prose-v1"), body
    if name not in ("story", "explanation", "python_tools_long", "python_tools"):
        raise ValueError("Profiled cohorts require the representative original tasks")
    temperature = 0.4 if name in ("story", "explanation") else 0.0
    variant = "balanced-prose-v1" if name in ("story", "explanation") else "balanced-code-v1"
    effort = "xhigh"
    thinking = "on"
    if profile in ("focused-v1", "pragmatic-v1", "direct-code-v1"):
        prefix = {"focused-v1": "focused", "pragmatic-v1": "pragmatic", "direct-code-v1": "direct"}[profile]
        variant = prefix + ("-prose-v1" if name in ("story", "explanation") else "-code-v1")
        effort = "xhigh" if name in ("story", "explanation") else "low"
    if profile in ("pragmatic-v1", "direct-code-v1"):
        temperature = 0.4
    if profile == "direct-code-v1" and name in ("python_tools_long", "python_tools"):
        temperature, thinking = 0.2, "off"
    return variant, sampling_body(temperature, thinking) | {"reasoning_effort": effort}


def agent(name, profile):
    if profile not in PROFILES:
        raise ValueError("Unknown explicit cohort profile")
    if profile == "original":
        return "build"
    return "sol-bench-prose" if name in ("story", "explanation") else "sol-bench-code"


def agents(profile):
    if profile == "original":
        return {}
    if profile == "reasoned-code-v1":
        return agents("direct-code-v1")
    if profile not in ("balanced-v1", "focused-v1", "pragmatic-v1", "direct-code-v1"):
        raise ValueError("Unknown explicit cohort profile")
    result = {
        "sol-bench-prose": {"description": "Owned prose benchmark without tools", "mode": "primary",
                            "permissions": [{"action": "*", "resource": "*", "effect": "deny"}],
                            "system": "Write only the complete prose answer requested by the user. Do not use tools, count words with code, create files, ask questions, or delegate."},
        "sol-bench-code": {"description": "Owned coding benchmark with bounded workflow", "mode": "primary",
                           "permissions": [{"action": action, "resource": "*", "effect": "deny"}
                                           for action in ("subagent", "external_directory", "question", "webfetch", "websearch")],
                           "system": "Make a brief plan, implement the requested program and meaningful tests, run tests, fix actual implementation defects, then stop with a short summary. Stay in the active workspace; no network, outside paths, or delegation. Never make an edit with identical old and new text. Do not repeat an unchanged successful probe. Do not weaken a correct test merely to obtain a pass. Preserve real validation and useful errors; no placeholder code, padding, or compressed formatting."},
    }
    if profile in ("focused-v1", "pragmatic-v1", "direct-code-v1"):
        result["sol-bench-prose"]["system"] += " In recovery explanations, distinguish particular checkpoint designs, retained earlier undo/redo history, and surviving data pages from backup-based restoration; do not assert a blanket log cutoff."
        result["sol-bench-code"]["system"] += " Keep private reasoning focused and act with tools promptly. Include negative and regression tests, not only happy paths. Validate complete stored records and CSV row/header structure. Keep file/database writes atomic, preserve exact money and text round trips, reconcile transfers across at least three accounts, and report expected parsing, decoding, and storage errors cleanly. Stop after the required tests pass."
    if profile in ("pragmatic-v1", "direct-code-v1"):
        result["sol-bench-prose"]["system"] += " State the persistence invariant: durable WAL precedes durable changed data pages. Durable commit acknowledgment follows a flush of the commit record and required earlier WAL; durable group commit batches that flush, whereas asynchronous acknowledgment before it relaxes durability. Retain both dirty-page redo history, including earlier committed updates, and active-transaction undo history. Ensure surviving-page examples agree with the surviving log. An ordinary live checkpoint cannot supply changed pages to an older restored backup or shorten its required log history by itself."
        result["sol-bench-code"]["system"] += " Preserve existing programs and tests belonging to other tasks. If the task lets you choose the program, use new task-specific filenames and do not overwrite budget.py or test_budget.py. Approximate line counts are not exact limits: do not cycle through cosmetic shortening or repeated line-count probes. Once meaningful tests pass, give the requested summary and stop."
    if profile == "direct-code-v1":
        result["sol-bench-prose"]["system"] += " Track scene location and time explicitly: journeys precede arrival, inspecting a bicycle left in the shop happens after returning, and recovered objects are placed there only after arrival. Fuzzy checkpoint metadata locates the earliest required redo record, which can precede the checkpoint; the analysis and redo boundaries need not coincide."
        result["sol-bench-code"]["system"] += " Act with implementation and regression-test edits, not merely a disposable probe or proposed fix. For SQLite, acquire the write lock before checking funds or migration schema, recheck schema under it, preserve legacy records, and use Python integer aggregates instead of overflowing SUM or floating-point TOTAL. Validate supported ASCII numeric syntax, normalize leading zeros before bounded conversion, and handle storage/calendar limits cleanly. Parse CSV strictly, validate row width before blank-cell handling, support LF/CRLF/CR and exact descriptions, and report physical reader line numbers. If the user lets you choose a small program, choose a useful purpose suited to about 100 lines, not another large ledger. If persistence is used, validate every stored account/transaction record, references, required fields, unique identities, next-ID invariants and export destination aliases before operations. Keep all test temporary directories explicitly inside the workspace. Coordinate concurrent-lock regression tests with bounded acquisition events and joins, not sleeps as evidence. Preserve real command exit statuses and assert negative CLI outcomes. Complete cleanup and all required edits before the final complete test suite; after it passes, issue only the requested summary and no further tool calls."
    return result
