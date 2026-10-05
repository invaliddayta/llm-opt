# GPU sampling with tool grammar

Deployed, opt-in with `LLAMA_GPU_SAMPLING=1`.

## Why

OpenCode attaches a lazy XML tool-call grammar to every request. Upstream llama.cpp at this
revision disables backend (GPU) sampling whenever a grammar is present, so each speculative
step copies the target logits (8 rows x 248,320 floats = 7.9 MB) to the host and samples on the
CPU. Over the USB4 eGPU link that is ~1.2 ms per ~36 ms step.

## Design

- Grammar to DFA: at request setup, the CPU-parsed GBNF stack states are compiled into an
  exact finite automaton with character equivalence classes. Bounded regular subset only.
  Grammars with stack-growing recursion or token terminals are rejected at setup, and the
  request falls back to standard sampling. The real 11-tool OpenCode grammar: 6494 states,
  70 classes, 454,580 transitions, 1.82 MB table.
- Mask on the GPU: a dedicated ggml op (`GGML_OP_GRAMMAR_MASK`) walks each vocabulary piece
  through the DFA with the same UTF-8 semantics as the CPU grammar (EOG, empty pieces, inverse
  classes, partial UTF-8) and masks the logits.
- Speculative rows: persistent base state holds only committed tokens. The first verify row
  applies pending committed tokens, then copies base to a working state; later rows advance the
  working state by the previous row's sampled token. Rows after a mismatch are discarded and
  never touch base.
- Lazy triggers and reasoning: one literal word or exact token trigger (the server's
  `<tool_call>`) and unlimited reasoning-tag suppression live in device state.
- RNG: device Philox4x32-10 with its counter in GPU state. Seeded outputs differ from the CPU
  MT19937 path; the distribution is the same.
- State handling: accepted-token packets carry 64-bit stamps so a replayed packet is
  idempotent. Clones share immutable tables and own their mutable state; graph reserve/probe
  never commits pending tokens.

## Fallback

Requests the GPU sampler cannot serve (penalties, logprobs, reasoning budget, regex triggers,
non-lazy grammar prefill) use standard sampling for that request only.

## Validation

- Masks: 6,208,000 CPU/GPU vocabulary-mask comparisons on the active grammar, 11,919,360 on
  lazy/reasoning cases.
- Edge cases: tags crossing token boundaries, reasoning start/end replay, malformed UTF-8, EOS,
  repeated graph replays, clone/copy/reset, zero and full acceptance, sleep/wake.
- Server: `bench/compat_check.py` 9/9 (penalties, logprobs, JSON schema, tools, image).
