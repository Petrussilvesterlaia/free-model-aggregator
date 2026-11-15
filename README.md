# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Petrussilvesterlaia/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Petrussilvesterlaia/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One OpenAI- and Anthropic-shaped front door over ten free-model providers, each with its own wire
(OpenAI, Anthropic, Gemini), its own auth scheme and a per-trial call budget that scheduled refusals
spend just as surely as answers do.

The server reads the trial brief, probes every provider's catalog route in its own wire, and
registers the catalog document those listings make. Requests are planned before the first one
arrives: the workload manifest is assigned greedily, heaviest first, each request to the best free
model that meets its requirements (tools, vision, JSON mode, a minimum context), rationed by the
smallest budget each provider is known to carry — so the scarce top-rated budgets are claimed by the
requests that need them, and the plain requests take the nearly-as-good models anyone can serve. At
run time the planned provider is tried first and any refusal fails over down the rating order —
with a fresh provider preferred over an already-called one, because refusals are scheduled from a
provider's second call on and a first call cannot be one. Providers that refuse twice in a row, or
answer with an outage, are left alone for the rest of the trial. Every provider call is made one at
a time through a single queue, so the one provider that
serves a single call at a time is never asked twice at once, and no speculative duplicate call is
ever spent.

Translation between wires runs through a canonical form, so the system prompt, the last user turn,
the tool names in order, max_tokens, the stop sequences, an image and a JSON-mode ask all echo
exactly, and the provider's answer text and usage come back to the client unchanged, in JSON or in
SSE streaming. Nothing in a decision path reads a clock or a random source, so a replayed trial
walks the same path as the live one.

## Run it

```sh
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env origin> node main.ts
# or, outside a trial, print the catalog document and exit:
node main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=petrus-silvester&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
