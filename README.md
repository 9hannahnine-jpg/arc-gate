# Arc Gate

**Real-time prompt injection detection and LLM monitoring proxy.**

Arc Gate sits in front of any OpenAI-compatible LLM endpoint and blocks prompt injection attacks before they reach your model. One environment variable to configure. No GPU required on your side.

## Benchmark

Evaluated on 40 out-of-distribution prompts using indirect, hypothetical, roleplay, and technical framings — the hardest category for existing systems:

| System | Precision | Recall | F1 |
| --- | --- | --- | --- |
| **Arc Gate** | **1.00** | **1.00** | **1.00** |
| OpenAI Moderation API | 1.00 | 0.75 | 0.86 |
| LlamaGuard 3 8B | 1.00 | 0.55 | 0.71 |

Arc Gate catches every harmful prompt in this category with zero false positives. OpenAI Moderation misses 1 in 4. LlamaGuard misses nearly half.

Average block latency: **329ms**. Blocked prompts never reach your model.

## What it does

- Sits in front of any OpenAI-compatible LLM endpoint
- Blocks prompt injection attempts via four detection layers:
  1. **Behavioral pre-filter** — SVM classifier trained on 200 labeled prompts. Catches indirect, roleplay, and technical framings. P=1.00, R=1.00, F1=1.00 on OOD holdout.
  2. **Phrase check** — 80+ injection patterns, zero latency
  3. **Geometric detection** — Fisher-Rao distance from clean prompt centroid
  4. **Session D(t) monitor** — catches multi-turn Crescendo-style attacks
- Logs all requests, drift events, and costs to SQLite
- Serves a live monitoring dashboard at `/dashboard`
- Detection overhead: ~350ms. Blocked prompts average 329ms and never reach your model.

## Deploy on Railway

1. Fork this repo
2. Create a new Railway project from this repo
3. Set environment variables:
   - `OPENAI_API_KEY` — your OpenAI key
   - `GATE_BLOCK_MODE` — `true` to enable blocking
   - `GATE_UPSTREAM` — `https://api.openai.com`
   - `GATE_BASE_URL` — your Railway URL
4. Railway will auto-deploy via the Procfile

## Try the demo

Dashboard: https://web-production-6e47f.up.railway.app/dashboard

## Related

- **Arc Sentry** — pip package for in-process prompt injection detection on self-hosted models
- **PyPI:** `pip install arc-sentry`

## License

Commercial license required for production use. Contact 9hannahnine@gmail.com.

Patent pending. Methods covered by provisional patent applications filed by Hannah Nine / Bendex Geometry LLC.

---

Bendex Geometry LLC · 2026 Hannah Nine · [bendexgeometry.com](https://bendexgeometry.com)
