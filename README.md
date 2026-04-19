# Arc Gate

Real-time prompt injection detection and LLM monitoring proxy.
Part of the **Bendex Geometry** security suite.

## What it does

- Sits in front of any OpenAI-compatible LLM endpoint
- Blocks prompt injection attempts via phrase detection + Fisher-Rao geometric monitoring
- Logs all requests, drift events, and costs to SQLite
- Serves a live monitoring dashboard at `/dashboard`

## Deploy on Railway

1. Fork this repo
2. Create a new Railway project from this repo
3. Set environment variables:
   - `OPENAI_API_KEY` — your OpenAI key
   - `GATE_BLOCK_MODE` — `true` to enable blocking
   - `GATE_UPSTREAM` — `https://api.openai.com`
   - `GATE_BASE_URL` — your Railway URL
4. Railway will auto-deploy via the Procfile

## Related

- **Arc Sentry** — pip package for in-process training stability detection
- **PyPI: bendex** — `pip install bendex`
