# Arc Gate

Proxy server for LLM deployments with real-time prompt injection detection.

Part of the [Bendex Geometry](https://bendexgeometry.com) security suite.

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
   - `SENTRY_BLOCK_MODE` — `true`
   - `SENTRY_UPSTREAM` — `https://api.openai.com`
4. Railway will auto-deploy via the Procfile

## Related
- [arc-sentry](https://github.com/9hannahnine-jpg/arc-sentry) — pip package for in-process detection
- [PyPI: arc-sentry](https://pypi.org/project/arc-sentry/)
