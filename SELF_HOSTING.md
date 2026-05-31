# Self-Hosting Arc Gate

Arc Gate can be self-hosted on any server that can run Python 3.10+. This guide covers running Arc Gate on your own infrastructure.

## Requirements

- Python 3.10+
- 2GB RAM minimum (4GB recommended for all ML layers)
- PostgreSQL (recommended) or SQLite
- An OpenAI-compatible upstream LLM endpoint

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/9hannahnine-jpg/arc-gate.git
cd arc-gate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file or set these environment variables:

```bash
# Required
GATE_API_KEYS=ag-your-api-key-here        # Comma-separated list of valid API keys
UPSTREAM_URL=https://api.openai.com       # Your LLM endpoint

# Database (choose one)
DATABASE_URL=postgresql://user:pass@host:5432/dbname  # Recommended
# or leave unset for SQLite (development only)

# Optional
ARC_FAIL_MODE=fail_restricted             # fail_restricted | fail_open | fail_closed
ARC_POLICY_MODE=balanced                  # balanced | strict | browser_agent | finance_agent | rag_assistant
PORT=8080
```

### 3. Run Arc Gate

```bash
uvicorn arc_gate:app --host 0.0.0.0 --port 8080
```

### 4. Connect your agent

```python
from openai import OpenAI

client = OpenAI(
    api_key="ag-your-api-key-here",
    base_url="http://your-server:8080/v1"
)
```

### 5. View your Console

Open the Bendex Arc Console at [app.bendexgeometry.com/console](https://app.bendexgeometry.com/console) and enter your deployment ID and API key.

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["uvicorn", "arc_gate:app", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
docker build -t arc-gate .
docker run -p 8080:8080 \
  -e GATE_API_KEYS=ag-your-key \
  -e UPSTREAM_URL=https://api.openai.com \
  arc-gate
```

## Policy Modes

| Mode | Description | Use case |
|------|-------------|----------|
| `balanced` | General purpose | Most deployments |
| `strict` | Maximum enforcement | High-security environments |
| `browser_agent` | Web browsing agents | Agents that fetch URLs |
| `finance_agent` | Financial data agents | Agents with financial tool access |
| `rag_assistant` | Document retrieval | RAG pipelines |

## Fail Modes

| Mode | Behavior | Use case |
|------|----------|----------|
| `fail_restricted` | Strip tool calls, pass through | Default — preserves availability |
| `fail_open` | Pass request unchanged | Maximum availability |
| `fail_closed` | Return 503 | Maximum security |

## Data Residency

When self-hosting, all request data stays on your infrastructure. No data is sent to Bendex Geometry servers. The Console at app.bendexgeometry.com reads from your Railway/self-hosted Postgres via the sentry API — if you need fully air-gapped operation, host the Console separately.

## Support

Questions? Open an issue at [github.com/9hannahnine-jpg/arc-gate](https://github.com/9hannahnine-jpg/arc-gate) or email support@bendexgeometry.com.
