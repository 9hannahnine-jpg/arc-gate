# Bendex Arc — OWASP LLM Top 10 Coverage

This document maps Bendex Arc's runtime governance capabilities against the OWASP Top 10 for Large Language Model Applications (2025) and OWASP Top 10 for Agentic Applications (2026).

---

## OWASP Top 10 for LLM Applications (2025)

| # | Risk | Arc Gate Coverage | Details |
|---|------|-------------------|---------|
| LLM01 | Prompt Injection | Full | Single-turn and multi-turn Crescendo attacks. Authority state machine blocks instruction-authority transfer from untrusted sources. Geometric drift detection catches gradual session manipulation. |
| LLM02 | Sensitive Information Disclosure | Partial | Capability revocation prevents agents from accessing or exfiltrating data when session risk is elevated. Arc Replay provides full audit trail of what data was accessed. |
| LLM03 | Supply Chain | Out of scope | Arc Gate operates at runtime, not at model or dependency supply chain level. |
| LLM04 | Data and Model Poisoning | Partial | Arc Memory monitors for identity drift and memory manipulation across multi-turn sessions. |
| LLM05 | Improper Output Handling | Full | All tool results and model outputs pass through governance before reaching downstream systems. Blocked outputs never reach the execution layer. |
| LLM06 | Excessive Agency | Full | Capability revocation in fail_restricted and fail_closed modes. Tool calls stripped from model responses when session risk exceeds threshold. Arc Approve adds human-in-the-loop before high-risk actions. |
| LLM07 | System Prompt Leakage | Partial | Authority state machine flags requests attempting to probe or extract system prompt contents. |
| LLM08 | Vector and Embedding Weaknesses | Partial | Arc Gate inspects retrieved documents before they reach the model. Does not operate at the embedding layer. |
| LLM09 | Misinformation | Out of scope | Arc Gate governs instruction authority, not factual accuracy. |
| LLM10 | Unbounded Consumption | Partial | Rate limiting on demo keys. Fail_closed mode prevents downstream requests when governance is unavailable. |

---

## OWASP Top 10 for Agentic Applications (2026)

| # | Risk | Arc Gate Coverage | Details |
|---|------|-------------------|---------|
| AA01 | Prompt Injection | Full | Direct and indirect injection across all source types: user input, tool output, webpage, email, document, database row. Multi-turn Crescendo detection via CUSUM session tracking. |
| AA02 | Excessive Agency | Full | Capability revocation at proxy level. Tool calls stripped before model response reaches execution layer. Arc Approve adds approval gate before high-risk actions. |
| AA03 | Insecure Memory & RAG | Partial | Arc Memory monitors behavioral drift across sessions. Retrieved documents treated as untrusted by default. |
| AA04 | Insufficient Monitoring | Full | Every request logged with full decision trace: layer, signal, score, session risk, authority decision. Arc Replay provides queryable session history. CSV audit export for SIEM integration. |
| AA05 | Improper Output Handling | Full | Tool results inspected before model sees them. Blocked content returns error response, never reaches downstream execution. |
| AA06 | Insecure Agent Cooperation | Partial | CAIAT benchmark demonstrates Arc Gate catches cross-agent instruction authority transfer in 81% of tested scenarios vs 50% for LLM Guard. |
| AA07 | Prompt Chaining Vulnerabilities | Partial | Session-level tracking catches chained attacks that span multiple turns. |
| AA08 | Context Manipulation | Full | Authority state machine enforces source-based trust hierarchy. External content cannot override developer-defined system instructions. |
| AA09 | Inadequate Human Oversight | Full | Arc Approve provides configurable human-in-the-loop before high-risk tool calls. Fail modes ensure graceful degradation when governance is unavailable. |
| AA10 | Uncontrolled Third-Party Tools | Full | Arc Gate MCP wraps any MCP server. Every tool result from third-party sources passes through governance before reaching the agent. |

---

## Coverage Summary

| Framework | Full Coverage | Partial Coverage | Out of Scope |
|-----------|--------------|-----------------|--------------|
| OWASP LLM Top 10 (2025) | 5/10 | 3/10 | 2/10 |
| OWASP Agentic Top 10 (2026) | 7/10 | 3/10 | 0/10 |

---

## Benchmark Results

| Benchmark | Result |
|-----------|--------|
| AgentDojo v1 (ETH Zurich, ICLR 2024) | 100% unsafe action prevention, 0% FPR |
| InjecAgent (U. Illinois, ACL 2024) | 99% TPR blind test |
| CAIAT Cross-Agent Injection | 81% vs LLM Guard 50% |
| TAB Platform Independent Verification | 25/25 (100%) vs 76% baseline |

---

## References

- [OWASP Top 10 for LLM Applications 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications/)
- [Bendex Arc Platform](https://bendexgeometry.com)
- [Arc Gate GitHub](https://github.com/9hannahnine-jpg/arc-gate)
