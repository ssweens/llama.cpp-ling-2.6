# Lessons

- 2026-05-01: When the user says to commit only on “positive progress” during model-bringup, interpret that as **coherent/generated-output progress**, not merely diagnostic probes, documentation, or smoke-test progress. Do not commit or push quality-debug changes unless the model produces coherent output or the user explicitly asks for that specific commit.
- 2026-05-01: Do **not** launch pi subagents, tmux background agents, Docker-backed helper processes, or long-running background GPU jobs without explicit user permission. Never start subagents while GPU/driver state is unstable or after a runtime hang; keep debugging in the main session unless the user asks for delegation.
