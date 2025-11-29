# Rubriq-Agent-as-a-Judge

Agentic rubric assistant built with Google ADK.

Rubriq is a small agentic system that helps humans apply rubrics to project work. It assists both reviewers and learners.

Architecture
------------
- AnalysisAgent (LLM):   infers criteria from free-form rubric + project + code, and summarises the project.
- ScoringAgent  (LLM):   scores each inferred criterion with reasons.
- FeedbackAgent (LLM):   turns summary + scores into a final JSON result.
- PipelineAgent (SequentialAgent): runs Analysis -> Scoring -> Feedback in strict order.
- Orchestrator  (LLM):   uses PipelineAgent as a tool; exposes a single entrypoint.

Inputs (all free-text strings):
    rubric_text      : free-form rubric or guidelines
    project_writeup  : free-form project writeup
    code_text        : Python code (string)

Basic memory:
- InMemorySessionService is used.
- Orchestrator + pipeline + sub-agents all share (user_id, session_id) context via ADK.
