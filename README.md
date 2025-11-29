# Rubriq-Agent-as-a-Judge
Rubriq is a small agentic system that helps humans apply rubrics to project work. It assists both reviewers and learners.
Usage:
1. Paste rubric text that follows the simple pattern: category lines with totals, bullet lines with 'max N'.
2. Paste project writeup, optional readme, and short code snippets.
3. Call orchestrator.run_review(...).
4. Inspect scores, reasons, and learner feedback.

Design:
- AnalysisAgent: project summary, claims, rubric parsing, evidence mapping.
- ScoringAgent: per-item scores and overall comment.
- FeedbackAgent: strengths, improvements, self review questions.
- SimpleMemory: keeps recent reviews as a light long term store.
