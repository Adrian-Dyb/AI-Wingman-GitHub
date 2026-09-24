# AI-Wingman

AI-Wingman is a research prototype of a human-in-the-loop system
based on large language models (LLMs), designed to support users
in composing messages in online dating conversations.

The system was developed as part of a Master's thesis in Computer
Science (Artificial Intelligence) at Collegium Da Vinci.

## Master's thesis

**Title:**  
*Wykorzystanie dużych modeli językowych do wspomagania komunikacji interpersonalnej w aplikacjach randkowych – projekt i ewaluacja prototypu systemu konwersacyjnego*

**Author:** Adrian Dybziński

## System architecture

AI-Wingman follows a human-in-the-loop architecture:

User → Web Interface → FastAPI Backend → LLM → Guardrails → Suggestions → User Decision

The language model does not communicate autonomously with another
person. It generates three candidate responses that can be accepted,
edited or rejected by the user.

## Technology

- Python 3.11
- FastAPI
- OpenAI API
- HTML / JavaScript
- CSV-based experimental logging
- SciPy / statistical analysis

## Experimental evaluation

The study included:

- 20 participants
- 5 scenarios per participant
- 100 final messages
- 50 manual messages
- 50 AI-assisted messages
- 3 independent blinded judges
- 4 perceptual evaluation dimensions

## Main findings

No statistically significant differences were found between manual
and AI-assisted conditions in perceived quality, naturalness,
communicative attractiveness, intention to continue the conversation,
or heuristic open-question frequency.

AI-assisted messages were significantly shorter.

In 64% of AI-assisted trials, participants edited a selected AI
suggestion, while only 10% used a suggestion without modification.

## Safety layer

The prototype includes a multi-stage guardrail mechanism combining:

- prompt-level constraints,
- deterministic validation,
- one repair attempt,
- fail-closed behavior.

The guardrails reduce selected classes of undesirable outputs but
should not be interpreted as guaranteeing safe output.

## Repository structure

- `app/` – application source code
- `config/` – prompts and experimental configuration
- `analysis/` – reproducible statistical analysis
- `guardrail_validation/` – safety evaluation documentation
- `docs/` – figures and screenshots

## Data availability

Raw participant messages are not publicly distributed in this
repository. Publicly available files contain only selected aggregated
or non-identifying research outputs.

## Disclaimer

This repository contains a research prototype developed for academic
evaluation. It is not intended as a production dating platform or an
autonomous communication agent.