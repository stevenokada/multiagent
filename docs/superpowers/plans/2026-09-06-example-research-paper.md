# Example Research Paper Authoring Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution of this authoring task. No experimental software changes or new GPU runs are part of this plan.

**Goal:** Deliver a complete, editable example manuscript with exact completed methodology and explicitly proposed future experiments.

**Architecture:** A dated draft-v1 Markdown manuscript, self-contained HTML and PDF, with copied numerical evidence, two figures, provenance, and a portable ZIP. Existing report editions remain independent historical artifacts.

**Tech Stack:** Markdown, Python, Markdown2, Matplotlib, Playwright/Chromium and PyMuPDF.

**Spec:** `docs/probe-context-pilot-2026-09-06.md`, the user's research-paper request, and the exact runtime/scorer artifacts at commits c116f11 and d2f4adf.

## Global Constraints

- Present a completed exploratory context study; no simulated results or contagion claims.
- Distinguish relation polarity from prioritization, full reasoning quality, adoption and persistence.
- Verify exact prompts, personas, hidden-state storage/capture, scoring, aggregation, failures and model revisions against source artifacts.
- Cite primary literature. Keep all proposed sample sizes, estimands and acceptance criteria explicitly prospective.
- Preserve report V1–V4 and pre-existing uncommitted files.
- Standing user authorization covers drafting without additional permission requests.

## Tasks

- [x] Inspect actual runtime, prompts, configs, exported scores and operational drivers; verify primary references.
- [x] Draft abstract, introduction, related work, completed methods/results, limitations, prospective protocol, and exact-reproduction appendices.
- [x] Copy the empirical figure, generate a clearly labeled prospective protocol figure, and freeze the relevant evidence with hashes.
- [x] Export readable HTML/PDF/ZIP. Check all tables against JSON, figure loads, links, PDF pagination and clipping, and prospective wording.
- [x] Commit only the manuscript artifacts and plan. Return direct PDF and editable source links.
