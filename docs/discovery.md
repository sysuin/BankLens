# Discovery: statement review in a bank's branch back office

**Requested by:** Head of Branch Banking (composite persona, see the honesty note)
**Owner:** Sunny Singh
**Date:** 2026-09-08 · **Status:** Phase 0 of the BankLens Platform plan (`Career/README.md`, Part 7)

> **Honesty note.** No real bank stakeholder was interviewed for this document. The persona is a
> composite of how relationship managers (RMs) and credit reviewers work, drawn from my own fintech
> and lending analytics experience and from public descriptions of retail-banking onboarding. Every
> number that is an assumption is labelled as one and has a "how I would validate it" column. In an
> interview I present this as the document I would write *before* a real engagement, not as field data.

---

## 1. The friction, in the stakeholder's words

> "An RM gets a customer's statements before every review meeting and before every product pitch.
> They read it line by line, work out what the customer earns and spends, and only then decide what
> to offer. When the declared income on the onboarding form does not match what the statement shows,
> it has to go to a reviewer, and that queue is a spreadsheet. Nobody can tell me how many of those
> we caught last quarter, or how many we missed."

## 2. What I checked before agreeing it was a problem

| Question | What I found | Source | Status |
| --- | --- | --- | --- |
| What does an RM actually do with a statement? | Two jobs: (a) prepare a cross-sell pitch grounded in the customer's real cash flow, (b) flag income or cash-flow discrepancies for review | Own experience; public RM role descriptions | Reasonable |
| How long does one manual review take? | **~20 minutes** for a 100–150 line statement: read, categorise mentally, estimate income and savings, pick products | Assumption | Validate with 3 RM shadowing sessions and a stopwatch |
| How many per RM per week? | **~40** (8 a day) in a mid-size branch | Assumption | Validate from the meeting calendar and the review-queue spreadsheet |
| What share need a human decision? | Most reviews are lookup and arithmetic. The genuine judgment call is the discrepancy case, assumed **10–15 %** | Assumption | Validate from one quarter of review-queue rows |
| What goes wrong today? | Inconsistent categorisation between RMs; credit products pitched to customers already in deficit; discrepancies caught late or not at all; no audit trail of who decided what | Own experience; the reranker A/B in this repo showed the deficit-pitch risk is real (harmful passages 0.40 → 0.91 when a reranker was enabled blindly) | Observed in the repo |
| What does the current BankLens do? | Parses CSV/PDF, masks PII, categorises, computes metrics deterministically, retrieves products, writes a profile and two recommendations, answers chat questions, runs as an MCP tool | Repo, measured 2026-09-08 | Measured |

**The important number is the 10–15 %.** Only the discrepancy cases need a person to *decide*.
Everything else is parsing, arithmetic and retrieval, which is exactly what a deterministic pipeline
with a narrating model is for.

## 3. Requirements

**Must**
1. Compute income, expenses, savings rate, risk band and health score **in code**, never in the model, so two RMs get the same answer for the same statement.
2. Mask PII **before** any text leaves the bank's boundary.
3. Ground every product recommendation in the bank's own catalogue with a **citation to the source section**, and never pitch unsecured credit to a customer in cash-flow deficit.
4. Compare **declared income** (onboarding) with **observed income** (statement) and route discrepancies above a threshold to a **human review queue**; the pipeline pauses and resumes from where it stopped when the reviewer decides.
5. Record **who decided what, on which inputs, with which prompt and model version**, queryable later.
6. Keep each bank's customers, catalogue and decisions **isolated** from any other tenant.
7. Say "I can't answer that from this statement" rather than guess. Non-negotiable.
8. Read-only against the statement and the catalogue. Decisions happen through the review queue.

**Should**
9. Answer numeric questions in chat through **vetted queries** over defined metrics, not free-form SQL.
10. Run with **no external API key** for demos and for banks that require on-premise inference, at a measured quality cost.
11. Trace every run so latency and cost per statement are **measured, not asserted**.
12. Handle bulk uploads (a branch's weekly batch) through a queue, with cost per statement reported.

**Won't (this phase)**
- An external customer-facing chatbot. Different threat model, identity, consent and consumer-regulation story; it would be a second system. The platform is built so it could be a later front door.
- Writing to any core-banking system.
- Real customer data. Every tenant, customer and statement is synthetic.

## 4. Success metrics

Agreed with the persona *before* building, so the result cannot be graded on a moving target.
Baselines are from `docs/numbers_card.md`, measured 2026-09-08.

| Metric | Baseline (today) | Target (end of Phase 7) | Where it comes from |
| --- | --- | --- | --- |
| Minutes per statement review (RM) | ~20 (assumption) | < 5, with the RM reading a profile instead of a ledger | Discovery validation; later `audit` table timestamps |
| Discrepancy cases reaching a reviewer | unknown (spreadsheet) | 100 % above threshold, with an audit row each | `decisions` table |
| Golden-set pass rate, deterministic layer | 65 / 65 | 65 / 65, plus new income-verification cases | `python -m evals.run_evals` |
| Grounded checks (sampled) | 5 blocking checks at 100 %; 1 advisory miss in 6 | 100 % blocking; advisory tracked | `--with-llm` |
| Retrieval hit@4 (hybrid, no reranker) | 1.000 (harmful 0.400 on deficit cases) | ≥ 0.95 with harmful ≤ 0.40 | `--retrieval-ab` |
| p50 / p95 latency, full analysis | 8.2 s / 14.6 s | p50 ≤ 8 s hosted; local model reported separately | `evals/baseline` |
| Cost per statement (hosted) | $0.0098 | ≤ $0.010, with cache savings reported | `evals/baseline` |
| Guardrail block rate (red-team suite) | none exists | ≥ 95 % of suite blocked | Phase 5 |
| Tenant isolation | none exists | test fails when the filter is removed | Phase 1 |
| Zero-key demo | not possible (OpenAI-only) | full demo on a local model | Phase 4 |

## 5. Risks named up front

| Risk | What I do about it |
| --- | --- |
| **Automation bias.** RMs stop reading and trust the profile. | The profile shows its numbers and sources; the discrepancy decision is never automated; reviewers see the raw comparison, not a verdict. |
| **PII leaving the boundary.** Vision OCR on scanned PDFs sends page images to an external API *before* masking. | Vision OCR stays opt-in and off by default; the data retention statement says so; a local OCR path is a candidate for Phase 4. |
| **Deficit customers pitched credit.** Already observed when a reranker was enabled without deficit awareness. | The credit guardrail is a blocking eval check; any retrieval change re-runs the A/B. |
| **A lower-quality local model.** | Reported side by side with the hosted model on the same suite; the demo says out loud where it is worse. |
| **The discrepancy threshold is arbitrary.** | Start at 20 % relative difference, log every case above 10 %, and tune from the reviewer's decisions. |
| **Synthetic data proves less than real data.** | Say so. The system's claims are about mechanism (reproducibility, isolation, audit, block rate), not about model accuracy on real customers. |

## 6. Decisions recorded

- Lead project: BankLens, promoted to a platform with one consumer (Part 7, §7.1).
- Internal chat only, two roles (RM, Reviewer). External chat scoped out.
- Two synthetic tenants with an isolation test.
- Income verification is the consequential decision.
- Metrics live in Postgres behind a semantic layer with row-level security; chat uses vetted templates.
- Gateway with OpenAI and Ollama; same golden suite on both.
