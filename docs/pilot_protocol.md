# Pilot protocol: replacing the twenty-minute assumption with a number

**Tracked, ships.** `docs/discovery.md` assumes a manual statement review takes
about twenty minutes. Every "minutes saved" figure rests on that guess. This
protocol turns it into a measurement in about half a day, with three
relationship managers and a stopwatch. The platform already records its own
time; what it cannot record is how long a person takes.

## What is being measured

For each statement, one of two modes, timed from opening the statement to the
moment the RM has written down (a) the customer's monthly income, (b) whether
they are in cash-flow deficit, and (c) the product they would pitch.

| Mode | The RM works from | Stop the clock when |
|---|---|---|
| `manual` | The ledger only (CSV or PDF), as today | All three answers are written down |
| `assisted` | The BankLens profile, with the ledger available | All three answers are written down |

Minutes saved is the median manual time minus the median assisted time. The
report prints a measured saving only when both modes have been timed. With
only manual timings, it prints the measured manual time and keeps the saving
labelled as partial.

## Setup (15 minutes)

1. Pick **three RMs** with different tenure. Ask each for 60 to 90 minutes.
2. Pick **six statements** of 100 to 150 lines: two each of high saver, active
   spender and cash-flow stressed. Use real ones only under the bank's data
   handling rules; the synthetic samples in `data/` are fine for a dry run.
3. Split them so every RM does three manual and three assisted, and every
   statement is seen manual by one RM and assisted by another. Alternate the
   order (manual first for one RM, assisted first for the next) so practice
   does not favour one mode.
4. Copy `data/pilot/timings.example.csv` to `data/pilot/timings.csv`. That
   file is gitignored because it names people.

## During each session

- Start the stopwatch when the RM opens the statement. Do not help.
- Stop it when the three answers are written. Record minutes to one decimal.
- Write one line per statement in `data/pilot/timings.csv`:

```
session_date,rm_id,tenant,statement_ref,statement_lines,mode,minutes,notes
2026-10-01,rm-a,meridian,S-001,128,manual,21.5,read ledger end to end
```

- Use a code (`rm-a`), never a name, in `rm_id`.
- Note anything unusual (interrupted, statement unreadable) and drop that row
  from the median rather than guessing a time.
- After each assisted statement, ask one question and note the answer: "Did
  you disagree with anything in the profile?" Disagreements are more
  valuable than the minutes.

## Afterwards

```bash
make pilot TENANT=meridian
```

The report reads `data/pilot/timings.csv` for that bank and prints the
measured medians next to the platform's own timings. Then:

1. Put the measured manual minutes, assisted minutes and saving in
   `docs/numbers_card.md` with the date, the number of RMs and statements, and
   the command.
2. Change the "Minutes per statement review" row in `docs/discovery.md` from
   "Assumption" to "Measured".
3. Record any profile disagreements as golden-set cases in `evals/dataset.py`.
4. Close the pilot item in `TODO.md`.

## Honest limits of this design

- **Six statements and three RMs is a pilot, not a study.** It tells you
  whether twenty minutes is roughly right, not the true distribution.
- **Being watched changes behaviour.** Both modes are watched equally, so the
  difference is fairer than either absolute number.
- **Assisted time excludes the platform's own time.** The report prints
  seconds to a profile separately; add it if the RM waits for the profile.
