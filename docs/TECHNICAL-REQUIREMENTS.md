# The MLOps control plane, explained for someone outside software

This is a report on a system that manages the lifecycle of machine-learning models inside a
company: deciding which model version is allowed to serve real traffic, enforcing rules about
when a change is safe to release, keeping a tamper-evident record of every decision, and
watching for the ways things quietly go wrong afterward. Think of it as the control tower and the
flight-recorder for a fleet of models, rather than the aircraft themselves — it doesn't do the
machine learning; it decides which trained model is trusted to run, under what conditions, and
keeps an honest record of that decision so it can be checked later.

It currently stands at about 19,000 lines of code, all written and verified inside this project,
and it is roughly three-quarters custom-built and one-quarter built on established, off-the-shelf
tools. That ratio is itself a design choice, and the rest of this report explains why.

## What the system is made of

Picture the system in three layers, like a building: a foundation that nobody sees but everything
stands on, a set of rooms with specific jobs, and a front desk that the outside world actually
talks to.

**The foundation** is the part that decides what "correct" and "safe" mean for this system, and it
was deliberately built from scratch rather than borrowed. It keeps a chained, signed log of every
action taken — every model registered, every policy changed, every incident opened — structured so
that if anyone tried to alter an old entry, the chain would visibly break, the same way a forged
page in a bound ledger disturbs the binding. It holds the rules that decide whether a new model
version is allowed to go live: things like "this model must pass its validation tests," "a looser
safety rule can never be activated without a human's sign-off," and "a tenant gets a guaranteed
share of capacity even under load." And it holds a simulation of what happens when that capacity
runs out — not a live production test, but a careful, honest model of the failure mode, built so
the team can reason about it before it happens for real. None of this foundation depends on any
third-party library. That was intentional: this is the part of the system where a bug is most
expensive, and where the team wanted to understand and control every line rather than trust an
outside package's behavior.

**The rooms** are where the system talks to the outside world and keeps its books. One room is a
web service — the front desk anyone can reach over the network, built on a widely-used, well-
regarded web framework rather than hand-rolled, because a front desk that speaks a standard
protocol correctly is a solved problem and not worth re-solving. It checks who's asking, what
they're allowed to do, and hands the request to the foundation underneath. Another room watches
the system from the outside: it collects statistics (how many models are registered, how the audit
log is growing, how many incidents are open) in a format that standard monitoring tools already
know how to read, and it traces requests as they move through the system so that if something is
slow or fails, there's a trail to follow — again using a standard, widely-adopted tool rather than
inventing our own tracing format. A third room is where model and dataset records are actually
saved to disk in an ordinary database, using a well-known database toolkit, because that is a
mature, unglamorous problem best left to mature, unglamorous tools. The tamper-evident log
described above deliberately does NOT live in that same database — it's kept in its own, custom,
crash-tested storage, because the properties that log needs (nothing can be silently altered, ever,
even after a crash mid-write) are stronger than what an ordinary database promises out of the box,
and the team was not willing to bet that guarantee on a tool built for a different job.

**The front desk** also includes a command-line tool and a small client library, so a person or
another program can talk to the web service without knowing its internal details — the equivalent
of a switchboard operator standing between a caller and the department they actually need.

## Why it's built this way

The single biggest decision in this project was where to draw the line between "build it
ourselves" and "use an established tool," and the reasoning was consistent throughout: **use an
established tool wherever the problem is a solved, general one; build it ourselves wherever the
correctness of the answer is the whole point.**

Handling web requests correctly — parsing them, matching routes, returning proper error codes — is
a solved problem, and the team adopted a standard framework for it rather than re-deriving that
work. The same reasoning applied to statistics: rather than hand-writing the formulas behind a
particular significance test, the team switched to a well-established statistics library, because
during that switch they discovered their own hand-written version had a subtle error in how it
handled tied values — a mature library maintained by a wide community had already found and fixed
that exact class of mistake. That's the clearest case in the whole project of "somebody else has
already solved this better than we would have."

The opposite reasoning applied to the tamper-evident record-keeping and the safety-rule engine.
Those are not solved problems in any general library — they encode this specific company's
specific idea of what "safe" and "provable" mean, and getting that logic exactly right, in a form
the team can fully audit, mattered more than saving the time of writing it. So it was built by
hand, and tested unusually hard: every rule was written down as a strict expectation BEFORE any
code was written to satisfy it, so that the test could never quietly bend to match whatever the
code happened to do.

The cost of this split is real and worth naming plainly: it means the team maintains more
hand-written code than a typical project its size would, and every one of those hand-written pieces
is a piece nobody else is maintaining or hardening for us. The team judged that cost acceptable
because the alternative — trusting a general-purpose library with the one part of the system where
"trust, but verify" is the entire product — felt like the wrong kind of convenience.

## What went wrong, and what it taught the team

No system this size gets built without mistakes, and a few are worth telling plainly because each
one changed how the team works afterward, not just what the code says.

Early on, a simulation of what happens when the system runs low on capacity was quietly built so
that its test data came partly from the answer it was supposed to be testing — like grading an exam
using the answer key as one of the inputs. It was caught before shipping and rebuilt so the
simulation only ever sees the same evidence a real decision-maker would see. The lesson the team
took from it: a simulation is only honest if it could, in principle, come out wrong.

Later, a piece of the safety-rule engine checked things in the wrong order — it consulted a general
safety policy before checking whether the person asking was even authorized to make the request at
all, which is backwards, since an unauthorized request should never get far enough to have its
substance evaluated. It was caught, reordered, and a test was added specifically to keep that order
locked in place.

More recently, while extending the system to use more off-the-shelf tools, two separate,
independently-written attempts at the same small feature both ran into the identical bug: a piece
of tracing code silently gave up and reported "no route" whenever the request passed through a
particular kind of internal routing that hadn't existed in the system before. That two unrelated
attempts hit the exact same wall was itself useful evidence that the bug was real and not a fluke.
The lazy fix both attempts reached for — quietly falling back to "no route" instead of crashing —
would have technically stopped the errors while silently throwing away the very information the
feature existed to capture. The team rejected that fix and replaced it with one that asks the web
framework itself, after the fact, what request actually got handled — which turned out to always
have the right answer, just not at the moment the earlier code was asking for it.

And under a closer security review, a piece of code that counts how many safety incidents are
currently open was found to be reading a shared, unlocked piece of memory directly, which is safe
only if nothing else touches it at the same instant. Under real, simultaneous traffic, it crashed
about once every twenty attempts — reproduced deliberately, on purpose, before being trusted as a
real problem — and was fixed by taking a quick, safe snapshot before reading it instead of reading
the live data directly.

The common thread through all four: the team's habit of independently reproducing a claimed problem
before trusting it, and independently reproducing a claimed fix before accepting it, rather than
taking any report — including its own — at its word.

## What's actually running today, and what's still just a plan

Everything described above under "what the system is made of" is built, tested, and running today.
Concretely, that includes the tamper-evident logging, the safety-rule engine, the capacity
simulation, the model lifecycle tracking, the Kubernetes-manifest safety checker, the drift
detector that watches whether incoming data has quietly shifted from what the model was trained on,
the web service and its command-line tool, and the pieces that trace requests and export
statistics. All of it is covered by a little over two thousand automated checks, and a separate,
much simpler smoke test confirms the system starts up and does the basic things it claims to do.

Three further pieces of work exist only as a written plan right now, with no code behind them yet,
and it's worth being explicit about that so the plan is never mistaken for the product. One is
extending the database and the web service to store and stream more of the system's records —
safety-incident history, drift baselines — and to produce a real, industry-standard software bill
of materials, the kind of document that lists exactly which third-party components a piece of
software is built from, for security auditing. Another is teaching the system to read and validate
configuration files more safely, which sounds simple but has a real trap in it: a config-parsing
library can be perfectly safe to load a file with, and still be unsafe to process afterward, if the
file uses internal shortcuts that make a small file expand into a much larger structure once
something tries to walk through all of it — a version of the old "zip bomb" idea, but for
configuration files. The plan already accounts for that trap; the code implementing the safeguard
does not exist yet. The third piece would compare this system's own safety-rule engine against a
well-known, independent policy library, side by side, on the same test cases — not to replace the
custom engine, but to have an outside check on whether it behaves the way the team believes it
does.

## What to watch

A little under a quarter of the system's code now depends on outside libraries rather than being
written from scratch, up from about a fifth before the most recent round of work, and that share is
expected to keep climbing as the remaining planned work is completed — deliberately, and only in
the places judged to be solved problems, following the same reasoning laid out above.

One honest gap worth naming: a chunk of an older test suite, covering an earlier version of the
system, was lost during a workspace move and was never rebuilt; the newer suite doesn't cover
exactly the same ground. And several of the system's more interesting behaviors — what happens
under a simulated cluster failure, how a capacity shortfall plays out, how a data-drift alert is
triggered — are careful, honest simulations of behavior, not tests run against real infrastructure
under real load. They're built to be truthful models of the failure mode, and they're labeled as
simulations everywhere they appear, but a simulation is still a step removed from the real thing,
and that distinction is worth keeping in mind rather than letting the word "tested" quietly cover
both.
