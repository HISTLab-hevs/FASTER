Experiment Setup Guide
======================

This guide explains how Faster presents round-level training schedules and
round-level client allocation in the experiment setup flow.

Simple vs Advanced
------------------

Faster exposes two setup modes:

- ``Simple`` keeps the workflow lightweight. Faster automatically spreads the
  federated training split across aggregation rounds and automatically reuses
  the prepared client distribution inside each round.
- ``Advanced`` unlocks manual control when you want to shape round coverage,
  edit per-round client allocation, reproduce prior runs more closely, inspect
  churn behavior in more detail, or work with the extra tuning controls that
  stay hidden in Simple mode, including seed and initial eligible clients.

Round-level training schedule
-----------------------------

The round-level training schedule answers one question:

How much of the federated training split should each aggregation round cover?

Important behavior:

- Only the federated training split changes across rounds.
- Validation and test stay fixed every round.
- The full training plan must add up to ``100`` across all aggregation rounds.
- These percentages describe the plan for the run, not a promise of exact
  realized sample counts.

In ``Automatic`` mode, Faster builds that schedule for you.

In ``Manual`` mode, the setup UI exposes a structured round editor instead of a
comma-separated text field. Each round gets its own percentage input, along
with a running total and helper actions such as an even split preset. This
makes larger runs easier to review and edit because users can scan the plan at
a glance and immediately see whether the full schedule totals ``100``.

Round-level client allocation
-----------------------------

Round-level client allocation answers a different question:

Once a round has been assigned its training subset, how should that round be
distributed across clients?

Important behavior:

- Client allocation is the second planning layer after round coverage.
- Each round defines one percentage per client.
- Every round must total ``100``.

In ``Automatic`` mode, Faster reuses the prepared client distribution for every
round.

In ``Manual`` mode, Faster exposes a round-by-round allocation editor. Users
can jump between rounds, inspect per-round totals, split the active round
evenly, or copy the current round's allocation to every round. This keeps the
workflow manageable even when a run has many aggregation rounds.

What churn changes
------------------

The configured client allocation is defined against the full planned client
pool.

If churn makes some clients inactive in a round, Faster automatically
redistributes that round's planned client share across the clients that are
still eligible. In practice, that means:

- the round still uses its full scheduled training subset
- the allocation adapts to the clients that are actually available
- users should expect the effective per-client share to change when churn
  changes round eligibility

This behavior is intentional. It preserves full use of the round's planned data
coverage instead of leaving part of the round unassigned when some clients are
inactive.
