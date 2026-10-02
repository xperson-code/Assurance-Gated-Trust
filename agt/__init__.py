"""Assurance-Gated Trust (AGT) — a demonstration reference monitor.

This package implements, as runnable code, the core semantic objects of an
assurance-gated extension of a trust-regulated dynamic epistemic framework
for heterogeneous multi-agent systems: the trust-only baseline, the
LLM-specific assurance layer (verification, certificates, review/commit
lifecycle, conflict resolution, trusted justification traces) and the
control-safety invariants Phi_ctl.

It is a *demonstration* vehicle for a vision paper, not a production
verification system: beliefs are represented by minimal propositional atoms
(enough to drive the control semantics that is the actual contribution),
and verifiers are pluggable mock/random functions rather than real model
checkers or LLM critics. See the top-level README.md for a section-by-section
map from paper equations to the functions below.
"""
