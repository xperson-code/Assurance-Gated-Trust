"""Minimal belief transformations for the mental-action grammar of Sec. 3:

    alpha ::= +phi | vdash(phi,psi) | cap(phi,psi) | down(phi,psi) | dashv(phi,psi)

Beliefs B_i are represented as a set of propositional atoms per agent;
background knowledge K_i as a set of known (chi, psi) implication pairs.
This is intentionally the simplest model that can carry the preconditions
each mental operator requires -- it is not a general epistemic/doxastic
model, since the paper's contribution being demonstrated here is the
*control* semantics layered on top of these operators, not the base
epistemic logic itself (which is inherited unchanged from the referenced
L-DINF framework).
"""
from __future__ import annotations

from typing import Tuple

from .core import World


def _conj(a: str, b: str) -> str:
    return f"{a}&{b}"


def preconditions_hold(world: World, agent: str, mental_op: str, args: Tuple[str, ...]) -> bool:
    """Ordinary epistemic preconditions of alpha, checked *before* the
    assurance-aware decision is taken (assurance never substitutes for
    these, Sec. 4.2)."""
    beliefs = world.belief(agent)
    knowledge = world.know(agent)
    if mental_op == "admit":
        return True  # "+phi" has no epistemic precondition beyond source_OK (checked separately)
    if mental_op == "infer":  # vdash(chi, psi): needs B_i(chi) and B_i(chi -> psi)
        chi, psi = args
        return chi in beliefs and f"{chi}=>{psi}" in beliefs
    if mental_op == "infercon":  # cap(chi, psi): needs B_i(chi) and B_i(psi)
        chi, psi = args
        return chi in beliefs and psi in beliefs
    if mental_op == "down":  # downarrow(chi, psi): needs B_i(chi) and K_i(chi -> psi)
        chi, psi = args
        return chi in beliefs and (chi, psi) in knowledge
    if mental_op == "revise":  # dashv(chi, psi): needs B_i(chi) and K_i(chi -> ~psi)
        chi, psi = args
        return chi in beliefs and (chi, "~" + psi) in knowledge
    raise ValueError(f"unknown mental operator: {mental_op}")


def apply_belief_transformation(world: World, agent: str, mental_op: str, args: Tuple[str, ...]) -> None:
    """Applies alpha's ordinary belief transformation *in place* on ``world``.

    Caller contract: only invoke this once the assurance-aware decision is
    "allow" (autonomous execution) or once a pending review has been
    successfully validated (Sec. 4.2) -- never on "block", and never on
    "review" before validation.
    """
    beliefs = world.belief(agent)
    if mental_op == "admit":
        (phi,) = args
        beliefs.add(phi)
    elif mental_op == "infer":
        chi, psi = args
        beliefs.add(psi)
    elif mental_op == "infercon":
        chi, psi = args
        beliefs.add(_conj(chi, psi))
    elif mental_op == "down":
        chi, psi = args
        beliefs.add(psi)
    elif mental_op == "revise":
        chi, psi = args
        beliefs.discard(psi)
    else:
        raise ValueError(f"unknown mental operator: {mental_op}")
