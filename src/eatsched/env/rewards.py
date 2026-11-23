"""Reward formulation for paper_mode and extended_mode in EAT-Sched."""

from __future__ import annotations

import math


def compute_paper_reward(
    quality: float,
    response_time: float,
    avg_queue_wait: float,
    w_q: float = 0.5,
    w_t1: float = 0.3,
    w_t2: float = 0.2,
    s_q: float = 0.5,
    s_t1: float = 40.0,
    s_t2: float = 30.0,
) -> float:
    """Compute immediate reward R_t for scheduled task according to Eq. 9 [PAPER].

    R_t = b + w_q * tanh(q / s_q) - w_t1 * tanh(t^r / s_t1) - w_t2 * tanh(t_Q^{avg} / s_t2)
    where b = w_t1 + w_t2.
    """
    bias = w_t1 + w_t2
    term_q = w_q * math.tanh(quality / s_q)
    term_t1 = w_t1 * math.tanh(response_time / s_t1)
    term_t2 = w_t2 * math.tanh(avg_queue_wait / s_t2)
    return bias + term_q - term_t1 - term_t2


def compute_idle_reward(
    avg_queue_wait: float,
    w_t2: float = 0.2,
    s_t2: float = 30.0,
) -> float:
    """Compute queue backlog penalty for idle or safe no-op decision steps [ASM-07].

    R_idle = - w_t2 * tanh(t_Q^{avg} / s_t2)
    """
    return -w_t2 * math.tanh(avg_queue_wait / s_t2)


def compute_extended_reward(
    paper_reward: float,
    response_time: float,
    relative_deadline: float,
    priority: int,
    energy_joules: float,
    quality: float,
    quality_threshold: float,
    w_dl: float = 0.5,
    w_e: float = 0.1,
    s_e: float = 5000.0,
    bonus_val: float = 0.1,
) -> float:
    """Compute QoS and energy augmented reward for extended_mode [ABSTRACT Sec. E5].

    R_ext = R_paper - w_dl * p_k * 1[t^r > d^rel] - w_e * tanh(E / s_e) + R_bonus
    """
    # Deadline miss indicator
    deadline_miss = 1.0 if response_time > relative_deadline else 0.0
    deadline_penalty = w_dl * float(priority) * deadline_miss

    # Energy consumption penalty
    energy_penalty = w_e * math.tanh(energy_joules / s_e)

    # Quality compliance bonus / penalty
    bonus = bonus_val if quality >= quality_threshold else -bonus_val

    return paper_reward - deadline_penalty - energy_penalty + bonus
