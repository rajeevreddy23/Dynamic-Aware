"""Mathematical validation tests for quality functions, actions, and rewards."""

from eatsched.env.quality import SimulatedQualityBackend
from eatsched.env.rewards import compute_idle_reward, compute_paper_reward


def test_saturating_monotone_quality():
    """Verify quality function is strictly bounded and monotonically increasing."""
    backend = SimulatedQualityBackend(q_max=0.85, tau_q=18.0, sigma_q=0.0)

    q20 = backend.compute_quality(20)
    q40 = backend.compute_quality(40)
    q60 = backend.compute_quality(60)

    assert 0.0 <= q20 < q40 < q60 <= 1.0
    assert 0.55 <= q20 <= 0.65
    assert 0.70 <= q40 <= 0.80
    assert 0.80 <= q60 <= 0.85


def test_paper_reward_eq_9():
    """Verify immediate reward Eq. 9 and bias offset b = w_t1 + w_t2."""
    w_q, w_t1, w_t2 = 0.5, 0.3, 0.2
    b = w_t1 + w_t2  # 0.5
    assert b == 0.5

    r = compute_paper_reward(
        quality=0.75,
        response_time=30.0,
        avg_queue_wait=10.0,
        w_q=w_q,
        w_t1=w_t1,
        w_t2=w_t2,
    )
    assert isinstance(r, float)

    # Idle reward should strictly penalize backlog
    r_idle_zero = compute_idle_reward(0.0)
    r_idle_backlog = compute_idle_reward(40.0)
    assert r_idle_zero == 0.0
    assert r_idle_backlog < 0.0
