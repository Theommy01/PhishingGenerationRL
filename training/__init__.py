"""One training round, and what it is measured against.

    kto_trainer / bco_trainer   the two algorithms; a run uses one or the other
    reference_model             what the KL term is anchored to, shared by both
    policy_kl                   how far a trained policy ended up from that
                                anchor, measured after the fact on generated text

Run a trainer standalone with `python -m training.kto_trainer`.
"""
