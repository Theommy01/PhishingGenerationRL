"""The adversarial fine-tuning loop.

    store.py   MongoDB persistence for runs, rounds and generated messages

One round is: generate n messages per prompt with the current checkpoint, score
them with ScamLLM, add them to the run's message pool, train on the *cumulative*
pool, and evaluate the next checkpoint on the same prompts.

Deliberately self-contained rather than built on `phishnet_classes`, so the loop
can move while those packages are being overhauled. The schema mirrors their
shape (experiment / checkpoint / output) to keep the later migration cheap.
"""
