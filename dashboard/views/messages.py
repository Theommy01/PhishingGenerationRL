"""Read the actual text — the check no aggregate can do for you."""

import pandas as pd
import streamlit as st

from dashboard import data

COMPARISONS = {
    "Browse": "Filter and read.",
    "Same subject across rounds": (
        "One subject, what each round wrote for it. Drift and degeneration on a "
        "fixed input, with nothing else varying."
    ),
    "Train vs held out": (
        "One round, the two splits side by side. Does the text differ in kind "
        "for subjects the policy never trained on?"
    ),
    "Evaded vs blocked": (
        "One round, sorted by score. The fastest way to see what the policy "
        "found that the detector does not catch."
    ),
    "Across decoding runs": (
        "One subject, the same round, different runs — the temperature "
        "comparison. Subjects are shared between runs, so this is a join."
    ),
}


def _message_card(row, show=("score", "round", "split")) -> None:
    """One message, with the flags that decide whether it counts."""
    bits = []
    if "score" in show and pd.notna(row.get("score")):
        verdict = "evaded" if row.get("evaded") else "blocked"
        bits.append(f"**{row['score']:.3f}** {verdict}")
    elif pd.isna(row.get("score")):
        bits.append("_not scored yet_")
    if "round" in show:
        bits.append(f"round {row['round']}")
    if "split" in show:
        bits.append(row.get("split", ""))
    if pd.notna(row.get("temperature")):
        bits.append(f"T={row['temperature']}")

    flags = []
    for field, label in (("url_ok", "URL"), ("attachment_ok", "attach")):
        if field in row and pd.notna(row[field]):
            flags.append(f"{'✓' if row[field] else '✗'} {label}")
    if flags:
        bits.append(" ".join(flags))

    st.markdown(" · ".join(str(b) for b in bits if b))
    st.text(row["body"])


def _filters(df: pd.DataFrame) -> pd.DataFrame:
    columns = st.columns(5)
    rounds = sorted(df["round"].unique())
    chosen = columns[0].multiselect("round", rounds, default=rounds)
    splits = sorted(df["split"].dropna().unique())
    split = columns[1].multiselect("split", splits, default=splits)
    verdict = columns[2].selectbox("verdict", ["all", "evaded", "blocked", "unscored"])
    categories = sorted(df["category"].dropna().unique()) if "category" in df else []
    category = columns[3].multiselect("category", categories, default=categories)
    generators = sorted(df["generator"].dropna().unique()) if "generator" in df else []
    generator = columns[4].multiselect("generator", generators, default=generators)

    search = st.text_input("search the body or subject", "")

    out = df[df["round"].isin(chosen) & df["split"].isin(split)]
    if categories:
        out = out[out["category"].isin(category)]
    if generators:
        out = out[out["generator"].isin(generator)]
    if verdict == "evaded":
        out = out[out.get("evaded") == True]  # noqa: E712
    elif verdict == "blocked":
        out = out[out.get("evaded") == False]  # noqa: E712
    elif verdict == "unscored":
        out = out[~out["scored"]]
    if search:
        mask = out["body"].str.contains(search, case=False, na=False)
        if "subject_text" in out:
            mask |= out["subject_text"].str.contains(search, case=False, na=False)
        out = out[mask]
    return out


def render(run_id: int) -> None:
    df = data.messages_frame(run_id)
    if df.empty:
        st.info("No messages yet.")
        return

    mode = st.radio("mode", list(COMPARISONS), horizontal=True, key="messages-mode")
    st.caption(COMPARISONS[mode])
    st.divider()

    if mode == "Browse":
        filtered = _filters(df)
        st.caption(f"{len(filtered)} of {len(df)} messages")
        for _, row in filtered.head(50).iterrows():
            with st.container(border=True):
                st.caption(row.get("subject_text", ""))
                _message_card(row)
        if len(filtered) > 50:
            st.caption("showing the first 50 — narrow the filters to see others")

    elif mode == "Same subject across rounds":
        subjects = sorted(df["subject_text"].dropna().unique())
        subject = st.selectbox("subject", subjects)
        rows = df[df["subject_text"] == subject]
        for round_index in sorted(rows["round"].unique()):
            st.markdown(f"**Round {round_index}**")
            columns = st.columns(min(4, len(rows[rows["round"] == round_index])))
            for column, (_, row) in zip(columns, rows[rows["round"] == round_index].iterrows()):
                with column, st.container(border=True):
                    _message_card(row, show=("score",))

    elif mode == "Train vs held out":
        round_index = st.selectbox("round", sorted(df["round"].unique()))
        rows = df[df["round"] == round_index]
        left, right = st.columns(2)
        for column, split in ((left, "train"), (right, "holdout")):
            with column:
                subset = rows[rows["split"] == split]
                evaded = subset["evaded"].mean() * 100 if "evaded" in subset else float("nan")
                st.markdown(f"**{split}** — {len(subset)} messages, {evaded:.0f}% evaded")
                for _, row in subset.head(8).iterrows():
                    with st.container(border=True):
                        st.caption(row.get("subject_text", ""))
                        _message_card(row, show=("score",))

    elif mode == "Evaded vs blocked":
        round_index = st.selectbox("round", sorted(df["round"].unique()))
        rows = data.scored(df[df["round"] == round_index]).sort_values(
            "score", ascending=False
        )
        if rows.empty:
            st.info("this round has not been scored yet")
            return
        left, right = st.columns(2)
        with left:
            st.markdown("**Evaded** (highest scoring first)")
            for _, row in rows.head(8).iterrows():
                with st.container(border=True):
                    st.caption(row.get("subject_text", ""))
                    _message_card(row, show=("score", "split"))
        with right:
            st.markdown("**Blocked** (lowest scoring first)")
            for _, row in rows.tail(8).iloc[::-1].iterrows():
                with st.container(border=True):
                    st.caption(row.get("subject_text", ""))
                    _message_card(row, show=("score", "split"))

    elif mode == "Across decoding runs":
        runs = data.list_runs()
        if len(runs) < 2:
            st.info(
                "Only one run so far. This view compares the same subject across "
                "runs — it becomes useful after the second decoding run."
            )
            return
        chosen = st.multiselect(
            "runs", runs["run_id"], default=list(runs["run_id"][:2]), format_func=str
        )
        subjects = sorted(df["subject_text"].dropna().unique())
        subject = st.selectbox("subject", subjects, key="cross-run-subject")
        rows = data.messages_for_subject_text(subject, chosen)
        if rows.empty:
            st.info("no messages for that subject in the chosen runs")
            return
        columns = st.columns(len(chosen))
        for column, run in zip(columns, chosen):
            with column:
                config = data.run_config(run)
                temperature = (config.get("gen_args") or {}).get("temperature")
                st.markdown(f"**run {run}** — T={temperature}")
                for _, row in rows[rows["run_id"] == run].head(6).iterrows():
                    with st.container(border=True):
                        _message_card(row, show=("score", "round"))
