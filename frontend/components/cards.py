import streamlit as st


def render_metric_card(
    title: str, value: str, delta: str = None, delta_color: str = "normal"
) -> None:
    """Renders a standardized KPI metric card."""
    st.metric(label=title, value=value, delta=delta, delta_color=delta_color)


def render_risk_banner(probability: float, risk_segment: str) -> None:
    """Renders visual risk alert cards based on prediction band."""
    if risk_segment == "Low":
        st.success(
            f"**Risk Tier: LOW ({probability * 100:.1f}%)**\n\n"
            "Customer shows strong retention signals. No urgent intervention required."
        )
    elif risk_segment == "Medium":
        st.warning(
            f"**Risk Tier: MEDIUM ({probability * 100:.1f}%)**\n\n"
            "Customer engagement is declining. Recommend targeted feature education or contract incentives."
        )
    else:
        st.error(
            f"**Risk Tier: HIGH ({probability * 100:.1f}%)**\n\n"
            "Critical flight risk! Recommend priority outreach by relationship managers or specialized loyalty discounts."
        )


def render_data_completeness_badge(
    total_records: int, missing_records: int = 0
) -> None:
    """Renders an audit card summarizing batch file processing health."""
    completeness = (
        ((total_records - missing_records) / max(1, total_records)) * 100
    )
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Records Processed", value=f"{total_records:,}")
    with col2:
        st.metric(label="Data Ingestion Quality", value=f"{completeness:.1f}%")