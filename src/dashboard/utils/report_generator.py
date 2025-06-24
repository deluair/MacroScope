"""Utility to generate a simple Markdown report for the MacroScope dashboard.

The function `generate_markdown_report` is intentionally lightweight so it has
no additional dependencies.  It accepts the pre-loaded `data` dictionary that
is passed around the dashboard components and returns a Markdown string that can
be served through the Streamlit `download_button`.

If richer PDF / DOCX export is needed later we can build on top of this module
and plug in additional tools (e.g. `markdown2`, `pdfkit`, or `docx`), but for
now we keep things minimal.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


def _kpi_section(data: Dict[str, Any]) -> str:
    """Return a Markdown bullet list of headline KPIs."""
    lines: List[str] = ["### Key Metrics\n"]

    # Guard-clauses in case some keys are missing
    gdp = data.get("fred_GDP")  # Example structure key
    cpi = data.get("fred_CPIAUCSL")
    unemployment = data.get("bls_LNS14000000")  # Example BLS series
    sp500 = data.get("yfinance_^GSPC")

    if gdp is not None and not pd.isna(gdp):
        lines.append(f"* **Real GDP (latest):** {gdp:,.0f} billion USD")
    if cpi is not None and not pd.isna(cpi):
        lines.append(f"* **CPI (YoY %):** {cpi:.1f}%")
    if unemployment is not None and not pd.isna(unemployment):
        lines.append(f"* **Unemployment Rate:** {unemployment:.1f}%")
    if sp500 is not None and not pd.isna(sp500):
        lines.append(f"* **S&P 500:** {sp500:,.0f}")

    if len(lines) == 1:
        lines.append("*No KPI data available.*")

    lines.append("\n")
    return "\n".join(lines)


def _table_section(data: Dict[str, Any]) -> str:
    """Return a Markdown table with the first few rows of each dataset."""
    lines: List[str] = ["### Data Snapshots\n"]
    for name, df in data.items():
        if isinstance(df, pd.DataFrame):
            lines.append(f"#### {name}\n")
            # Show top 5 rows
            sample = df.head().to_markdown()
            lines.append(sample + "\n")
    return "\n".join(lines)


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Generate a Markdown report string.

    Parameters
    ----------
    data : dict
        The dictionary returned by `load_all_data` containing pandas DataFrames
        keyed by descriptive names.

    Returns
    -------
    str
        Markdown-formatted report ready to download.
    """
    report_lines: List[str] = ["# MacroScope Weekly Report\n"]
    report_lines.append(f"_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n")

    # KPI section
    report_lines.append(_kpi_section(data))

    # Table previews
    report_lines.append(_table_section(data))

    return "\n".join(report_lines)


def generate_html_report(data: Dict[str, Any]) -> str:
    """Generate an HTML version of the report using the markdown output as base.

    If the *markdown* package is installed it will be used for proper
    conversion. Otherwise we fall back to wrapping the markdown text in a
    simple `<pre>` block so the user still gets an HTML file.
    """
    md_text = generate_markdown_report(data)
    try:
        import markdown  # type: ignore

        html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])  # basic conversion
        # Add a minimal style for readability
        style = """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
                   Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; margin: 40px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; }
            th { background-color: #f2f2f2; }
        </style>
        """
        return f"<html><head>{style}</head><body>{html}</body></html>"
    except ImportError:
        # Fallback – plain preformatted markdown
        return f"<html><body><pre>{md_text}</pre></body></html>"
