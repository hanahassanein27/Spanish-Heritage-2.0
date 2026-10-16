"""
Hispanic Heritage Historical Data Explorer
A Streamlit classroom app for World Bank historical data and function analysis.
"""
from __future__ import annotations

import io
import math
import re
import time
from datetime import date
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📈", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current mortality patterns persisted.", lambda x: x > 0),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, national estimate.", lambda x: 0 <= x <= 100),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population. It is not average wealth or household income.", lambda x: x >= 0),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count.", lambda x: x >= 0),
    "Net migration": ("SM.POP.NETM", "people (net)", "Net number of migrants during the period; it is not an emigration count.", lambda x: True),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people.", lambda x: x >= 0),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index.", lambda x: x > -100),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP.", lambda x: x >= 0),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, valued in current US dollars.", lambda x: x >= 0),
    "External balance on goods/services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP.", lambda x: -100 <= x <= 100),
}
CONTEXTS = {
    "Business": ("Revenue", "dollars", "Time", "years", "100 + 8*x - 0.3*x**2", "Revenue changes over time; derivatives describe the rate and acceleration of revenue."),
    "Engineering": ("Position", "meters", "Time", "seconds", "2*x**3 - 9*x**2 + 12*x", "The first derivative is velocity (m/s); the second derivative is acceleration (m/s²)."),
    "Medical": ("Drug concentration", "mg/L", "Time", "hours", "80*exp(-0.35*x)", "The first derivative is the rate concentration changes; the second describes how that rate changes."),
    "Sports science": ("Jump height", "meters", "Time", "seconds", "4*x - 0.8*x**2", "The derivative is vertical velocity; the second derivative is vertical acceleration."),
    "Biology": ("Population", "organisms", "Time", "days", "20*exp(0.12*x)", "The derivative is population change per day; the second derivative describes changing growth."),
    "Bioengineering": ("Sensor response", "units", "Input", "units", "0.5*x**3 - 2*x + 4", "Derivatives describe sensitivity and how sensitivity changes."),
    "Economic forecasting": ("Price index", "index points", "Time", "years", "100*exp(0.03*x)", "The first derivative is the index's rate of change; the second shows whether that rate is increasing."),
    "Weather": ("Temperature", "°C", "Time", "hours", "18 + 6*sin(2*pi*x/24)", "The derivative is temperature change per hour; the second derivative describes how that rate changes."),
    "Custom": ("Dependent variable", "units", "Independent variable", "units", "x**2", "Interpret derivatives using the variables and units you provide."),
}
WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=200"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country: str, indicator: str, start: int, end: int) -> tuple[pd.DataFrame, str]:
    """Fetch a World Bank indicator. Errors are returned as readable messages, never fake data."""
    url = WB_URL.format(country=country, indicator=indicator, start=start, end=end)
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Hispanic-Heritage-Data-Explorer/1.0"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            message = payload[0].get("message", "The API returned no observations.") if isinstance(payload, list) and payload and isinstance(payload[0], dict) else "Unexpected or empty World Bank response."
            return pd.DataFrame(columns=["year", "value"]), message
        rows = [{"year": int(item["date"]), "value": item.get("value")} for item in payload[1] if item.get("date")]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["year", "value"]), "The API returned no observations for this period."
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
        return df, f"Retrieved from the World Bank API: {url}"
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not retrieve data ({type(exc).__name__}: {exc}). Check your connection or try again."

def fit_polynomial(df: pd.DataFrame, degree: int):
    clean = df[["year", "value"]].copy()
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna().drop_duplicates("year").sort_values("year")
    if len(clean) < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid years; only {len(clean)} are available.")
    base_year = int(clean["year"].min())
    t = clean["year"].to_numpy(dtype=float) - base_year
    y = clean["value"].to_numpy(dtype=float)
    coeff = np.polyfit(t, y, degree)
    pred = np.polyval(coeff, t)
    return clean, base_year, coeff, pred

def poly_value(year: float, base_year: int, coeff: np.ndarray) -> float:
    return float(np.polyval(coeff, float(year) - base_year))

def poly_equation(coeff: np.ndarray, base_year: int, unit: str) -> str:
    pieces = []
    degree = len(coeff) - 1
    for i, c in enumerate(coeff):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        term = f"{c:.8g}" if power == 0 else (f"{c:.8g}(t^{power})" if power > 1 else f"{c:.8g}t")
        pieces.append(term)
    return "y(t) = " + (" + ".join(pieces).replace("+ -", "- ") if pieces else "0") + f", where t = year − {base_year}; y in {unit}"

def safe_expression(text: str, x: sp.Symbol) -> sp.Expr:
    """Parse a student-entered expression with a limited set of mathematical names."""
    cleaned = text.strip().replace("^", "**")
    if not cleaned or len(cleaned) > 180:
        raise ValueError("Enter a non-empty expression shorter than 180 characters.")
    allowed = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
               "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E}
    try:
        expr = sp.sympify(cleaned, locals=allowed, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not parse that expression. Try forms such as x**2, 1/(x-2), exp(-x), log(x), or sin(x).") from exc
    if expr.free_symbols - {x}:
        raise ValueError("Use x as the only variable.")
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("The expression contains an undefined or infinite constant.")
    return expr

def numeric_function(expr: sp.Expr, x: sp.Symbol):
    f = sp.lambdify(x, expr, modules=["numpy"])
    def evaluate(values):
        arr = np.asarray(values, dtype=float)
        with np.errstate(all="ignore"):
            result = np.asarray(f(arr), dtype=float)
        return np.where(np.isfinite(result), result, np.nan)
    return evaluate

def fmt_num(v: Any, digits: int = 6) -> str:
    try:
        v = float(v)
        return f"{v:.{digits}g}" if np.isfinite(v) else "undefined"
    except Exception:
        return str(v)

def make_pdf(title: str, metadata: list[str], equation: str, metrics: list[list[str]], chart_png: bytes | None,
             preview: pd.DataFrame, results: pd.DataFrame, findings: list[str]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=32, leftMargin=32, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"])]
    for line in metadata:
        story.append(Paragraph(line.replace("&", "&amp;"), styles["BodyText"]))
    story += [Spacer(1, 8), Paragraph("Model equation", styles["Heading2"]), Paragraph(equation.replace("&", "&amp;"), styles["BodyText"]),
              Spacer(1, 6), Paragraph("Metrics", styles["Heading2"])]
    table = Table(metrics, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("GRID", (0,0), (-1,-1), .4, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 8), ("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(table)
    if chart_png:
        story += [Spacer(1, 8), RLImage(io.BytesIO(chart_png), width=650, height=300)]
    story.append(Paragraph("Findings and limitations", styles["Heading2"]))
    for finding in findings:
        story.append(Paragraph("• " + finding.replace("&", "&amp;"), styles["BodyText"]))
    for name, frame in [("Data preview", preview), ("Calculated years", results)]:
        story += [Spacer(1, 8), Paragraph(name, styles["Heading2"])]
        if frame is not None and not frame.empty:
            shown = frame.head(25).fillna("").astype(str)
            data = [list(shown.columns)] + shown.values.tolist()
            tab = Table(data, repeatRows=1)
            tab.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("GRID", (0,0), (-1,-1), .3, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 7)]))
            story.append(tab)
        else:
            story.append(Paragraph("No calculated results yet.", styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()

st.title("📈 Hispanic Heritage Historical Data Explorer")
st.caption("Hispanic Heritage Data Lab • World Bank historical indicators + interactive function analysis")
st.info("Brazil is included as a Latin American comparison country. World Bank coverage varies by indicator and country; missing years are not fabricated.")

with st.sidebar:
    st.header("Historical data")
    country_label = st.selectbox("Featured country", list(COUNTRIES))
    indicator_label = st.selectbox("Indicator", list(INDICATORS))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    compare_labels = st.multiselect("Comparison countries", [x for x in COUNTRIES if x != country_label], default=[])
    printer = st.checkbox("Printer-friendly report view", value=False)

current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
indicator_code, indicator_unit, indicator_definition, constraint = INDICATORS[indicator_label]
country_code = COUNTRIES[country_label]
source_url = WB_URL.format(country=country_code, indicator=indicator_code, start=start_year, end=end_year)
with st.spinner("Retrieving World Bank observations…"):
    source_df, source_message = fetch_world_bank(country_code, indicator_code, start_year, end_year)
if "source_df" not in st.session_state or st.session_state.get("source_key") != (country_code, indicator_code, start_year, end_year):
    st.session_state.source_df = source_df.copy()
    st.session_state.source_key = (country_code, indicator_code, start_year, end_year)
    st.session_state.edit_df = source_df.copy()
    st.session_state.edited = False
    st.session_state.calculated_years = []
    st.session_state.batch_results = pd.DataFrame()
source_df = st.session_state.source_df.copy()
if not source_message.startswith("Retrieved"):
    st.warning(source_message)
st.markdown(f"**Definition:** {indicator_definition}  \n**Unit:** {indicator_unit}  \n**Indicator code:** `{indicator_code}`  \n**Requested period:** {start_year}–{end_year} (70 calendar years)  \n**Source:** [World Bank API]({source_url})  \n**Retrieved on:** {date.today().isoformat()}")
st.caption(f"Actual source coverage: {int(source_df.year.min()) if not source_df.empty else 'none'}–{int(source_df.year.max()) if not source_df.empty else 'none'}; {len(source_df)} non-null annual observations. Missing years can occur.")
if source_df.empty:
    st.error("No usable source observations are available. Check the API message above. The app will not substitute random values.")
    st.stop()

st.subheader("1. Review and edit the working data")
st.write("The source table is kept separately from your editable working table. Edits create a user-modified scenario and do not change the World Bank source.")
col1, col2 = st.columns([3, 1])
with col1:
    edited_df = st.data_editor(st.session_state.edit_df, num_rows="dynamic", use_container_width=True, key="data_editor")
with col2:
    st.metric("Source observations", len(source_df))
    st.metric("Working rows", len(edited_df))
    st.write("Status:", "User-modified scenario" if st.session_state.edited else "Source data")
    if st.button("Reset to source", use_container_width=True):
        st.session_state.edit_df = source_df.copy()
        st.session_state.edited = False
        st.session_state.calculated_years = []
        st.session_state.batch_results = pd.DataFrame()
        st.rerun()
if not edited_df.equals(source_df):
    st.session_state.edited = True
st.session_state.edit_df = edited_df.copy()

try:
    numeric_years = pd.to_numeric(edited_df["year"], errors="coerce")
    numeric_values = pd.to_numeric(edited_df["value"], errors="coerce")
    if numeric_years.isna().any() or not np.all(numeric_years == np.floor(numeric_years)):
        raise ValueError("Years must be whole-number integers.")
    if numeric_years.duplicated().any():
        raise ValueError("Duplicate years are not allowed. Keep one row per year.")
    if numeric_values.isna().any():
        raise ValueError("Every working row must have a numeric value.")
    invalid = [(int(y), float(v)) for y, v in zip(numeric_years, numeric_values) if not constraint(float(v))]
    if invalid:
        raise ValueError(f"Some values violate the indicator's basic range constraints: {invalid[:5]}")
    validated = pd.DataFrame({"year": numeric_years.astype(int), "value": numeric_values.astype(float)}).sort_values("year")
    fitted, base_year, coeff, fitted_values = fit_polynomial(validated, degree)
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"Data/model validation: {exc}")
    st.stop()

eq = poly_equation(coeff, base_year, indicator_unit)
r2 = r2_score(fitted["value"], fitted_values) if len(fitted) > 1 else float("nan")
mae = mean_absolute_error(fitted["value"], fitted_values)
residuals = fitted["value"].to_numpy() - fitted_values
metrics = [["Metric", "Value"], ["Observation count", str(len(fitted))], ["Historical range", f"{int(fitted.year.min())}–{int(fitted.year.max())}"],
           ["Base year", str(base_year)], ["Degree", str(degree)], ["In-sample R²", fmt_num(r2)], ["In-sample MAE", f"{fmt_num(mae)} {indicator_unit}"],
           ["Observed mean", f"{fmt_num(fitted.value.mean())} {indicator_unit}"]]
holdout_text = "Holdout comparison unavailable: not enough observations for a chronological holdout plus degree+1 training points."
if len(fitted) >= degree + 5:
    test_n = max(2, int(round(len(fitted) * 0.2)))
    train, test = fitted.iloc[:-test_n], fitted.iloc[-test_n:]
    if len(train) >= degree + 1:
        hold_base = int(train.year.min())
        hold_coeff = np.polyfit(train.year.to_numpy(float) - hold_base, train.value.to_numpy(float), degree)
        hold_pred = np.polyval(hold_coeff, test.year.to_numpy(float) - hold_base)
        holdout_text = f"Chronological holdout MAE (last {len(test)} observations; trained on earlier years): {fmt_num(mean_absolute_error(test.value, hold_pred))} {indicator_unit}. This is separate from in-sample metrics."

st.subheader("2. Fit and evaluate the historical model")
st.code(eq, language=None)
m1, m2, m3, m4 = st.columns(4)
m1.metric("In-sample R²", fmt_num(r2))
m2.metric("In-sample MAE", f"{fmt_num(mae)} {indicator_unit}")
m3.metric("Observed mean", f"{fmt_num(fitted.value.mean())} {indicator_unit}")
m4.metric("Observations", len(fitted))
st.caption(f"Centered time uses t = year − {base_year}. R² and MAE are in-sample fit metrics, not proof of causation or future accuracy. {holdout_text}")
with st.expander("Inspect residuals"):
    residual_table = fitted[["year", "value"]].copy()
    residual_table["fitted_value"] = fitted_values
    residual_table["residual (actual − fitted)"] = residuals
    st.dataframe(residual_table, use_container_width=True)

st.subheader("3. Calculate one year or a batch of years")
c1, c2 = st.columns(2)
with c1:
    one_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=min(max(int(fitted.year.max()), 1950), 2100), step=1)
    if st.button("Calculate Value"):
        st.session_state.calculated_years = [int(one_year)]
        st.session_state.batch_results = pd.DataFrame()
with c2:
    batch_mode = st.radio("Batch input", ["Year range", "Comma-separated custom years"], horizontal=True)
    if batch_mode == "Year range":
        batch_start = st.number_input("Start year", 1950, 2100, int(max(1950, fitted.year.min())), key="batch_start")
        batch_end = st.number_input("End year", 1950, 2100, int(min(2100, fitted.year.max() + 5)), key="batch_end")
        batch_step = st.slider("Batch step (years)", 1, 10, 1)
        batch_text = ""
    else:
        batch_text = st.text_input("Custom years", value="2000, 2010, 2020")
        batch_start, batch_end, batch_step = 0, 0, 1
    if st.button("Calculate batch"):
        try:
            if batch_mode == "Year range":
                if batch_end < batch_start:
                    raise ValueError("End year must be at least the start year.")
                years = list(range(int(batch_start), int(batch_end) + 1, int(batch_step)))
            else:
                pieces = [p.strip() for p in batch_text.split(",") if p.strip()]
                if not pieces:
                    raise ValueError("Enter one or more comma-separated years.")
                if any(not re.fullmatch(r"\d{4}", p) for p in pieces):
                    raise ValueError("Each custom year must be a four-digit integer.")
                years = [int(p) for p in pieces]
            if any(y < 1950 or y > 2100 for y in years):
                raise ValueError("Requested years must be between 1950 and 2100.")
            st.session_state.calculated_years = sorted(set(years))
            st.session_state.batch_results = pd.DataFrame()
        except ValueError as exc:
            st.error(f"Batch input error: {exc}")
if st.session_state.get("calculated_years"):
    results = pd.DataFrame({"year": st.session_state.calculated_years})
    results["estimate"] = [poly_value(y, base_year, coeff) for y in results.year]
    actual_lookup = dict(zip(fitted.year, fitted.value))
    results["actual observation"] = results.year.map(actual_lookup)
    results["classification"] = np.where(results.year.between(fitted.year.min(), fitted.year.max()), "within fitted historical range", "outside fitted historical range")
    results["status"] = np.where(results["actual observation"].notna(), "Observed year: compare actual and fitted estimate", "Model estimate only; no actual observation in working data")
    st.session_state.batch_results = results
    st.dataframe(results, use_container_width=True)
    for _, row in results.iterrows():
        if row["classification"] == "outside fitted historical range":
            st.warning(f"{int(row.year)} is outside the fitted range. Polynomial extrapolation may be implausible; estimates are not clipped.")
    if len(results) == 1:
        row = results.iloc[0]
        st.success(f"Estimate for {int(row.year)}: {fmt_num(row.estimate)} {indicator_unit}. Actual fitted range: {int(fitted.year.min())}–{int(fitted.year.max())}. {row.classification}.")
    st.session_state.batch_results = results

st.subheader("4. Visualize the historical model")
plot_end = min(2150, int(fitted.year.max()) + int(horizon))
xx = np.linspace(float(fitted.year.min()), float(max(fitted.year.max(), plot_end)), max(250, (plot_end - int(fitted.year.min()) + 1) * 3))
yy = np.polyval(coeff, xx - base_year)
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.scatter(fitted.year, fitted.value, label="Working data (observed/edited)", color="tab:blue", s=26, zorder=3)
inside = xx <= fitted.year.max()
ax.plot(xx[inside], yy[inside], label="Fitted curve (historical range)", color="tab:orange", linewidth=2)
if np.any(~inside):
    ax.plot(xx[~inside], yy[~inside], label="Extrapolated segment", color="tab:red", linestyle="--", linewidth=2)
ax.axvline(float(fitted.year.max()), color="grey", linestyle=":", label="Historical boundary")
sample_years = np.arange(int(fitted.year.min()), int(fitted.year.max()) + 1, sample_step)
ax.scatter(sample_years, [poly_value(y, base_year, coeff) for y in sample_years], marker="x", label=f"Model markers every {sample_step} year(s)", color="black")
if st.session_state.get("calculated_years"):
    ys = st.session_state.calculated_years
    ax.scatter(ys, [poly_value(y, base_year, coeff) for y in ys], marker="D", s=55, label="Calculated estimates", color="tab:green", zorder=4)
ax.set_xlabel("Year")
ax.set_ylabel(f"{indicator_label} ({indicator_unit})")
ax.set_title(f"{country_label}: {indicator_label}")
ax.grid(True, alpha=.25)
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
chart_buf = io.BytesIO()
fig.savefig(chart_buf, format="png", dpi=150, bbox_inches="tight")
chart_png = chart_buf.getvalue()
plt.close(fig)

if compare_labels:
    st.markdown("#### Same-indicator country comparison")
    comp_fig, comp_ax = plt.subplots(figsize=(10, 4.2))
    comp_ax.plot(fitted.year, fitted.value, marker="o", markersize=3, label=country_label)
    comp_coverage = []
    for label in compare_labels:
        df2, msg2 = fetch_world_bank(COUNTRIES[label], indicator_code, start_year, end_year)
        if not df2.empty:
            comp_ax.plot(df2.year, df2.value, marker=".", markersize=3, label=label)
            comp_coverage.append(f"{label}: {int(df2.year.min())}–{int(df2.year.max())} ({len(df2)} observations)")
        else:
            st.warning(f"{label}: {msg2}")
    comp_ax.set_xlabel("Year"); comp_ax.set_ylabel(f"{indicator_label} ({indicator_unit})")
    comp_ax.set_title("Same indicator, different country coverage"); comp_ax.grid(True, alpha=.25); comp_ax.legend()
    comp_fig.tight_layout(); st.pyplot(comp_fig, use_container_width=True)
    st.caption("Coverage: " + "; ".join(comp_coverage) + ". Gaps can reflect missing or unreported indicator observations; lines connect available observations only.")
    plt.close(comp_fig)

st.subheader("5. Function lab: change context, function, and course level")
course = st.radio("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"], horizontal=True)
context = st.selectbox("Context", list(CONTEXTS))
dep_default, dep_unit_default, indep_default, indep_unit_default, suggested, context_note = CONTEXTS[context]
function_mode = st.radio("Function source", ["Historical curve of best fit", "Suggested function", "Build from parameters", "Enter my own function"], horizontal=True)
if function_mode == "Historical curve of best fit":
    # Express the centered polynomial as a symbolic function of x (calendar year).
    x = sp.Symbol("x", real=True)
    expression = sum(sp.Float(float(c), 12) * (x - base_year) ** (len(coeff) - i - 1) for i, c in enumerate(coeff))
    function_label = "Centered polynomial fitted to the selected historical data"
elif function_mode == "Suggested function":
    expression_text = suggested
    function_label = f"Suggested {context.lower()} function"
elif function_mode == "Build from parameters":
    builder_kind = st.selectbox("Model family", ["Polynomial (quadratic)", "Rational", "Exponential", "Logarithmic", "Trigonometric"])
    a = st.number_input("Parameter a", value=1.0, key="param_a")
    b = st.number_input("Parameter b", value=1.0, key="param_b")
    c = st.number_input("Parameter c", value=0.0, key="param_c")
    if builder_kind == "Polynomial (quadratic)":
        expression_text = f"({a})*x**2 + ({b})*x + ({c})"
    elif builder_kind == "Rational":
        expression_text = f"({a})/(x-({b})) + ({c})"
    elif builder_kind == "Exponential":
        expression_text = f"({a})*exp(({b})*x) + ({c})"
    elif builder_kind == "Logarithmic":
        expression_text = f"({a})*log(x-({b})) + ({c})"
    else:
        expression_text = f"({a})*sin(({b})*x) + ({c})"
    function_label = f"Parameter-built {builder_kind.lower()} function"
else:
    expression_text = st.text_input("Function in x", value="x**2 - 4*x + 3", help="Use * for multiplication, ** or ^ for powers, exp(x), log(x), sin(x), cos(x), tan(x), sqrt(x), and pi.")
    function_label = "Student-entered function"

d1, d2, d3 = st.columns(3)
with d1:
    dep_name = st.text_input("Dependent variable", dep_default)
    dep_unit = st.text_input("Dependent units", dep_unit_default)
with d2:
    indep_name = st.text_input("Independent variable", indep_default)
    indep_unit = st.text_input("Independent units", indep_unit_default)
with d3:
    context_description = st.text_input("Context description", context_note)
try:
    x = sp.Symbol("x", real=True)
    if function_mode == "Historical curve of best fit":
        expr = expression
    else:
        expr = safe_expression(expression_text, x)
    derivative = sp.diff(expr, x)
    second_derivative = sp.diff(derivative, x)
    f = numeric_function(expr, x); df = numeric_function(derivative, x); ddf = numeric_function(second_derivative, x)
except Exception as exc:
    st.error(f"Function input error: {exc}")
    st.stop()

st.markdown(f"**Selected model:** {function_label}")
st.latex(sp.latex(sp.Eq(sp.Symbol(dep_name if dep_name.isidentifier() else 'y'), expr)))
st.caption(f"Independent variable: {indep_name} ({indep_unit}); dependent variable: {dep_name} ({dep_unit}). {context_description}")

if course == "Honors Algebra 2":
    st.markdown("**Algebra 2 focus:** domain and range on the displayed interval, end behavior, graphical trends, and average rates of change.")
elif course == "Precalculus":
    st.markdown("**Precalculus focus:** degree and coefficient effects, turning points, concavity, and how average rates change.")
else:
    st.markdown("**Calculus focus:** first/second derivatives, tangent and secant lines, critical points, extrema, concavity, and inflection points.")

default_lo, default_hi = (0.0, 10.0)
interval = st.slider("Analysis interval", min_value=-20.0, max_value=30.0, value=(default_lo, default_hi), step=0.5)
lo, hi = interval
if hi <= lo:
    st.error("Analysis interval must have its right endpoint greater than its left endpoint.")
    st.stop()
point = st.slider("Current input value x", min_value=float(lo), max_value=float(hi), value=float((lo+hi)/2), step=max((hi-lo)/100, .01))
sec_cols = st.columns(2)
with sec_cols[0]:
    a_sec = st.slider("Secant interval start", min_value=float(lo), max_value=float(hi), value=float(lo), step=max((hi-lo)/100, .01))
with sec_cols[1]:
    b_sec = st.slider("Secant interval end", min_value=float(lo), max_value=float(hi), value=float(hi), step=max((hi-lo)/100, .01))
show_f = st.checkbox("Show function", True)
show_df = st.checkbox("Show first derivative", True)
show_ddf = st.checkbox("Show second derivative", course == "Calculus")
speed = st.slider("Animation speed (frames per second)", 1, 12, 5)
play_cols = st.columns(3)
play = play_cols[0].button("▶ Play animation")
reset = play_cols[1].button("↺ Reset point")
pause = play_cols[2].button("Ⅱ Pause / hold")
if reset:
    point = float((lo + hi) / 2)
if play:
    placeholder = st.empty()
    frames = np.linspace(lo, hi, 24)
    for frame_x in frames:
        frame_fig, frame_ax = plt.subplots(figsize=(8, 3.2))
        grid = np.linspace(lo, hi, 400)
        frame_y = f(grid)
        frame_ax.plot(grid, frame_y, label=dep_name)
        val = float(f(frame_x))
        if np.isfinite(val):
            frame_ax.scatter([frame_x], [val], color="red", zorder=4)
        frame_ax.set_xlabel(f"{indep_name} ({indep_unit})"); frame_ax.set_ylabel(f"{dep_name} ({dep_unit})")
        frame_ax.grid(True, alpha=.2); frame_ax.legend(); frame_fig.tight_layout()
        placeholder.pyplot(frame_fig, use_container_width=True)
        plt.close(frame_fig)
        time.sleep(1 / speed)
    st.caption("Animation completed. Use the input slider to inspect a specific point. Pause/hold stops automatic playback between runs.")
# Numerical plotting masks undefined values so discontinuities are not connected.
grid = np.linspace(lo, hi, 800)
ygrid = f(grid)
valid = np.isfinite(ygrid) & (np.abs(ygrid) < 1e8)
ygrid = np.where(valid, ygrid, np.nan)
fig2, ax2 = plt.subplots(figsize=(10, 4.8))
if show_f:
    ax2.plot(grid, ygrid, label=f"f(x) = {dep_name}", linewidth=2)
if show_df:
    dydx = df(grid)
    dydx = np.where(np.isfinite(dydx) & (np.abs(dydx) < 1e8), dydx, np.nan)
    ax2.plot(grid, dydx, label="f′(x)", alpha=.85)
if show_ddf:
    d2ydx2 = ddf(grid)
    d2ydx2 = np.where(np.isfinite(d2ydx2) & (np.abs(d2ydx2) < 1e8), d2ydx2, np.nan)
    ax2.plot(grid, d2ydx2, label="f″(x)", alpha=.85)
point_y = float(f(point)); slope = float(df(point))
if np.isfinite(point_y):
    ax2.scatter([point], [point_y], color="red", zorder=5, label="Current point")
    if np.isfinite(slope):
        tangent_x = np.linspace(max(lo, point-(hi-lo)*.18), min(hi, point+(hi-lo)*.18), 60)
        ax2.plot(tangent_x, point_y + slope*(tangent_x-point), linestyle="--", label="Tangent line at current point")
if abs(b_sec - a_sec) > 1e-9:
    ya, yb = float(f(a_sec)), float(f(b_sec))
    if np.isfinite(ya) and np.isfinite(yb):
        sec_slope = (yb-ya)/(b_sec-a_sec)
        sx = np.linspace(a_sec, b_sec, 60)
        ax2.plot(sx, ya+sec_slope*(sx-a_sec), linestyle=":", linewidth=2, label="Secant line")
ax2.set_xlabel(f"{indep_name} ({indep_unit})"); ax2.set_ylabel("Function / derivative values")
ax2.set_title("Function and derivative analysis")
ax2.grid(True, alpha=.22); ax2.legend(fontsize=8); fig2.tight_layout()
st.pyplot(fig2, use_container_width=True)
function_png_buf = io.BytesIO()
fig2.savefig(function_png_buf, format="png", dpi=140, bbox_inches="tight")
function_png = function_png_buf.getvalue()
plt.close(fig2)

avg_rate = None
if abs(b_sec-a_sec) > 1e-9:
    ya, yb = float(f(a_sec)), float(f(b_sec))
    if np.isfinite(ya) and np.isfinite(yb):
        avg_rate = (yb-ya)/(b_sec-a_sec)
st.markdown("#### Live values and contextual meaning")
v1, v2, v3 = st.columns(3)
v1.metric(f"{dep_name} at x = {fmt_num(point)}", f"{fmt_num(point_y)} {dep_unit}")
v2.metric("Instantaneous rate f′(x)", f"{fmt_num(slope)} {dep_unit}/{indep_unit}")
v3.metric("Second derivative f″(x)", f"{fmt_num(float(ddf(point)))} {dep_unit}/{indep_unit}²")
if avg_rate is not None:
    st.write(f"Average rate of change from x = {fmt_num(a_sec)} to x = {fmt_num(b_sec)}: **{fmt_num(avg_rate)} {dep_unit}/{indep_unit}**.")
else:
    st.write("Average rate is undefined for this interval or when the endpoints are equal.")
st.write(f"At the selected point, the function value is {fmt_num(point_y)} {dep_unit}. The first derivative is the instantaneous rate of change in {dep_name} per {indep_unit}; the second derivative describes how that rate changes, in {dep_unit}/{indep_unit}².")
st.latex(r"f'(x) = " + sp.latex(derivative))
st.latex(r"f''(x) = " + sp.latex(second_derivative))

if course == "Calculus":
    st.markdown("#### Critical points, extrema, concavity, and domain")
    # Candidate roots are numerical and are explicitly labelled approximations.
    critical = []
    inflections = []
    try:
        roots = sp.solve(sp.Eq(derivative, 0), x)
        for root in roots:
            if root.is_real is True:
                rv = float(root)
                if lo <= rv <= hi and np.isfinite(float(f(rv))):
                    critical.append(rv)
    except Exception:
        critical = []
    try:
        roots2 = sp.solve(sp.Eq(second_derivative, 0), x)
        for root in roots2:
            if root.is_real is True:
                rv = float(root)
                if lo < rv < hi:
                    eps = max(1e-4, (hi-lo)*1e-4)
                    left2, right2 = float(ddf(rv-eps)), float(ddf(rv+eps))
                    if np.isfinite(left2) and np.isfinite(right2) and left2*right2 < 0:
                        inflections.append(rv)
    except Exception:
        inflections = []
    # Include endpoints when finding absolute extrema on the closed interval.
    candidate_x = [lo, hi] + critical
    candidate_vals = [(z, float(f(z))) for z in candidate_x if np.isfinite(float(f(z)))]
    if candidate_vals:
        abs_min = min(candidate_vals, key=lambda p:p[1]); abs_max = max(candidate_vals, key=lambda p:p[1])
        st.write(f"**Absolute minimum on the sampled critical-point candidates and endpoints:** x ≈ {fmt_num(abs_min[0])}, f(x) ≈ {fmt_num(abs_min[1])}.")
        st.write(f"**Absolute maximum on the sampled critical-point candidates and endpoints:** x ≈ {fmt_num(abs_max[0])}, f(x) ≈ {fmt_num(abs_max[1])}.")
    if critical:
        st.write("**Critical points (numerically/symbolically solved):** " + ", ".join(f"x ≈ {fmt_num(z)}, f(x) ≈ {fmt_num(float(f(z)))}" for z in critical))
        for z in critical:
            eps = max(1e-4, (hi-lo)*1e-4)
            left_s, right_s = float(df(z-eps)), float(df(z+eps))
            if left_s > 0 and right_s < 0:
                kind = "local maximum (f′ changes + to −)"
            elif left_s < 0 and right_s > 0:
                kind = "local minimum (f′ changes − to +)"
            else:
                kind = "not classified as a local extremum by the first-derivative sign test"
            st.write(f"- x ≈ {fmt_num(z)}: {kind}.")
    else:
        st.write("No critical points were found by the symbolic solver in this interval. This does not prove none exist for every function.")
    if inflections:
        st.write("**Inflection points with a verified numerical concavity change:** " + ", ".join(f"(x ≈ {fmt_num(z)}, f(x) ≈ {fmt_num(float(f(z)))})" for z in inflections))
    else:
        st.write("No inflection points with a verified numerical concavity change were found in this interval.")
    st.write("**Increasing/decreasing and concavity:** use the plotted signs of f′ and f″ and the derivative values above; numerical solvers may miss roots for complicated functions.")
    try:
        st.write("**Symbolic domain:** " + str(sp.calculus.util.continuous_domain(expr, x, sp.Interval(lo, hi))))
    except Exception:
        st.write("**Domain note:** points where the function is undefined appear as gaps in the graph; inspect the expression and interval.")
    st.caption("Extrema and inflection analysis is symbolic where possible and numerical for sign-change checks. For complicated transcendental functions, results may be incomplete; asymptotes and discontinuities are not exhaustively detected.")

st.subheader("6. Export current report")
status = "User-modified scenario" if st.session_state.edited else "Source data"
working_export = validated.copy()
working_export["fitted_value"] = fitted_values
working_export["residual"] = residuals
summary = pd.DataFrame([
    ["Country", country_label], ["Indicator", indicator_label], ["Indicator code", indicator_code], ["Unit", indicator_unit],
    ["Source URL", source_url], ["Retrieved date", date.today().isoformat()], ["Working-data status", status],
    ["Polynomial degree", degree], ["Base year", base_year], ["Equation", eq], ["In-sample R²", r2], ["In-sample MAE", mae],
    ["Historical range", f"{int(fitted.year.min())}-{int(fitted.year.max())}"], ["Observation count", len(fitted)],
], columns=["field", "value"])
csv_bytes = working_export.to_csv(index=False).encode("utf-8")
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    working_export.to_excel(writer, sheet_name="Working data", index=False)
    st.session_state.get("batch_results", pd.DataFrame()).to_excel(writer, sheet_name="Calculated years", index=False)
pdf_metrics = [["Metric", "Value"], *[[r[0], r[1]] for r in metrics[1:]]]
findings = [
    f"Model: degree-{degree} polynomial fit to {len(fitted)} working observations.",
    f"In-sample R² = {fmt_num(r2)}; in-sample MAE = {fmt_num(mae)} {indicator_unit}. These describe fit to the data used.",
    holdout_text,
    "Polynomial extrapolations can grow rapidly and should not be treated as reliable forecasts without domain-specific validation.",
    "Correlation and fitted curves do not establish causes. Missing years and country-specific coverage affect comparisons.",
    f"Working data status: {status}. Source data are available from the World Bank API."
]
pdf_bytes = make_pdf("Hispanic Heritage Historical Data Explorer", [
    f"Country: {country_label} | Indicator: {indicator_label} ({indicator_code}) | Units: {indicator_unit}",
    f"Source: {source_url} | Retrieved: {date.today().isoformat()} | Status: {status}",
], eq, pdf_metrics, chart_png, working_export.head(25), st.session_state.get("batch_results", pd.DataFrame()), findings)
ex1, ex2, ex3 = st.columns(3)
ex1.download_button("Download working data CSV", csv_bytes, file_name="hispanic_heritage_working_data.csv", mime="text/csv", use_container_width=True)
ex2.download_button("Download report Excel", xlsx_buf.getvalue(), file_name="hispanic_heritage_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
ex3.download_button("Download report PDF", pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf", use_container_width=True)

if printer:
    st.markdown("---")
    st.subheader("Printer-friendly summary")
    st.write(f"**{country_label} — {indicator_label}** ({indicator_unit})")
    st.write(f"Coverage: {int(fitted.year.min())}–{int(fitted.year.max())}; n = {len(fitted)}. Data status: {status}.")
    st.write(f"Equation: {eq}")
    st.write(f"In-sample R²: {fmt_num(r2)} | In-sample MAE: {fmt_num(mae)} {indicator_unit}")
    st.write("Limitations: missing observations, polynomial model assumptions, extrapolation uncertainty, and no causal interpretation.")

with st.expander("Sources, definitions, and model limitations"):
    st.markdown("- World Bank indicators API: https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=200")
    st.markdown("- Indicator metadata: https://api.worldbank.org/v2/indicator")
    st.markdown("- This app retrieves annual observations; actual coverage differs by country and indicator.")
    st.markdown("- The centered polynomial uses t = year − base year to improve numerical conditioning. High-degree polynomials may overfit and extrapolate poorly.")
    st.markdown("- Residuals and MAE summarize in-sample errors. The chronological holdout MAE is a separate check, not a guarantee of future performance.")
    st.markdown("- Function-lab critical points and inflections may be incomplete for difficult expressions. Inflections are only reported when a numerical concavity sign change is confirmed.")
