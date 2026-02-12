#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trade_setup_app.py

Final, fully expanded Streamlit app with no omissions.
It provides:

- Data upload & indicator calc
- Volatility Zone definition
- Condition builders for start/continue/reentry (Long/Short, High/Low)
- Trade sizing (lots/currency), plus reentry sizes
- Target & Stop settings (initial & reentry) with "allow reentry after exit" toggles
- Global Trailing Stop (both ATR & Indicator)
- Entry-Level Trailing Stops (initial & reentry), each with "allow reentry after TS" toggles
- Preferred Trend Start & Target vs Stop preference
- Export Setup as JSON
- Normal Simulation (apply_trade_setup_df) & saving CSVs
- Variant Simulation (simulate_variant) & saving CSVs
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import logging
import plotly.express as px

# Import the final core_processing code (with no omissions):
from core_processing import (
    apply_trade_setup_df,
    simulate_variant,
    get_normalized_trades,
    condition_to_expr,               # If you want to debug expressions
    evaluate_condition_structured,   # If you want to debug or test
    simulate_ma_combo,
    simulate_indicator_percentile_combo,
    simulate_breakout_combo
)

st.set_page_config(page_title="Combined Trade Setup Designer", layout="wide")
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# THEME SWITCH
# -----------------------------------------------------------------------------
theme_choice = st.sidebar.radio("Theme", ["Dark", "Light"], index=0, key="theme_choice")
if theme_choice == "Dark":
    css = """
    <style>
    body {
        background-color: #1e1e1e; 
        color: #e0e0e0;
    }
    </style>
    """
else:
    css = """
    <style>
    body {
        background-color: #f7f7f7;
        color: #333333;
    }
    </style>
    """
st.markdown(css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# JSON Setup Upload Option
# -----------------------------------------------------------------------------
json_uploaded = st.sidebar.file_uploader("Upload Setup JSON", type=["json"], key="json_uploader")
populate_from_json = st.sidebar.checkbox("Populate fields from uploaded JSON", key="populate_from_json")
if json_uploaded is not None and populate_from_json:
    try:
        json_str = json_uploaded.read().decode("utf-8")
        json_data = json.loads(json_str)
        st.session_state["json_settings"] = json_data
    except Exception as e:
        st.error("Error loading JSON file: " + str(e))

def get_default(key, default_value):
    if "json_settings" in st.session_state and key in st.session_state["json_settings"]:
        return st.session_state["json_settings"][key]
    else:
        return default_value

# -----------------------------------------------------------------------------
# SIDEBAR - DATA & BASIC SETTINGS
# -----------------------------------------------------------------------------
st.sidebar.header("Data & Basic Settings")

decimals = st.sidebar.number_input("Decimal Precision", value=get_default("decimals", 8), min_value=0, max_value=10, step=1, key="decimal_precision")
fmt = f"%.{decimals}f"

tick_size = st.sidebar.number_input("Tick Size", value=get_default("tick_size", 0.00005), format=fmt, key="tick_size")
tick_value= st.sidebar.number_input("Tick Value", value=get_default("tick_value", 5.0), format=fmt, key="tick_value")

normal_output_prefix = st.sidebar.text_input("Normal Output Filename Prefix", value=get_default("normal_output_prefix", "mySim"), key="normal_out_prefix")
variant_output_suffix= st.sidebar.text_input("Variant File Suffix", value=get_default("variant_output_suffix", "testVar"), key="variant_out_suffix")

# Data Upload
uploaded_file = st.sidebar.file_uploader("Upload OHLC CSV file", type=["csv"], key="file_uploader")
if uploaded_file:
    temp_file = "temp_ohlc.csv"
    with open(temp_file,"wb") as f:
        f.write(uploaded_file.getbuffer())
    # Suppose we have data_loader.py:
    import data_loader
    loaded_data = data_loader.load_data(temp_file)
    if loaded_data is not None:
        st.sidebar.success("Data loaded successfully.")
        # Suppose we have indicators.py:
        import indicators
        df_proc = indicators.calculate_indicators(loaded_data)
        df = df_proc.copy()
    else:
        st.sidebar.error("Error loading the CSV. Check format.")
else:
    df = pd.DataFrame()

# -----------------------------------------------------------------------------
# VOLATILITY ZONE DEFINITION (MANUAL)
# -----------------------------------------------------------------------------

# Define the default volatility indicator
default_vol_indicator = "(Range_360_Range_120)_ATR_20"

# Build a list of available indicators from the DataFrame.
# Ensure the default indicator is in the list.
if not df.empty:
    available_indicators = df.columns.tolist()
else:
    available_indicators = []
if default_vol_indicator not in available_indicators:
    available_indicators.insert(0, default_vol_indicator)

# HIGH VOLATILITY CONDITION INPUTS
high_indicator = st.sidebar.selectbox(
    "High Volatility Indicator",
    options=available_indicators,
    index=available_indicators.index(default_vol_indicator),
    key="HighVolInd"
)
high_operator = st.sidebar.selectbox(
    "High Volatility Operator",
    options=["<", ">", "<=", ">=", "=="],
    index=["<", ">", "<=", ">=", "=="].index(get_default("HighVolOp", "<")),
    key="HighVolOp"
)
high_threshold = st.sidebar.number_input(
    "High Volatility Threshold",
    value=get_default("HighVolThr", 10.0),
    format=fmt,
    key="HighVolThr"
)

# LOW VOLATILITY CONDITION INPUTS
low_indicator = st.sidebar.selectbox(
    "Low Volatility Indicator",
    options=available_indicators,
    index=available_indicators.index(default_vol_indicator),
    key="LowVolInd"
)
low_operator = st.sidebar.selectbox(
    "Low Volatility Operator",
    options=["<", ">", "<=", ">=", "=="],
    index=["<", ">", "<=", ">=", "=="].index(get_default("LowVolOp", ">=")),
    key="LowVolOp"
)
low_threshold = st.sidebar.number_input(
    "Low Volatility Threshold",
    value=get_default("LowVolThr", 10.0),
    format=fmt,
    key="LowVolThr"
)

if not df.empty:
    import numpy as np
    
    # Create boolean Series for high and low conditions using vectorized comparisons
    try:
        # Assuming that the comparison operators are among these common ones:
        if high_operator == "<":
            high_cond = df[high_indicator] < high_threshold
        elif high_operator == ">":
            high_cond = df[high_indicator] > high_threshold
        elif high_operator == "<=":
            high_cond = df[high_indicator] <= high_threshold
        elif high_operator == ">=":
            high_cond = df[high_indicator] >= high_threshold
        elif high_operator == "==":
            high_cond = df[high_indicator] == high_threshold
        else:
            high_cond = np.zeros(len(df), dtype=bool)
    
        if low_operator == "<":
            low_cond = df[low_indicator] < low_threshold
        elif low_operator == ">":
            low_cond = df[low_indicator] > low_threshold
        elif low_operator == "<=":
            low_cond = df[low_indicator] <= low_threshold
        elif low_operator == ">=":
            low_cond = df[low_indicator] >= low_threshold
        elif low_operator == "==":
            low_cond = df[low_indicator] == low_threshold
        else:
            low_cond = np.zeros(len(df), dtype=bool)
    except Exception:
        # In case of any error, default to a Series of False
        high_cond = np.zeros(len(df), dtype=bool)
        low_cond = np.zeros(len(df), dtype=bool)
    
    # Use vectorized operations to create the new column
    df["Volatility_Zone"] = np.where(high_cond, "High", np.where(low_cond, "Low", "Undefined"))
    
    
st.sidebar.subheader("Available Indicators")
ind_search= st.sidebar.text_input("Search Indicators", value=get_default("IndSearch", ""), key="IndSearch")
if not df.empty:
    all_cols = df.columns.tolist()
    filtered_indicators = [c for c in all_cols if ind_search.lower() in c.lower()] if ind_search else all_cols
    st.sidebar.dataframe(pd.DataFrame({"Indicator":filtered_indicators}), use_container_width=True)
else:
    filtered_indicators = []

# -----------------------------------------------------------------------------
# HELPER: Condition Builders
# -----------------------------------------------------------------------------
def condition_to_expr(condition):
    if condition["type"]=="simple":
        left = str(condition["left_value"]) if condition["left_type"]=="constant" else condition["left_value"]
        right= str(condition["right_value"]) if condition["right_type"]=="constant" else condition["right_value"]
        return f"({left} {condition['operator']} {right})"
    elif condition["type"]=="group":
        exprs=[condition_to_expr(c) for c in condition["conditions"]]
        joiner= " and " if condition["logic"]=="AND" else " or "
        return "(" + joiner.join(exprs)+ ")"
    else:
        return ""

def get_condition_group(key_prefix, indicator_list, default=None):
    st.markdown(f"**Condition Group: {key_prefix}**")
    if default:
        # If 'n' is not explicitly provided, use the length of the conditions list.
        default_conditions = default.get("conditions", [])
        default_n = default.get("n", len(default_conditions)) if default_conditions else 1
        default_logic = default.get("logic","AND")
    else:
        default_n = 1
        default_logic = "AND"
        default_conditions = []
        
    logic= st.selectbox(
        f"{key_prefix} - Logic",
        ["AND","OR"],
        index=["AND","OR"].index(default_logic.upper()),
        key=f"{key_prefix}_logic"
    )
    n= st.number_input(
        f"{key_prefix} - # Conditions",
        min_value=1, max_value=5,
        value=default_n,
        key=f"{key_prefix}_n"
    )
    conditions=[]
    for i in range(int(n)):
        default_cond = default_conditions[i] if i < len(default_conditions) else None
        default_type = default_cond.get("type", "simple").lower() if default_cond else "simple"
        cond_type = st.selectbox(
            f"{key_prefix} Condition {i+1} Type",
            ["Simple","Nested Group"],
            index=0 if default_type=="simple" else 1,
            key=f"{key_prefix}_condtype_{i}"
        )
        if cond_type=="Simple":
            st.markdown(f"*Simple Condition {i+1}*")
            def_lt = default_cond.get("left_type", "Indicator").title() if default_cond else "Indicator"
            left_type = st.selectbox("Left Type", ["Indicator","Constant"], index=["Indicator","Constant"].index(def_lt), key=f"{key_prefix}_lt_{i}")
            if left_type=="Indicator":
                if indicator_list:
                    def_lv = default_cond.get("left_value", indicator_list[0]) if default_cond else indicator_list[0]
                    if def_lv not in indicator_list:
                        def_lv = indicator_list[0]
                    left_val = st.selectbox("Left Operand", indicator_list, index=indicator_list.index(def_lv), key=f"{key_prefix}_lval_{i}")
                else:
                    left_val = ""
            else:
                def_lv = default_cond.get("left_value", 0) if default_cond else 0
                left_val = st.number_input("Left Constant", value=float(def_lv), format=fmt, key=f"{key_prefix}_lconst_{i}")

            def_op = default_cond.get("operator", ">") if default_cond else ">"
            operator = st.selectbox("Operator", [">","<",">=","<=","==","!="], index=[">","<",">=","<=","==","!="].index(def_op), key=f"{key_prefix}_op_{i}")

            def_rt = default_cond.get("right_type", "Indicator").title() if default_cond else "Indicator"
            right_type = st.selectbox("Right Type", ["Indicator","Constant"], index=["Indicator","Constant"].index(def_rt), key=f"{key_prefix}_rt_{i}")
            if right_type=="Indicator":
                if indicator_list:
                    def_rv = default_cond.get("right_value", indicator_list[0]) if default_cond else indicator_list[0]
                    if def_rv not in indicator_list:
                        def_rv = indicator_list[0]
                    right_val = st.selectbox("Right Operand", indicator_list, index=indicator_list.index(def_rv), key=f"{key_prefix}_rval_{i}")
                else:
                    right_val = ""
            else:
                def_rv = default_cond.get("right_value", 0) if default_cond else 0
                right_val = st.number_input("Right Constant", value=float(def_rv), format=fmt, key=f"{key_prefix}_rconst_{i}")

            conditions.append({
                "type": "simple",
                "left_type": left_type.lower(),
                "left_value": left_val,
                "operator": operator,
                "right_type": right_type.lower(),
                "right_value": right_val
            })
        else:
            st.markdown(f"*Nested Condition {i+1}*")
            nested_default = default_cond if (default_cond and default_cond.get("type", "simple").lower() == "group") else None
            nested = get_condition_group(f"{key_prefix}_nested{i+1}", indicator_list, default=nested_default)
            conditions.append(nested)

    group = {
        "type": "group",
        "logic": logic.upper(),
        "conditions": conditions
    }
    expr_gen = condition_to_expr(group)
    st.caption(f"Generated Expression: {expr_gen}")
    return group

# We can define some quick defaults if desired:
default_long_start = {
    "logic": "AND",
    "n": 2,
    "conditions": [
        {"type": "simple", "left_type": "Indicator", "left_value": "SMA_5", "operator": ">", "right_type": "Indicator", "right_value": "SMA_10"},
        {"type": "simple", "left_type": "Indicator", "left_value": "Open", "operator": ">", "right_type": "Indicator", "right_value": "SMA_5"}
    ]
}
default_long_continue = {
    "logic": "AND",
    "n": 1,
    "conditions": [
        {"type": "simple", "left_type": "Indicator", "left_value": "SMA_5", "operator": ">", "right_type": "Indicator", "right_value": "SMA_10"}
    ]
}
default_short_start = {
    "logic": "AND",
    "n": 2,
    "conditions": [
        {"type": "simple", "left_type": "Indicator", "left_value": "SMA_5", "operator": "<", "right_type": "Indicator", "right_value": "SMA_10"},
        {"type": "simple", "left_type": "Indicator", "left_value": "Open", "operator": "<", "right_type": "Indicator", "right_value": "SMA_5"}
    ]
}
default_short_continue = {
    "logic": "AND",
    "n": 1,
    "conditions": [
        {"type": "simple", "left_type": "Indicator", "left_value": "SMA_5", "operator": "<", "right_type": "Indicator", "right_value": "SMA_10"}
    ]
}

# -----------------------------------------------------------------------------
# MAIN TABS
# -----------------------------------------------------------------------------
st.title("Trade Setup Inputs")

setup_tabs = st.tabs([
    "Initial Trend Entry Conditions",
    "Reentry Conditions",
    "Trade Sizing & Entry Size",
    "Target Settings",
    "Stop Settings",
    "Global Trailing Stops",
    "Entry-Level Trailing Stops",
    "Preferred Trend Start",
    "Target vs Stop Preference",
    "Export & Simulation",
    "MA Combo Simulation",
    "Indicator Percentile Simulation",
    "Support/Resistance Breakout Simulation"
])

# 1) Initial Trend Entry
with setup_tabs[0]:
    st.subheader("Initial Trend Entry Conditions")
    st.write("### Long Trends")
    with st.expander("Long, High Volatility", expanded=True):
        long_high_start = get_condition_group("Long_High_Start", filtered_indicators, default=get_default("long_high_start", default_long_start))
        long_high_continue = get_condition_group("Long_High_Continue", filtered_indicators, default=get_default("long_high_continue", default_long_continue))
    with st.expander("Long, Low Volatility", expanded=True):
        long_low_start = get_condition_group("Long_Low_Start", filtered_indicators, default=get_default("long_low_start", default_long_start))
        long_low_continue = get_condition_group("Long_Low_Continue", filtered_indicators, default=get_default("long_low_continue", default_long_continue))

    st.write("### Short Trends")
    with st.expander("Short, High Volatility", expanded=True):
        short_high_start = get_condition_group("Short_High_Start", filtered_indicators, default=get_default("short_high_start", default_short_start))
        short_high_continue = get_condition_group("Short_High_Continue", filtered_indicators, default=get_default("short_high_continue", default_short_continue))
    with st.expander("Short, Low Volatility", expanded=True):
        short_low_start = get_condition_group("Short_Low_Start", filtered_indicators, default=get_default("short_low_start", default_short_start))
        short_low_continue = get_condition_group("Short_Low_Continue", filtered_indicators, default=get_default("short_low_continue", default_short_continue))

# 2) Reentry Conditions
with setup_tabs[1]:
    st.subheader("Reentry Conditions")
    st.markdown("#### Long Reentry Conditions")
    with st.expander("Long, High Volatility", expanded=True):
        long_high_reentry = get_condition_group("Long_High_Reentry", filtered_indicators, default=get_default("long_high_reentry", None))
    with st.expander("Long, Low Volatility", expanded=True):
        long_low_reentry = get_condition_group("Long_Low_Reentry", filtered_indicators, default=get_default("long_low_reentry", None))

    st.markdown("#### Short Reentry Conditions")
    with st.expander("Short, High Volatility", expanded=True):
        short_high_reentry = get_condition_group("Short_High_Reentry", filtered_indicators, default=get_default("short_high_reentry", None))
    with st.expander("Short, Low Volatility", expanded=True):
        short_low_reentry = get_condition_group("Short_Low_Reentry", filtered_indicators, default=get_default("short_low_reentry", None))

# 3) Trade Sizing
with setup_tabs[2]:
    st.subheader("Trade Sizing & Entry Size")
    col1, col2 = st.columns(2)
    with col1:
        initial_entry_type = st.radio("Entry Size Type", ["lots", "currency"], index=["lots", "currency"].index(get_default("initial-entry-type", "lots")), horizontal=True)
        init_long_high_size = st.number_input("Initial (Long,High)", value=get_default("init_long_high_size", 1.0), step=1.0, format=fmt)
        init_long_low_size = st.number_input("Initial (Long,Low)", value=get_default("init_long_low_size", 2.0), step=1.0, format=fmt)
    with col2:
        init_short_high_size = st.number_input("Initial (Short,High)", value=get_default("init_short_high_size", 3.0), step=1.0, format=fmt)
        init_short_low_size = st.number_input("Initial (Short,Low)", value=get_default("init_short_low_size", 4.0), step=1.0, format=fmt)

    max_reentries = st.number_input("Max Reentries Allowed per Trend", value=get_default("max_reentries", 2), step=1)

    st.markdown("**Reentry Sizes**:")
    reentry_long_high_sizes, reentry_long_low_sizes = [], []
    reentry_short_high_sizes, reentry_short_low_sizes = [], []
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("Long Reentries")
        for i in range(int(max_reentries)):
            default_val_lh = get_default("reentry_long_high_sizes", [])
            lh_default = default_val_lh[i] if len(default_val_lh) > i else 1.0
            lh_ = st.number_input(f"Long High Reentry {i+1}", value=lh_default, step=1.0, format=fmt, key=f"reLH_{i}")
            default_val_ll = get_default("reentry_long_low_sizes", [])
            ll_default = default_val_ll[i] if len(default_val_ll) > i else 2.0
            ll_ = st.number_input(f"Long Low Reentry {i+1}", value=ll_default, step=1.0, format=fmt, key=f"reLL_{i}")
            reentry_long_high_sizes.append(lh_)
            reentry_long_low_sizes.append(ll_)
    with c4:
        st.markdown("Short Reentries")
        for i in range(int(max_reentries)):
            default_val_sh = get_default("reentry_short_high_sizes", [])
            sh_default = default_val_sh[i] if len(default_val_sh) > i else 3.0
            sh_ = st.number_input(f"Short High Reentry {i+1}", value=sh_default, step=1.0, format=fmt, key=f"reSH_{i}")
            default_val_sl = get_default("reentry_short_low_sizes", [])
            sl_default = default_val_sl[i] if len(default_val_sl) > i else 4.0
            sl_ = st.number_input(f"Short Low Reentry {i+1}", value=sl_default, step=1.0, format=fmt, key=f"reSL_{i}")
            reentry_short_high_sizes.append(sh_)
            reentry_short_low_sizes.append(sl_)

# 4) Target Settings
with setup_tabs[3]:
    st.subheader("Target Settings")
    target_defined = st.radio("Define Targets?", ["Yes", "No"], index=["Yes", "No"].index(get_default("target_defined", "No") if isinstance(get_default("target_defined", "No"), str) else ("Yes" if get_default("target_defined", False) else "No")), horizontal=True)
    
    with st.expander("Initial Target Settings", expanded=(target_defined == "Yes")):
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("**Long Targets** (High & Low)")
            default_method = get_default("init_target_method_long_high", "absolute")
            index_m = ["absolute", "atr", "percentage"].index(default_method) if default_method in ["absolute", "atr", "percentage"] else 0
            init_target_method_long_high = st.selectbox("Method (Long,High)", ["absolute", "atr", "percentage"], index=index_m, key="init_t_m_lh")
            init_target_long_high = st.number_input("Value (Long,High)", value=get_default("init_target_long_high", 0.0), step=0.1, format=fmt, key="init_t_v_lh")
            init_target_allow_reentry_long_high = st.checkbox("Allow Reentry After Target? (Long,High,Initial)", value=get_default("init_target_allow_reentry_long_high", False), key="init_t_allowR_lh")

            default_method_ll = get_default("init_target_method_long_low", "absolute")
            index_m_ll = ["absolute", "atr", "percentage"].index(default_method_ll) if default_method_ll in ["absolute", "atr", "percentage"] else 0
            init_target_method_long_low = st.selectbox("Method (Long,Low)", ["absolute", "atr", "percentage"], index=index_m_ll, key="init_t_m_ll")
            init_target_long_low = st.number_input("Value (Long,Low)", value=get_default("init_target_long_low", 0.0), step=0.1, format=fmt, key="init_t_v_ll")
            init_target_allow_reentry_long_low = st.checkbox("Allow Reentry After Target? (Long,Low,Initial)", value=get_default("init_target_allow_reentry_long_low", False), key="init_t_allowR_ll")

        with c6:
            st.markdown("**Short Targets** (High & Low)")
            default_method_sh = get_default("init_target_method_short_high", "absolute")
            index_m_sh = ["absolute", "atr", "percentage"].index(default_method_sh) if default_method_sh in ["absolute", "atr", "percentage"] else 0
            init_target_method_short_high = st.selectbox("Method (Short,High)", ["absolute", "atr", "percentage"], index=index_m_sh, key="init_t_m_sh")
            init_target_short_high = st.number_input("Value (Short,High)", value=get_default("init_target_short_high", 0.0), step=0.1, format=fmt, key="init_t_v_sh")
            init_target_allow_reentry_short_high = st.checkbox("Allow Reentry After Target? (Short,High,Initial)", value=get_default("init_target_allow_reentry_short_high", False), key="init_t_allowR_sh")

            default_method_sl = get_default("init_target_method_short_low", "absolute")
            index_m_sl = ["absolute", "atr", "percentage"].index(default_method_sl) if default_method_sl in ["absolute", "atr", "percentage"] else 0
            init_target_method_short_low = st.selectbox("Method (Short,Low)", ["absolute", "atr", "percentage"], index=index_m_sl, key="init_t_m_sl")
            init_target_short_low = st.number_input("Value (Short,Low)", value=get_default("init_target_short_low", 0.0), step=0.1, format=fmt, key="init_t_v_sl")
            init_target_allow_reentry_short_low = st.checkbox("Allow Reentry After Target? (Short,Low,Initial)", value=get_default("init_target_allow_reentry_short_low", False), key="init_t_allowR_sl")

    with st.expander("Reentry Target Settings", expanded=(target_defined == "Yes")):
        st.markdown("For each reentry (up to max_reentries): define method/value + allow reentry toggle.")
        reentry_target_method_long_high, reentry_target_long_high = [], []
        reentry_target_allow_reentry_long_high = []
        reentry_target_method_long_low, reentry_target_long_low = [], []
        reentry_target_allow_reentry_long_low = []
        reentry_target_method_short_high, reentry_target_short_high = [], []
        reentry_target_allow_reentry_short_high = []
        reentry_target_method_short_low, reentry_target_short_low = [], []
        reentry_target_allow_reentry_short_low = []
        for i in range(int(max_reentries)):
            st.markdown(f"**Reentry #{i+1}**")
            colA, colB = st.columns(2)
            with colA:
                st.write("Long, High")
                default_method = get_default("reentry_target_method_long_high", [])
                method_default = default_method[i] if len(default_method) > i else "absolute"
                index_method = ["absolute", "atr", "percentage"].index(method_default) if method_default in ["absolute", "atr", "percentage"] else 0
                m_lh = st.selectbox(f"Method (LH Re#{i+1})", ["absolute", "atr", "percentage"], index=index_method, key=f"re_t_m_lh_{i}")
                default_val = get_default("reentry_target_long_high", [])
                v_lh = default_val[i] if len(default_val) > i else 0.0
                v_lh = st.number_input(f"Value (LH Re#{i+1})", value=v_lh, step=0.1, format=fmt, key=f"re_t_v_lh_{i}")
                default_allow = get_default("reentry_target_allow_reentry_long_high", [])
                allow_lh = default_allow[i] if len(default_allow) > i else False
                allow_lh = st.checkbox(f"Allow Reentry (LH Re#{i+1})", value=allow_lh, key=f"re_t_allow_lh_{i}")
                reentry_target_method_long_high.append(m_lh)
                reentry_target_long_high.append(v_lh)
                reentry_target_allow_reentry_long_high.append(allow_lh)

                st.write("Long, Low")
                default_method = get_default("reentry_target_method_long_low", [])
                method_default = default_method[i] if len(default_method) > i else "absolute"
                index_method = ["absolute", "atr", "percentage"].index(method_default) if method_default in ["absolute", "atr", "percentage"] else 0
                m_ll = st.selectbox(f"Method (LL Re#{i+1})", ["absolute", "atr", "percentage"], index=index_method, key=f"re_t_m_ll_{i}")
                default_val = get_default("reentry_target_long_low", [])
                v_ll = default_val[i] if len(default_val) > i else 0.0
                v_ll = st.number_input(f"Value (LL Re#{i+1})", value=v_ll, step=0.1, format=fmt, key=f"re_t_v_ll_{i}")
                default_allow = get_default("reentry_target_allow_reentry_long_low", [])
                allow_ll = default_allow[i] if len(default_allow) > i else False
                allow_ll = st.checkbox(f"Allow Reentry (LL Re#{i+1})", value=allow_ll, key=f"re_t_allow_ll_{i}")
                reentry_target_method_long_low.append(m_ll)
                reentry_target_long_low.append(v_ll)
                reentry_target_allow_reentry_long_low.append(allow_ll)

            with colB:
                st.write("Short, High")
                default_method = get_default("reentry_target_method_short_high", [])
                method_default = default_method[i] if len(default_method) > i else "absolute"
                index_method = ["absolute", "atr", "percentage"].index(method_default) if method_default in ["absolute", "atr", "percentage"] else 0
                m_sh = st.selectbox(f"Method (SH Re#{i+1})", ["absolute", "atr", "percentage"], index=index_method, key=f"re_t_m_sh_{i}")
                default_val = get_default("reentry_target_short_high", [])
                v_sh = default_val[i] if len(default_val) > i else 0.0
                v_sh = st.number_input(f"Value (SH Re#{i+1})", value=v_sh, step=0.1, format=fmt, key=f"re_t_v_sh_{i}")
                default_allow = get_default("reentry_target_allow_reentry_short_high", [])
                allow_sh = default_allow[i] if len(default_allow) > i else False
                allow_sh = st.checkbox(f"Allow Reentry (SH Re#{i+1})", value=allow_sh, key=f"re_t_allow_sh_{i}")
                reentry_target_method_short_high.append(m_sh)
                reentry_target_short_high.append(v_sh)
                reentry_target_allow_reentry_short_high.append(allow_sh)

                st.write("Short, Low")
                default_method = get_default("reentry_target_method_short_low", [])
                method_default = default_method[i] if len(default_method) > i else "absolute"
                index_method = ["absolute", "atr", "percentage"].index(method_default) if method_default in ["absolute", "atr", "percentage"] else 0
                m_sl = st.selectbox(f"Method (SL Re#{i+1})", ["absolute", "atr", "percentage"], index=index_method, key=f"re_t_m_sl_{i}")
                default_val = get_default("reentry_target_short_low", [])
                v_sl = default_val[i] if len(default_val) > i else 0.0
                v_sl = st.number_input(f"Value (SL Re#{i+1})", value=v_sl, step=0.1, format=fmt, key=f"re_t_v_sl_{i}")
                default_allow = get_default("reentry_target_allow_reentry_short_low", [])
                allow_sl = default_allow[i] if len(default_allow) > i else False
                allow_sl = st.checkbox(f"Allow Reentry (SL Re#{i+1})", value=allow_sl, key=f"re_t_allow_sl_{i}")
                reentry_target_method_short_low.append(m_sl)
                reentry_target_short_low.append(v_sl)
                reentry_target_allow_reentry_short_low.append(allow_sl)

# 5) Stop Settings
with setup_tabs[4]:
    st.subheader("Stop Settings")
    stop_defined = st.radio("Define Stops?", ["Yes", "No"], index=["Yes", "No"].index(get_default("stop_defined", "No") if isinstance(get_default("stop_defined", "No"), str) else ("Yes" if get_default("stop_defined", False) else "No")), horizontal=True)
    with st.expander("Initial Stop Settings", expanded=(stop_defined == "Yes")):
        c7, c8 = st.columns(2)
        with c7:
            st.markdown("**Long Stops (High & Low)**")
            default_method = get_default("init_stop_method_long_high", "absolute")
            index_method = ["absolute", "atr", "percentage"].index(default_method) if default_method in ["absolute", "atr", "percentage"] else 0
            init_stop_method_long_high = st.selectbox("Method (Long,High)", ["absolute", "atr", "percentage"], index=index_method, key="init_s_m_lh")
            init_stop_long_high = st.number_input("Value (Long,High)", value=get_default("init_stop_long_high", 0.0), step=0.1, format=fmt, key="init_s_v_lh")
            init_stop_allow_reentry_long_high = st.checkbox("Allow Reentry After Stop? (Long,High,Init)", value=get_default("init_stop_allow_reentry_long_high", False), key="init_s_allowR_lh")

            default_method = get_default("init_stop_method_long_low", "absolute")
            index_method = ["absolute", "atr", "percentage"].index(default_method) if default_method in ["absolute", "atr", "percentage"] else 0
            init_stop_method_long_low = st.selectbox("Method (Long,Low)", ["absolute", "atr", "percentage"], index=index_method, key="init_s_m_ll")
            init_stop_long_low = st.number_input("Value (Long,Low)", value=get_default("init_stop_long_low", 0.0), step=0.1, format=fmt, key="init_s_v_ll")
            init_stop_allow_reentry_long_low = st.checkbox("Allow Reentry After Stop? (Long,Low,Init)", value=get_default("init_stop_allow_reentry_long_low", False), key="init_s_allowR_ll")

        with c8:
            st.markdown("**Short Stops (High & Low)**")
            default_method = get_default("init_stop_method_short_high", "absolute")
            index_method = ["absolute", "atr", "percentage"].index(default_method) if default_method in ["absolute", "atr", "percentage"] else 0
            init_stop_method_short_high = st.selectbox("Method (Short,High)", ["absolute", "atr", "percentage"], index=index_method, key="init_s_m_sh")
            init_stop_short_high = st.number_input("Value (Short,High)", value=get_default("init_stop_short_high", 0.0), step=0.1, format=fmt, key="init_s_v_sh")
            init_stop_allow_reentry_short_high = st.checkbox("Allow Reentry After Stop? (Short,High,Init)", value=get_default("init_stop_allow_reentry_short_high", False), key="init_s_allowR_sh")

            default_method = get_default("init_stop_method_short_low", "absolute")
            index_method = ["absolute", "atr", "percentage"].index(default_method) if default_method in ["absolute", "atr", "percentage"] else 0
            init_stop_method_short_low = st.selectbox("Method (Short,Low)", ["absolute", "atr", "percentage"], index=index_method, key="init_s_m_sl")
            init_stop_short_low = st.number_input("Value (Short,Low)", value=get_default("init_stop_short_low", 0.0), step=0.1, format=fmt, key="init_s_v_sl")
            init_stop_allow_reentry_short_low = st.checkbox("Allow Reentry After Stop? (Short,Low,Init)", value=get_default("init_stop_allow_reentry_short_low", False), key="init_s_allowR_sl")

    with st.expander("Reentry Stop Settings", expanded=(stop_defined == "Yes")):
        st.markdown("For each reentry, define method/value + allow reentry toggles.")
        reentry_stop_method_long_high, reentry_stop_long_high = [], []
        reentry_stop_allow_reentry_long_high = []
        reentry_stop_method_long_low, reentry_stop_long_low = [], []
        reentry_stop_allow_reentry_long_low = []
        reentry_stop_method_short_high, reentry_stop_short_high = [], []
        reentry_stop_allow_reentry_short_high = []
        reentry_stop_method_short_low, reentry_stop_short_low = [], []
        reentry_stop_allow_reentry_short_low = []
        for i in range(int(max_reentries)):
            st.markdown(f"**Reentry #{i+1}**")
            cA, cB = st.columns(2)
            with cA:
                st.markdown("Long, High")
                default_method = get_default("reentry_stop_method_long_high", [])
                method_default = default_method[i] if len(default_method) > i else "absolute"
                index_method = ["absolute", "atr", "percentage"].index(method_default) if method_default in ["absolute", "atr", "percentage"] else 0
                ms_lh = st.selectbox(f"Stop Method (LH Re#{i+1})", ["absolute", "atr", "percentage"], index=index_method, key=f"re_s_m_lh_{i}")
                default_val = get_default("reentry_stop_long_high", [])
                vs_lh = default_val[i] if len(default_val) > i else 0.0
                vs_lh = st.number_input(f"Stop Value (LH Re#{i+1})", value=vs_lh, step=0.1, format=fmt, key=f"re_s_v_lh_{i}")
                default_allow = get_default("reentry_stop_allow_reentry_long_high", [])
                allow_lh = default_allow[i] if len(default_allow) > i else False
                allow_lh = st.checkbox(f"Allow Reentry After Stop (LH Re#{i+1})", value=allow_lh, key=f"re_s_allow_lh_{i}")
                reentry_stop_method_long_high.append(ms_lh)
                reentry_stop_long_high.append(vs_lh)
                reentry_stop_allow_reentry_long_high.append(allow_lh)

                st.markdown("Long, Low")
                default_method = get_default("reentry_stop_method_long_low", [])
                method_default = default_method[i] if len(default_method) > i else "absolute"
                index_method = ["absolute", "atr", "percentage"].index(method_default) if method_default in ["absolute", "atr", "percentage"] else 0
                ms_ll = st.selectbox(f"Stop Method (LL Re#{i+1})", ["absolute", "atr", "percentage"], index=index_method, key=f"re_s_m_ll_{i}")
                default_val = get_default("reentry_stop_long_low", [])
                vs_ll = default_val[i] if len(default_val) > i else 0.0
                vs_ll = st.number_input(f"Stop Value (LL Re#{i+1})", value=vs_ll, step=0.1, format=fmt, key=f"re_s_v_ll_{i}")
                default_allow = get_default("reentry_stop_allow_reentry_long_low", [])
                allow_ll = default_allow[i] if len(default_allow) > i else False
                allow_ll = st.checkbox(f"Allow Reentry After Stop (LL Re#{i+1})", value=allow_ll, key=f"re_s_allow_ll_{i}")
                reentry_stop_method_long_low.append(ms_ll)
                reentry_stop_long_low.append(vs_ll)
                reentry_stop_allow_reentry_long_low.append(allow_ll)

            with cB:
                st.markdown("Short, High")
                default_method = get_default("reentry_stop_method_short_high", [])
                method_default = default_method[i] if len(default_method) > i else "absolute"
                index_method = ["absolute", "atr", "percentage"].index(method_default) if method_default in ["absolute", "atr", "percentage"] else 0
                ms_sh = st.selectbox(f"Stop Method (SH Re#{i+1})", ["absolute", "atr", "percentage"], index=index_method, key=f"re_s_m_sh_{i}")
                default_val = get_default("reentry_stop_short_high", [])
                vs_sh = default_val[i] if len(default_val) > i else 0.0
                vs_sh = st.number_input(f"Stop Value (SH Re#{i+1})", value=vs_sh, step=0.1, format=fmt, key=f"re_s_v_sh_{i}")
                default_allow = get_default("reentry_stop_allow_reentry_short_high", [])
                allow_sh = default_allow[i] if len(default_allow) > i else False
                allow_sh = st.checkbox(f"Allow Reentry After Stop (SH Re#{i+1})", value=allow_sh, key=f"re_s_allow_sh_{i}")
                reentry_stop_method_short_high.append(ms_sh)
                reentry_stop_short_high.append(vs_sh)
                reentry_stop_allow_reentry_short_high.append(allow_sh)

                st.markdown("Short, Low")
                default_method = get_default("reentry_stop_method_short_low", [])
                method_default = default_method[i] if len(default_method) > i else "absolute"
                index_method = ["absolute", "atr", "percentage"].index(method_default) if method_default in ["absolute", "atr", "percentage"] else 0
                ms_sl = st.selectbox(f"Stop Method (SL Re#{i+1})", ["absolute", "atr", "percentage"], index=index_method, key=f"re_s_m_sl_{i}")
                default_val = get_default("reentry_stop_short_low", [])
                vs_sl = default_val[i] if len(default_val) > i else 0.0
                vs_sl = st.number_input(f"Stop Value (SL Re#{i+1})", value=vs_sl, step=0.1, format=fmt, key=f"re_s_v_sl_{i}")
                default_allow = get_default("reentry_stop_allow_reentry_short_low", [])
                allow_sl = default_allow[i] if len(default_allow) > i else False
                allow_sl = st.checkbox(f"Allow Reentry After Stop (SL Re#{i+1})", value=allow_sl, key=f"re_s_allow_sl_{i}")
                reentry_stop_method_short_low.append(ms_sl)
                reentry_stop_short_low.append(vs_sl)
                reentry_stop_allow_reentry_short_low.append(allow_sl)

# 6) Global Trailing Stops
with setup_tabs[5]:
    st.subheader("Global Trailing Stops (closes entire trend, no more entries)")
    colX, colY = st.columns(2)
    with colX:
        st.markdown("**Long, High**")
        default_val = get_default("global_trailing_stop_type_long_high", "None")
        global_trailing_stop_type_long_high = st.selectbox("Type (Long,High)", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val) if default_val in ["None", "ATR", "Indicator", "Both"] else 0, key="gts_type_lh")
        if global_trailing_stop_type_long_high != "None":
            global_trailing_stop_atr_multiplier_long_high = st.number_input("ATR Multiplier (LH)", value=get_default("global_trailing_stop_atr_multiplier_long_high", 1.5), step=0.1, format=fmt, key="gts_atr_lh")
            c_ind = st.number_input("#Indicators (LH)", value=get_default("gts_indcount_lh", 0), step=1, key="gts_indcount_lh")
            global_trailing_stop_indicator_long_high = []
            for i in range(int(c_ind)):
                ind_ = st.selectbox(f"Indicator {i+1} (LH)", filtered_indicators, key=f"gts_lh_ind_{i}")
                global_trailing_stop_indicator_long_high.append(ind_)
        else:
            global_trailing_stop_atr_multiplier_long_high = 0.0
            global_trailing_stop_indicator_long_high = []

        st.markdown("**Long, Low**")
        default_val = get_default("global_trailing_stop_type_long_low", "None")
        global_trailing_stop_type_long_low = st.selectbox("Type (Long,Low)", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val) if default_val in ["None", "ATR", "Indicator", "Both"] else 0, key="gts_type_ll")
        if global_trailing_stop_type_long_low != "None":
            global_trailing_stop_atr_multiplier_long_low = st.number_input("ATR Multiplier (LL)", value=get_default("global_trailing_stop_atr_multiplier_long_low", 1.5), step=0.1, format=fmt, key="gts_atr_ll")
            c_ind_ll = st.number_input("#Indicators (LL)", value=get_default("gts_indcount_ll", 0), step=1, key="gts_indcount_ll")
            global_trailing_stop_indicator_long_low = []
            for i in range(int(c_ind_ll)):
                ind_ = st.selectbox(f"Indicator {i+1} (LL)", filtered_indicators, key=f"gts_ll_ind_{i}")
                global_trailing_stop_indicator_long_low.append(ind_)
        else:
            global_trailing_stop_atr_multiplier_long_low = 0.0
            global_trailing_stop_indicator_long_low = []

    with colY:
        st.markdown("**Short, High**")
        default_val = get_default("global_trailing_stop_type_short_high", "None")
        global_trailing_stop_type_short_high = st.selectbox("Type (Short,High)", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val) if default_val in ["None", "ATR", "Indicator", "Both"] else 0, key="gts_type_sh")
        if global_trailing_stop_type_short_high != "None":
            global_trailing_stop_atr_multiplier_short_high = st.number_input("ATR Multiplier (SH)", value=get_default("global_trailing_stop_atr_multiplier_short_high", 1.5), step=0.1, format=fmt, key="gts_atr_sh")
            c_ind_sh = st.number_input("#Indicators (SH)", value=get_default("gts_indcount_sh", 0), step=1, key="gts_indcount_sh")
            global_trailing_stop_indicator_short_high = []
            for i in range(int(c_ind_sh)):
                ind_ = st.selectbox(f"Indicator {i+1} (SH)", filtered_indicators, key=f"gts_sh_ind_{i}")
                global_trailing_stop_indicator_short_high.append(ind_)
        else:
            global_trailing_stop_atr_multiplier_short_high = 0.0
            global_trailing_stop_indicator_short_high = []

        st.markdown("**Short, Low**")
        default_val = get_default("global_trailing_stop_type_short_low", "None")
        global_trailing_stop_type_short_low = st.selectbox("Type (Short,Low)", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val) if default_val in ["None", "ATR", "Indicator", "Both"] else 0, key="gts_type_sl")
        if global_trailing_stop_type_short_low != "None":
            global_trailing_stop_atr_multiplier_short_low = st.number_input("ATR Multiplier (SL)", value=get_default("global_trailing_stop_atr_multiplier_short_low", 1.5), step=0.1, format=fmt, key="gts_atr_sl")
            c_ind_sl = st.number_input("#Indicators (SL)", value=get_default("gts_indcount_sl", 0), step=1, key="gts_indcount_sl")
            global_trailing_stop_indicator_short_low = []
            for i in range(int(c_ind_sl)):
                ind_ = st.selectbox(f"Indicator {i+1} (SL)", filtered_indicators, key=f"gts_sl_ind_{i}")
                global_trailing_stop_indicator_short_low.append(ind_)
        else:
            global_trailing_stop_atr_multiplier_short_low = 0.0
            global_trailing_stop_indicator_short_low = []

# 7) Entry-Level Trailing Stops
with setup_tabs[6]:
    st.subheader("Entry-Level Trailing Stops (each entry uses entry_most_fav_price)")

    st.markdown("### Initial Entry Trailing Stops")
    with st.expander("Long, High"):
        default_val = get_default("init_entry_trailing_stop_type_long_high", "None")
        init_entry_trailing_stop_type_long_high = st.selectbox("Type", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val) if default_val in ["None", "ATR", "Indicator", "Both"] else 0, key="init_ts_lh")
        init_entry_trailing_stop_atr_multiplier_long_high = st.number_input("ATR Multiplier", value=get_default("init_entry_trailing_stop_atr_multiplier_long_high", 1.0), step=0.1, format=fmt, key="init_ts_lh_atr")
        c_ind_lh = st.number_input("Indicator Count", value=get_default("init_entry_trailing_stop_indicator_count_long_high", 0), step=1, key="init_ts_lh_indcount")
        init_entry_trailing_stop_indicator_long_high = []
        for i in range(int(c_ind_lh)):
            ind_ = st.selectbox(f"Indicator {i+1}", filtered_indicators, key=f"init_ts_lh_ind_{i}")
            init_entry_trailing_stop_indicator_long_high.append(ind_)
        init_entry_trailing_stop_allow_reentry_long_high = st.checkbox("Allow Reentry After TS Exit?", value=get_default("init_entry_trailing_stop_allow_reentry_long_high", True), key="init_ts_lh_allow")

    with st.expander("Long, Low"):
        default_val = get_default("init_entry_trailing_stop_type_long_low", "None")
        init_entry_trailing_stop_type_long_low = st.selectbox("Type", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val) if default_val in ["None", "ATR", "Indicator", "Both"] else 0, key="init_ts_ll")
        init_entry_trailing_stop_atr_multiplier_long_low = st.number_input("ATR Multiplier", value=get_default("init_entry_trailing_stop_atr_multiplier_long_low", 1.0), step=0.1, format=fmt, key="init_ts_ll_atr")
        c_ind_ll = st.number_input("Indicator Count", value=get_default("init_entry_trailing_stop_indicator_count_long_low", 0), step=1, key="init_ts_ll_indcount")
        init_entry_trailing_stop_indicator_long_low = []
        for i in range(int(c_ind_ll)):
            ind_ = st.selectbox(f"Indicator {i+1}", filtered_indicators, key=f"init_ts_ll_ind_{i}")
            init_entry_trailing_stop_indicator_long_low.append(ind_)
        init_entry_trailing_stop_allow_reentry_long_low = st.checkbox("Allow Reentry After TS Exit?", value=get_default("init_entry_trailing_stop_allow_reentry_long_low", True), key="init_ts_ll_allow")

    with st.expander("Short, High"):
        default_val = get_default("init_entry_trailing_stop_type_short_high", "None")
        init_entry_trailing_stop_type_short_high = st.selectbox("Type", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val) if default_val in ["None", "ATR", "Indicator", "Both"] else 0, key="init_ts_sh")
        init_entry_trailing_stop_atr_multiplier_short_high = st.number_input("ATR Multiplier", value=get_default("init_entry_trailing_stop_atr_multiplier_short_high", 1.0), step=0.1, format=fmt, key="init_ts_sh_atr")
        c_ind_sh = st.number_input("Indicator Count", value=get_default("init_entry_trailing_stop_indicator_count_short_high", 0), step=1, key="init_ts_sh_indcount")
        init_entry_trailing_stop_indicator_short_high = []
        for i in range(int(c_ind_sh)):
            ind_ = st.selectbox(f"Indicator {i+1}", filtered_indicators, key=f"init_ts_sh_ind_{i}")
            init_entry_trailing_stop_indicator_short_high.append(ind_)
        init_entry_trailing_stop_allow_reentry_short_high = st.checkbox("Allow Reentry After TS Exit?", value=get_default("init_entry_trailing_stop_allow_reentry_short_high", True), key="init_ts_sh_allow")

    with st.expander("Short, Low"):
        default_val = get_default("init_entry_trailing_stop_type_short_low", "None")
        init_entry_trailing_stop_type_short_low = st.selectbox("Type", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val) if default_val in ["None", "ATR", "Indicator", "Both"] else 0, key="init_ts_sl")
        init_entry_trailing_stop_atr_multiplier_short_low = st.number_input("ATR Multiplier", value=get_default("init_entry_trailing_stop_atr_multiplier_short_low", 1.0), step=0.1, format=fmt, key="init_ts_sl_atr")
        c_ind_sl = st.number_input("Indicator Count", value=get_default("init_entry_trailing_stop_indicator_count_short_low", 0), step=1, key="init_ts_sl_indcount")
        init_entry_trailing_stop_indicator_short_low = []
        for i in range(int(c_ind_sl)):
            ind_ = st.selectbox(f"Indicator {i+1}", filtered_indicators, key=f"init_ts_sl_ind_{i}")
            init_entry_trailing_stop_indicator_short_low.append(ind_)
        init_entry_trailing_stop_allow_reentry_short_low = st.checkbox("Allow Reentry After TS Exit?", value=get_default("init_entry_trailing_stop_allow_reentry_short_low", True), key="init_ts_sl_allow")

    st.markdown("---")
    st.subheader("Reentry Trailing Stops")
    reentry_entry_trailing_stop_type_long_high = []
    reentry_entry_trailing_stop_atr_multiplier_long_high = []
    reentry_entry_trailing_stop_indicator_long_high = []
    reentry_entry_trailing_stop_allow_reentry_long_high = []

    reentry_entry_trailing_stop_type_long_low = []
    reentry_entry_trailing_stop_atr_multiplier_long_low = []
    reentry_entry_trailing_stop_indicator_long_low = []
    reentry_entry_trailing_stop_allow_reentry_long_low = []

    reentry_entry_trailing_stop_type_short_high = []
    reentry_entry_trailing_stop_atr_multiplier_short_high = []
    reentry_entry_trailing_stop_indicator_short_high = []
    reentry_entry_trailing_stop_allow_reentry_short_high = []

    reentry_entry_trailing_stop_type_short_low = []
    reentry_entry_trailing_stop_atr_multiplier_short_low = []
    reentry_entry_trailing_stop_indicator_short_low = []
    reentry_entry_trailing_stop_allow_reentry_short_low = []

    for idx in range(int(max_reentries)):
        st.markdown(f"**Reentry #{idx+1}**")

        with st.expander(f"Long, High Reentry #{idx+1}", expanded=False):
            default_val = get_default("reentry_entry_trailing_stop_type_long_high", [])
            tstype = st.selectbox("Type", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val[idx]) if len(default_val) > idx and default_val[idx] in ["None", "ATR", "Indicator", "Both"] else 0, key=f"re_ts_lh_type_{idx}")
            default_atr = get_default("reentry_entry_trailing_stop_atr_multiplier_long_high", [])
            atrv = st.number_input("ATR Multiplier", value=default_atr[idx] if len(default_atr) > idx else 1.0, step=0.1, format=fmt, key=f"re_ts_lh_atr_{idx}")
            cind = st.number_input("Indicator Count", value=get_default("reentry_entry_trailing_stop_indicator_count_long_high", 0), step=1, key=f"re_ts_lh_indcount_{idx}")
            inds_ = []
            for i2 in range(int(cind)):
                ind_ = st.selectbox(f"Indicator {i2+1}", filtered_indicators, key=f"re_ts_lh_ind_{idx}_{i2}")
                inds_.append(ind_)
            default_allow = get_default("reentry_entry_trailing_stop_allow_reentry_long_high", [])
            allow_ = st.checkbox("Allow Reentry After TS Exit?", value=default_allow[idx] if len(default_allow) > idx else True, key=f"re_ts_lh_allow_{idx}")

            reentry_entry_trailing_stop_type_long_high.append(tstype)
            reentry_entry_trailing_stop_atr_multiplier_long_high.append(atrv)
            reentry_entry_trailing_stop_indicator_long_high.append(inds_)
            reentry_entry_trailing_stop_allow_reentry_long_high.append(allow_)

        with st.expander(f"Long, Low Reentry #{idx+1}", expanded=False):
            default_val = get_default("reentry_entry_trailing_stop_type_long_low", [])
            tstype = st.selectbox("Type", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val[idx]) if len(default_val) > idx and default_val[idx] in ["None", "ATR", "Indicator", "Both"] else 0, key=f"re_ts_ll_type_{idx}")
            default_atr = get_default("reentry_entry_trailing_stop_atr_multiplier_long_low", [])
            atrv = st.number_input("ATR Multiplier", value=default_atr[idx] if len(default_atr) > idx else 1.0, step=0.1, format=fmt, key=f"re_ts_ll_atr_{idx}")
            cind = st.number_input("Indicator Count", value=get_default("reentry_entry_trailing_stop_indicator_count_long_low", 0), step=1, key=f"re_ts_ll_indcount_{idx}")
            inds_ = []
            for i2 in range(int(cind)):
                ind_ = st.selectbox(f"Indicator {i2+1}", filtered_indicators, key=f"re_ts_ll_ind_{idx}_{i2}")
                inds_.append(ind_)
            default_allow = get_default("reentry_entry_trailing_stop_allow_reentry_long_low", [])
            allow_ = st.checkbox("Allow Reentry After TS Exit?", value=default_allow[idx] if len(default_allow) > idx else True, key=f"re_ts_ll_allow_{idx}")

            reentry_entry_trailing_stop_type_long_low.append(tstype)
            reentry_entry_trailing_stop_atr_multiplier_long_low.append(atrv)
            reentry_entry_trailing_stop_indicator_long_low.append(inds_)
            reentry_entry_trailing_stop_allow_reentry_long_low.append(allow_)

        with st.expander(f"Short, High Reentry #{idx+1}", expanded=False):
            default_val = get_default("reentry_entry_trailing_stop_type_short_high", [])
            tstype = st.selectbox("Type", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val[idx]) if len(default_val) > idx and default_val[idx] in ["None", "ATR", "Indicator", "Both"] else 0, key=f"re_ts_sh_type_{idx}")
            default_atr = get_default("reentry_entry_trailing_stop_atr_multiplier_short_high", [])
            atrv = st.number_input("ATR Multiplier", value=default_atr[idx] if len(default_atr) > idx else 1.0, step=0.1, format=fmt, key=f"re_ts_sh_atr_{idx}")
            cind = st.number_input("Indicator Count", value=get_default("reentry_entry_trailing_stop_indicator_count_short_high", 0), step=1, key=f"re_ts_sh_indcount_{idx}")
            inds_ = []
            for i2 in range(int(cind)):
                ind_ = st.selectbox(f"Indicator {i2+1}", filtered_indicators, key=f"re_ts_sh_ind_{idx}_{i2}")
                inds_.append(ind_)
            default_allow = get_default("reentry_entry_trailing_stop_allow_reentry_short_high", [])
            allow_ = st.checkbox("Allow Reentry After TS Exit?", value=default_allow[idx] if len(default_allow) > idx else True, key=f"re_ts_sh_allow_{idx}")

            reentry_entry_trailing_stop_type_short_high.append(tstype)
            reentry_entry_trailing_stop_atr_multiplier_short_high.append(atrv)
            reentry_entry_trailing_stop_indicator_short_high.append(inds_)
            reentry_entry_trailing_stop_allow_reentry_short_high.append(allow_)

        with st.expander(f"Short, Low Reentry #{idx+1}", expanded=False):
            default_val = get_default("reentry_entry_trailing_stop_type_short_low", [])
            tstype = st.selectbox("Type", ["None", "ATR", "Indicator", "Both"], index=["None", "ATR", "Indicator", "Both"].index(default_val[idx]) if len(default_val) > idx and default_val[idx] in ["None", "ATR", "Indicator", "Both"] else 0, key=f"re_ts_sl_type_{idx}")
            default_atr = get_default("reentry_entry_trailing_stop_atr_multiplier_short_low", [])
            atrv = st.number_input("ATR Multiplier", value=default_atr[idx] if len(default_atr) > idx else 1.0, step=0.1, format=fmt, key=f"re_ts_sl_atr_{idx}")
            cind = st.number_input("Indicator Count", value=get_default("reentry_entry_trailing_stop_indicator_count_short_low", 0), step=1, key=f"re_ts_sl_indcount_{idx}")
            inds_ = []
            for i2 in range(int(cind)):
                ind_ = st.selectbox(f"Indicator {i2+1}", filtered_indicators, key=f"re_ts_sl_ind_{idx}_{i2}")
                inds_.append(ind_)
            default_allow = get_default("reentry_entry_trailing_stop_allow_reentry_short_low", [])
            allow_ = st.checkbox("Allow Reentry After TS Exit?", value=default_allow[idx] if len(default_allow) > idx else True, key=f"re_ts_sl_allow_{idx}")

            reentry_entry_trailing_stop_type_short_low.append(tstype)
            reentry_entry_trailing_stop_atr_multiplier_short_low.append(atrv)
            reentry_entry_trailing_stop_indicator_short_low.append(inds_)
            reentry_entry_trailing_stop_allow_reentry_short_low.append(allow_)

# 8) Preferred Trend Start
with setup_tabs[7]:
    st.subheader("Preferred Trend Start")
    colA, colB = st.columns(2)
    with colA:
        preferred_high = st.selectbox("Preferred (High Volatility)", ["Long", "Short"], index=["Long", "Short"].index(get_default("preferred_high", "Long")), key="pref_high")
    with colB:
        preferred_low = st.selectbox("Preferred (Low Volatility)", ["Long", "Short"], index=["Long", "Short"].index(get_default("preferred_low", "Long")), key="pref_low")

# 9) Target vs Stop Preference
with setup_tabs[8]:
    st.subheader("If both target & stop are hit same candle, which has priority?")
    target_stop_preference = st.selectbox("Priority:", ["stop", "target"], index=["stop", "target"].index(get_default("target_stop_preference", "stop")), key="tspref")

# 10) Export & Simulation
with setup_tabs[9]:
    st.subheader("Export Setup & Simulation Buttons")

    def gather_params():
        """
        Gathers all UI inputs into a dictionary for apply_trade_setup_df or variant.
        """
        final_p = {}
        # Basic settings
        final_p["tick_size"] = tick_size
        final_p["tick_value"] = tick_value
        final_p["preferred_high"] = preferred_high
        final_p["preferred_low"] = preferred_low
        final_p["target_stop_preference"] = target_stop_preference
        final_p["max_reentries"] = int(max_reentries)
        final_p["initial-entry-type"] = initial_entry_type
    
        # Volatility Zone settings (new keys)
        final_p["HighVolInd"] = st.session_state.get("HighVolInd", "(Range_360_Range_120)_ATR_20")
        final_p["HighVolOp"]  = st.session_state.get("HighVolOp", "<")
        final_p["HighVolThr"] = st.session_state.get("HighVolThr", 10.0)
        final_p["LowVolInd"]  = st.session_state.get("LowVolInd", "(Range_360_Range_120)_ATR_20")
        final_p["LowVolOp"]   = st.session_state.get("LowVolOp", ">=")
        final_p["LowVolThr"]  = st.session_state.get("LowVolThr", 10.0)
    
        # Define booleans for target_defined, stop_defined
        final_p["target_defined"] = (target_defined == "Yes")
        final_p["stop_defined"] = (stop_defined == "Yes")
    
        # Condition groups (start/continue/reentry)
        final_p["long_high_start"] = long_high_start
        final_p["long_high_continue"] = long_high_continue
        final_p["long_low_start"] = long_low_start
        final_p["long_low_continue"] = long_low_continue
    
        final_p["short_high_start"] = short_high_start
        final_p["short_high_continue"] = short_high_continue
        final_p["short_low_start"] = short_low_start
        final_p["short_low_continue"] = short_low_continue
    
        final_p["long_high_reentry"] = long_high_reentry
        final_p["long_low_reentry"] = long_low_reentry
        final_p["short_high_reentry"] = short_high_reentry
        final_p["short_low_reentry"] = short_low_reentry
    
        # Trade sizing
        final_p["init_long_high_size"] = init_long_high_size
        final_p["init_long_low_size"] = init_long_low_size
        final_p["init_short_high_size"] = init_short_high_size
        final_p["init_short_low_size"] = init_short_low_size
    
        final_p["reentry_long_high_sizes"] = reentry_long_high_sizes
        final_p["reentry_long_low_sizes"] = reentry_long_low_sizes
        final_p["reentry_short_high_sizes"] = reentry_short_high_sizes
        final_p["reentry_short_low_sizes"] = reentry_short_low_sizes
    
        # Initial Target
        final_p["init_target_method_long_high"] = init_target_method_long_high
        final_p["init_target_long_high"] = init_target_long_high
        final_p["init_target_allow_reentry_long_high"] = init_target_allow_reentry_long_high
    
        final_p["init_target_method_long_low"] = init_target_method_long_low
        final_p["init_target_long_low"] = init_target_long_low
        final_p["init_target_allow_reentry_long_low"] = init_target_allow_reentry_long_low
    
        final_p["init_target_method_short_high"] = init_target_method_short_high
        final_p["init_target_short_high"] = init_target_short_high
        final_p["init_target_allow_reentry_short_high"] = init_target_allow_reentry_short_high
    
        final_p["init_target_method_short_low"] = init_target_method_short_low
        final_p["init_target_short_low"] = init_target_short_low
        final_p["init_target_allow_reentry_short_low"] = init_target_allow_reentry_short_low
    
        # Reentry Target
        final_p["reentry_target_method_long_high"] = reentry_target_method_long_high
        final_p["reentry_target_long_high"] = reentry_target_long_high
        final_p["reentry_target_allow_reentry_long_high"] = reentry_target_allow_reentry_long_high
    
        final_p["reentry_target_method_long_low"] = reentry_target_method_long_low
        final_p["reentry_target_long_low"] = reentry_target_long_low
        final_p["reentry_target_allow_reentry_long_low"] = reentry_target_allow_reentry_long_low
    
        final_p["reentry_target_method_short_high"] = reentry_target_method_short_high
        final_p["reentry_target_short_high"] = reentry_target_short_high
        final_p["reentry_target_allow_reentry_short_high"] = reentry_target_allow_reentry_short_high
    
        final_p["reentry_target_method_short_low"] = reentry_target_method_short_low
        final_p["reentry_target_short_low"] = reentry_target_short_low
        final_p["reentry_target_allow_reentry_short_low"] = reentry_target_allow_reentry_short_low
    
        # Initial Stop
        final_p["init_stop_method_long_high"] = init_stop_method_long_high
        final_p["init_stop_long_high"] = init_stop_long_high
        final_p["init_stop_allow_reentry_long_high"] = init_stop_allow_reentry_long_high
    
        final_p["init_stop_method_long_low"] = init_stop_method_long_low
        final_p["init_stop_long_low"] = init_stop_long_low
        final_p["init_stop_allow_reentry_long_low"] = init_stop_allow_reentry_long_low
    
        final_p["init_stop_method_short_high"] = init_stop_method_short_high
        final_p["init_stop_short_high"] = init_stop_short_high
        final_p["init_stop_allow_reentry_short_high"] = init_stop_allow_reentry_short_high
    
        final_p["init_stop_method_short_low"] = init_stop_method_short_low
        final_p["init_stop_short_low"] = init_stop_short_low
        final_p["init_stop_allow_reentry_short_low"] = init_stop_allow_reentry_short_low
    
        # Reentry Stop
        final_p["reentry_stop_method_long_high"] = reentry_stop_method_long_high
        final_p["reentry_stop_long_high"] = reentry_stop_long_high
        final_p["reentry_stop_allow_reentry_long_high"] = reentry_stop_allow_reentry_long_high
    
        final_p["reentry_stop_method_long_low"] = reentry_stop_method_long_low
        final_p["reentry_stop_long_low"] = reentry_stop_long_low
        final_p["reentry_stop_allow_reentry_long_low"] = reentry_stop_allow_reentry_long_low
    
        final_p["reentry_stop_method_short_high"] = reentry_stop_method_short_high
        final_p["reentry_stop_short_high"] = reentry_stop_short_high
        final_p["reentry_stop_allow_reentry_short_high"] = reentry_stop_allow_reentry_short_high
    
        final_p["reentry_stop_method_short_low"] = reentry_stop_method_short_low
        final_p["reentry_stop_short_low"] = reentry_stop_short_low
        final_p["reentry_stop_allow_reentry_short_low"] = reentry_stop_allow_reentry_short_low
    
        # Global Trailing Stops
        final_p["global_trailing_stop_type_long_high"] = global_trailing_stop_type_long_high
        final_p["global_trailing_stop_atr_multiplier_long_high"] = global_trailing_stop_atr_multiplier_long_high
        final_p["global_trailing_stop_indicator_long_high"] = global_trailing_stop_indicator_long_high
    
        final_p["global_trailing_stop_type_long_low"] = global_trailing_stop_type_long_low
        final_p["global_trailing_stop_atr_multiplier_long_low"] = global_trailing_stop_atr_multiplier_long_low
        final_p["global_trailing_stop_indicator_long_low"] = global_trailing_stop_indicator_long_low
    
        final_p["global_trailing_stop_type_short_high"] = global_trailing_stop_type_short_high
        final_p["global_trailing_stop_atr_multiplier_short_high"] = global_trailing_stop_atr_multiplier_short_high
        final_p["global_trailing_stop_indicator_short_high"] = global_trailing_stop_indicator_short_high
    
        final_p["global_trailing_stop_type_short_low"] = global_trailing_stop_type_short_low
        final_p["global_trailing_stop_atr_multiplier_short_low"] = global_trailing_stop_atr_multiplier_short_low
        final_p["global_trailing_stop_indicator_short_low"] = global_trailing_stop_indicator_short_low
    
        # Initial Entry-Level Trailing Stops
        final_p["init_entry_trailing_stop_type_long_high"] = init_entry_trailing_stop_type_long_high
        final_p["init_entry_trailing_stop_atr_multiplier_long_high"] = init_entry_trailing_stop_atr_multiplier_long_high
        final_p["init_entry_trailing_stop_indicator_long_high"] = init_entry_trailing_stop_indicator_long_high
        final_p["init_entry_trailing_stop_allow_reentry_long_high"] = init_entry_trailing_stop_allow_reentry_long_high
    
        final_p["init_entry_trailing_stop_type_long_low"] = init_entry_trailing_stop_type_long_low
        final_p["init_entry_trailing_stop_atr_multiplier_long_low"] = init_entry_trailing_stop_atr_multiplier_long_low
        final_p["init_entry_trailing_stop_indicator_long_low"] = init_entry_trailing_stop_indicator_long_low
        final_p["init_entry_trailing_stop_allow_reentry_long_low"] = init_entry_trailing_stop_allow_reentry_long_low
    
        final_p["init_entry_trailing_stop_type_short_high"] = init_entry_trailing_stop_type_short_high
        final_p["init_entry_trailing_stop_atr_multiplier_short_high"] = init_entry_trailing_stop_atr_multiplier_short_high
        final_p["init_entry_trailing_stop_indicator_short_high"] = init_entry_trailing_stop_indicator_short_high
        final_p["init_entry_trailing_stop_allow_reentry_short_high"] = init_entry_trailing_stop_allow_reentry_short_high
    
        final_p["init_entry_trailing_stop_type_short_low"] = init_entry_trailing_stop_type_short_low
        final_p["init_entry_trailing_stop_atr_multiplier_short_low"] = init_entry_trailing_stop_atr_multiplier_short_low
        final_p["init_entry_trailing_stop_indicator_short_low"] = init_entry_trailing_stop_indicator_short_low
        final_p["init_entry_trailing_stop_allow_reentry_short_low"] = init_entry_trailing_stop_allow_reentry_short_low
    
        # Reentry Entry-Level Trailing Stops
        final_p["reentry_entry_trailing_stop_type_long_high"] = reentry_entry_trailing_stop_type_long_high
        final_p["reentry_entry_trailing_stop_atr_multiplier_long_high"] = reentry_entry_trailing_stop_atr_multiplier_long_high
        final_p["reentry_entry_trailing_stop_indicator_long_high"] = reentry_entry_trailing_stop_indicator_long_high
        final_p["reentry_entry_trailing_stop_allow_reentry_long_high"] = reentry_entry_trailing_stop_allow_reentry_long_high
    
        final_p["reentry_entry_trailing_stop_type_long_low"] = reentry_entry_trailing_stop_type_long_low
        final_p["reentry_entry_trailing_stop_atr_multiplier_long_low"] = reentry_entry_trailing_stop_atr_multiplier_long_low
        final_p["reentry_entry_trailing_stop_indicator_long_low"] = reentry_entry_trailing_stop_indicator_long_low
        final_p["reentry_entry_trailing_stop_allow_reentry_long_low"] = reentry_entry_trailing_stop_allow_reentry_long_low
    
        final_p["reentry_entry_trailing_stop_type_short_high"] = reentry_entry_trailing_stop_type_short_high
        final_p["reentry_entry_trailing_stop_atr_multiplier_short_high"] = reentry_entry_trailing_stop_atr_multiplier_short_high
        final_p["reentry_entry_trailing_stop_indicator_short_high"] = reentry_entry_trailing_stop_indicator_short_high
        final_p["reentry_entry_trailing_stop_allow_reentry_short_high"] = reentry_entry_trailing_stop_allow_reentry_short_high
    
        final_p["reentry_entry_trailing_stop_type_short_low"] = reentry_entry_trailing_stop_type_short_low
        final_p["reentry_entry_trailing_stop_atr_multiplier_short_low"] = reentry_entry_trailing_stop_atr_multiplier_short_low
        final_p["reentry_entry_trailing_stop_indicator_short_low"] = reentry_entry_trailing_stop_indicator_short_low
        final_p["reentry_entry_trailing_stop_allow_reentry_short_low"] = reentry_entry_trailing_stop_allow_reentry_short_low
    
        return final_p

    if st.button("Export Setup"):
        if df.empty:
            st.warning("No data loaded.")
        else:
            p_dict = gather_params()
            json_str = json.dumps(p_dict, indent=2)
            st.download_button("Download Setup JSON", json_str, file_name="trade_setup_config.json", mime="application/json")

    if st.button("Apply Trade Setup (Normal)"):
        if df.empty:
            st.warning("No data loaded.")
        else:
            params_ = gather_params()
            updated_df, trade_events = apply_trade_setup_df(df.copy(), params_)
            norm_trades = get_normalized_trades(trade_events, tick_size, tick_value)

            os.makedirs("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/normal/", exist_ok=True)
            prefix_ = normal_output_prefix.strip()
            final_df_file = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/normal/", f"{prefix_}_final_df.csv")
            trade_list_file = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/normal/", f"{prefix_}_trade_list.csv")

            updated_df.to_csv(final_df_file, index=False)
            norm_trades.to_csv(trade_list_file, index=False)

            st.success(f"Normal Simulation completed! Files saved:\n{final_df_file}\n{trade_list_file}")

    if st.button("Run Variant Simulation"):
        if df.empty:
            st.warning("No data loaded.")
        else:
            base_params = gather_params()
            os.makedirs("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/variant/", exist_ok=True)
            stop_multipliers = [1, 1.5, 2]
            target_multipliers = [1, 2, 3]
            suffix_ = variant_output_suffix.strip()
            for sm in stop_multipliers:
                for tm in target_multipliers:
                    key_ = f"Stop{sm}_Target{tm}_{suffix_}"
                    var_df, var_events = simulate_variant(df.copy(), base_params, sm, tm)
                    norm_var = get_normalized_trades(var_events, tick_size, tick_value)

                    out_df = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/variant/", f"{key_}_final_df.csv")
                    out_tr = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/variant/", f"{key_}_trade_list.csv")
                    var_df.to_csv(out_df, index=False)
                    norm_var.to_csv(out_tr, index=False)

            st.success("Variant simulations done! Check 'C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/variant/' folder.")

# 11) MA Combo Simulation Tab
with setup_tabs[10]:
    st.subheader("MA Combo Simulation")
    st.markdown(
        """
        This simulation uses a moving average (MA) combination condition for both entry and continuation.
    
        **Initial Entry Condition:**  
        - For Long trades: The selected MAs (ordered from smallest to largest period) must be in descending order 
          (i.e. the fastest MA is greater than the slower MA) **and** the Open price must be greater than the value 
          of the fastest MA.
        - For Short trades: The selected MAs must be in ascending order 
          **and** the Open price must be less than the value of the fastest MA.
    
        **Continuation Condition:**  
        - Only the MA order condition is checked.
    
        No stops, targets, or reentries are used.
        """
    )
    
    def extract_period(ma_name):
        try:
            return int(ma_name.split('_')[1])
        except Exception:
            return float('inf')
    
    default_ma_selected = [
        "SMA_3", "EMA_3", "SMA_5", "SMA_50", "SMA_30", "SMA_26", "SMA_20", 
        "SMA_15", "SMA_12", "SMA_10", "SMA_100", "EMA_5", "EMA_50", "EMA_30", 
        "EMA_26", "EMA_20", "EMA_15", "EMA_12", "EMA_10", "EMA_100"
    ]
    
    if not df.empty:
        ma_options = [col for col in df.columns if ("SMA" in col.upper() or "EMA" in col.upper())]
        sorted_ma_options = sorted(ma_options, key=extract_period)
    else:
        sorted_ma_options = []
    
    st.markdown("### Select MA Columns")
    selected_ma = st.multiselect(
        "Select MA Columns (only columns containing SMA or EMA)", 
        options=sorted_ma_options, 
        default=get_default("ma_selected", default_ma_selected), 
        key="ma_selected"
    )
    
    st.markdown("### Combination Length Option")
    combo_length_option = st.radio("Select combination length", ["2 MA", "3 MA", "Both"], index=["2 MA", "3 MA", "Both"].index(get_default("ma_combo_option", "2 MA")), key="ma_combo_option")
    
    st.markdown("### Simulation Mode")
    sim_mode = st.radio("Simulation Mode", ["All possible combinations", "Manual selection"], index=["All possible combinations", "Manual selection"].index(get_default("ma_sim_mode", "All possible combinations")), key="ma_sim_mode")
    
    manual_combos = []
    if sim_mode == "Manual selection":
        import itertools
        possible_combos = []
        if combo_length_option == "2 MA":
            if selected_ma:
                possible_combos = list(itertools.combinations(selected_ma, 2))
        elif combo_length_option == "3 MA":
            if selected_ma:
                possible_combos = list(itertools.combinations(selected_ma, 3))
        else:
            if selected_ma:
                possible_combos = list(itertools.combinations(selected_ma, 2)) + list(itertools.combinations(selected_ma, 3))
        possible_combos = [tuple(sorted(combo, key=extract_period)) for combo in possible_combos]
        combo_strs = [", ".join(combo) for combo in possible_combos]
        
        default_manual_combo_strs = [
            "SMA_5, SMA_20",
            "SMA_5, SMA_10",
            "SMA_5, SMA_15",
            "SMA_5, EMA_5",
            "SMA_10, SMA_20",
            "EMA_5, EMA_10",
            "EMA_5, EMA_15",
            "EMA_5, EMA_20",
            "EMA_10, EMA_20",
            "SMA_12, SMA_26",
            "SMA_15, SMA_20",
            "SMA_10, SMA_30",
            "SMA_10, SMA_50",
            "EMA_12, EMA_26",
            "EMA_15, EMA_20",
            "EMA_10, EMA_30",
            "EMA_10, EMA_50",
            "SMA_20, SMA_50",
            "SMA_20, SMA_100",
            "SMA_50, SMA_100",
            "EMA_20, EMA_50",
            "EMA_20, EMA_100",
            "EMA_50, EMA_100",
            "SMA_10, EMA_10",
            "SMA_3, EMA_3"
        ]
        
        selected_combo_strs = st.multiselect(
            "Select one or more MA Combinations", 
            options=combo_strs, 
            default=get_default("manual_combo_select", default_manual_combo_strs), 
            key="manual_combo_select"
        )
        for combo_str in selected_combo_strs:
            manual_combos.append([x.strip() for x in combo_str.split(",")])
    
    st.markdown("### Preferred Side for Simulation")
    preferred_high = st.selectbox("Preferred side (High Volatility)", ["Long", "Short"], index=["Long", "Short"].index(get_default("ma_pref_high", "Long")), key="ma_pref_high")
    preferred_low = st.selectbox("Preferred side (Low Volatility)", ["Long", "Short"], index=["Long", "Short"].index(get_default("ma_pref_low", "Long")), key="ma_pref_low")
    
    st.markdown("### Fixed Lot Sizes for MA Simulation")
    colA, colB = st.columns(2)
    with colA:
        ma_long_high = st.number_input("Long, High Lot Size", value=get_default("ma_long_high", 1.0), step=1.0, format=fmt, key="ma_long_high")
        ma_long_low = st.number_input("Long, Low Lot Size", value=get_default("ma_long_low", 2.0), step=1.0, format=fmt, key="ma_long_low")
    with colB:
        ma_short_high = st.number_input("Short, High Lot Size", value=get_default("ma_short_high", 3.0), step=1.0, format=fmt, key="ma_short_high")
        ma_short_low = st.number_input("Short, Low Lot Size", value=get_default("ma_short_low", 4.0), step=1.0, format=fmt, key="ma_short_low")
    fixed_lots = {
        "Long_High": ma_long_high,
        "Long_Low": ma_long_low,
        "Short_High": ma_short_high,
        "Short_Low": ma_short_low
    }
    
    st.markdown("### Output Options")
    output_type = st.selectbox("Select Output Type", ["trade_list", "final_df", "both"], index=["trade_list", "final_df", "both"].index(get_default("ma_output_type", "trade_list")), key="ma_output_type")
    
    st.markdown("### File Prefix and Naming")
    sim_prefix = st.text_input("Simulation File Prefix", value=get_default("ma_prefix", "maSim"), key="ma_prefix")
    
    if st.button("Run MA Combo Simulation"):
        if df.empty:
            st.warning("No data loaded.")
        elif not selected_ma:
            st.warning("Please select at least one MA column.")
        else:
            import itertools, os
            common_params = {
                "fixed_lots": fixed_lots,
                "output_type": output_type,
                "tick_size": tick_size,
                "tick_value": tick_value,
                "preferred_high": preferred_high,
                "preferred_low": preferred_low
            }
            results_folder = "C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/ma_combo/"
            os.makedirs(results_folder, exist_ok=True)
            
            if sim_mode == "All possible combinations":
                if combo_length_option == "2 MA":
                    all_combos = list(itertools.combinations(selected_ma, 2))
                elif combo_length_option == "3 MA":
                    all_combos = list(itertools.combinations(selected_ma, 3))
                else:
                    all_combos = list(itertools.combinations(selected_ma, 2)) + list(itertools.combinations(selected_ma, 3))
                all_combos = [tuple(sorted(combo, key=extract_period)) for combo in all_combos]
                count = 0
                for combo in all_combos:
                    current_params = common_params.copy()
                    current_params["manual_combos"] = [list(combo)]
                    current_prefix = f"{sim_prefix}_{'-'.join(combo)}"
                    final_df, norm_trades = simulate_ma_combo(df.copy(), current_params)
                    if output_type in ["final_df", "both"]:
                        out_final = os.path.join(results_folder, f"{current_prefix}_final_df.csv")
                        final_df.to_csv(out_final, index=False)
                    if output_type in ["trade_list", "both"]:
                        out_trade = os.path.join(results_folder, f"{current_prefix}_trade_list.csv")
                        norm_trades.to_csv(out_trade, index=False)
                    count += 1
                st.success(f"MA Combo Simulation completed for {count} combinations!\nFiles are saved in '{results_folder}' with individual prefixes starting with '{sim_prefix}_'.")
            else:
                count = 0
                for combo in manual_combos:
                    current_params = common_params.copy()
                    current_params["manual_combos"] = [combo]
                    current_prefix = f"{sim_prefix}_{'-'.join(combo)}"
                    final_df, norm_trades = simulate_ma_combo(df.copy(), current_params)
                    if output_type in ["final_df", "both"]:
                        out_final = os.path.join(results_folder, f"{current_prefix}_final_df.csv")
                        final_df.to_csv(out_final, index=False)
                    if output_type in ["trade_list", "both"]:
                        out_trade = os.path.join(results_folder, f"{current_prefix}_trade_list.csv")
                        norm_trades.to_csv(out_trade, index=False)
                    count += 1
                st.success(f"MA Combo Simulation completed for {count} manually selected combinations!\nFiles are saved in '{results_folder}' with individual prefixes starting with '{sim_prefix}_'.")

# 12) Indicator Percentile Simulation Tab
with setup_tabs[11]:
    st.subheader("Indicator Percentile Simulation")
    st.markdown(
        """
        This simulation uses a single MA combination along with an extra indicator threshold condition.
        
        **Initial Entry Condition:**  
        - MA Condition: For Long trades, the selected MAs (ordered from smallest to largest period)
          must be in descending order (i.e. fastest MA > slower MA); for Short trades, in ascending order.
        - Indicator Condition: For the selected indicator, the row value must be 
          greater than (or less than) a specified percentile threshold.
        - Optional Open Condition: You may include the condition that the Open price must be 
          greater than (or less than) the fastest MA’s value.
          
        **Continuation Condition:**  
        - Only the MA condition is checked.
        
        No stops, targets, reentries, or trailing stops are used.
        """
    )
    
    def extract_period(ma_name):
        try:
            return int(ma_name.split('_')[1])
        except Exception:
            return float('inf')
    
    if not df.empty:
        ma_options = [col for col in df.columns if ("SMA" in col.upper() or "EMA" in col.upper())]
        sorted_ma_options = sorted(ma_options, key=extract_period)
    else:
        sorted_ma_options = []

    uploaded_settings = st.file_uploader(
        "Upload Indicator Percentile JSON Settings",
        type=["json"],
        key="indperc_json_uploader",
    )
    if uploaded_settings is not None:
        try:
            parsed_settings = json.loads(uploaded_settings.read().decode("utf-8"))
            if "json_settings" not in st.session_state:
                st.session_state["json_settings"] = {}
            st.session_state["json_settings"].update(parsed_settings)
        except Exception as e:
            st.error(f"Error loading JSON settings: {e}")

    st.markdown("### Select a Single MA Combination")
    import itertools
    possible_combos = list(itertools.combinations(selected_ma, 2)) + list(itertools.combinations(selected_ma, 3))
    
    possible_combos = [tuple(sorted(combo, key=extract_period)) for combo in possible_combos]
    combo_strs = [", ".join(combo) for combo in possible_combos]
    selected_combo_str = st.selectbox("Select one MA Combination", options=combo_strs, key="indperc_ma_combo")
    manual_ma_combo = [[x.strip() for x in selected_combo_str.split(",")]]
    
    st.markdown("### Select Indicator(s) for Extra Condition")
    all_columns = df.columns.tolist()
    selected_indicators = st.multiselect("Select one or more indicators", options=all_columns, default=get_default("indperc_indicator", []), key="indperc_indicator")
    if selected_indicators and not df.empty:
        percentile_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        percentile_df = df[selected_indicators].quantile([p/100 for p in percentile_levels])
        percentile_df.index = [f"{int(p)}th" for p in percentile_levels]
        st.markdown("#### Percentile Table for Selected Indicators")
        st.dataframe(percentile_df)
    else:
        percentile_df = None
    
    st.markdown("### Select Percentile Levels for Each Condition Direction")
    greater_percentiles = st.multiselect("Select percentile levels for 'Greater Than' condition", options=[f"{p}th" for p in percentile_levels], default=get_default("indperc_pct_gt", []), key="indperc_pct_gt")
    less_percentiles = st.multiselect("Select percentile levels for 'Less Than' condition", options=[f"{p}th" for p in percentile_levels], default=get_default("indperc_pct_lt", []), key="indperc_pct_lt")
    
    include_open = st.checkbox("Include Open Condition (e.g., Open > fastest MA for Long)", value=get_default("indperc_include_open", True), key="indperc_include_open")
    
    st.markdown("### Preferred Side and Fixed Lot Sizes")
    preferred_high = st.selectbox("Preferred side (High Volatility)", ["Long", "Short"], index=["Long", "Short"].index(get_default("indperc_pref_high", "Long")), key="indperc_pref_high")
    preferred_low = st.selectbox("Preferred side (Low Volatility)", ["Long", "Short"], index=["Long", "Short"].index(get_default("indperc_pref_low", "Long")), key="indperc_pref_low")
    colA, colB = st.columns(2)
    with colA:
        ind_long_high = st.number_input("Long, High Lot Size", value=get_default("indperc_long_high", 1.0), step=1.0, format=fmt, key="indperc_long_high")
        ind_long_low = st.number_input("Long, Low Lot Size", value=get_default("indperc_long_low", 2.0), step=1.0, format=fmt, key="indperc_long_low")
    with colB:
        ind_short_high = st.number_input("Short, High Lot Size", value=get_default("indperc_short_high", 3.0), step=1.0, format=fmt, key="indperc_short_high")
        ind_short_low = st.number_input("Short, Low Lot Size", value=get_default("indperc_short_low", 4.0), step=1.0, format=fmt, key="indperc_short_low")
    fixed_lots = {
        "Long_High": ind_long_high,
        "Long_Low": ind_long_low,
        "Short_High": ind_short_high,
        "Short_Low": ind_short_low
    }
    
    st.markdown("### Output Options and File Prefix")
    output_type = st.selectbox("Select Output Type", ["trade_list", "final_df", "both"], index=["trade_list", "final_df", "both"].index(get_default("indperc_output", "trade_list")), key="indperc_output")
    sim_prefix = st.text_input("Simulation File Prefix", value=get_default("indperc_prefix", "IndPercSim"), key="indperc_prefix")

    def gather_indperc_params():
        keys = [
            "indperc_ma_combo",
            "indperc_indicator",
            "indperc_pct_gt",
            "indperc_pct_lt",
            "indperc_include_open",
            "indperc_pref_high",
            "indperc_pref_low",
            "indperc_long_high",
            "indperc_long_low",
            "indperc_short_high",
            "indperc_short_low",
            "indperc_output",
            "indperc_prefix",
        ]
        return {k: st.session_state.get(k) for k in keys}

    if st.button("Export Indicator Percentile Settings"):
        params = gather_indperc_params()
        json_str = json.dumps(params, indent=2)
        st.download_button(
            "Download Indicator Percentile Settings",
            data=json_str,
            file_name="indicator_percentile_settings.json",
        )

    if st.button("Run Indicator Percentile Simulation"):
        if df.empty:
            st.warning("No data loaded.")
        elif not manual_ma_combo:
            st.warning("Please select a MA combination.")
        elif not selected_indicators:
            st.warning("Please select at least one indicator.")
        elif (not greater_percentiles) and (not less_percentiles):
            st.warning("Please select at least one percentile level for one of the conditions.")
        else:
            import os, itertools
            count = 0
            for indicator in selected_indicators:
                for perc in greater_percentiles:
                    if percentile_df is None or indicator not in percentile_df.columns:
                        continue
                    threshold = percentile_df.at[perc, indicator]
                    sim_params = {
                        "manual_combos": manual_ma_combo,
                        "indicator": indicator,
                        "threshold": threshold,
                        "direction": ">",
                        "include_open_condition": include_open,
                        "preferred_high": preferred_high,
                        "preferred_low": preferred_low,
                        "fixed_lots": fixed_lots,
                        "tick_size": tick_size,
                        "tick_value": tick_value,
                        "output_type": output_type
                    }
                    final_df, norm_trades = simulate_indicator_percentile_combo(df.copy(), sim_params)
                    current_prefix = f"{sim_prefix}_{selected_combo_str.replace(', ', '-')}_{indicator}_{perc}_gt"
                    os.makedirs("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/indicator_percentile/", exist_ok=True)
                    if output_type in ["final_df", "both"]:
                        out_final = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/indicator_percentile/", f"{current_prefix}_final_df.csv")
                        final_df.to_csv(out_final, index=False)
                    if output_type in ["trade_list", "both"]:
                        out_trade = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/indicator_percentile/", f"{current_prefix}_trade_list.csv")
                        norm_trades.to_csv(out_trade, index=False)
                    count += 1
                for perc in less_percentiles:
                    if percentile_df is None or indicator not in percentile_df.columns:
                        continue
                    threshold = percentile_df.at[perc, indicator]
                    sim_params = {
                        "manual_combos": manual_ma_combo,
                        "indicator": indicator,
                        "threshold": threshold,
                        "direction": "<",
                        "include_open_condition": include_open,
                        "preferred_high": preferred_high,
                        "preferred_low": preferred_low,
                        "fixed_lots": fixed_lots,
                        "tick_size": tick_size,
                        "tick_value": tick_value,
                        "output_type": output_type
                    }
                    final_df, norm_trades = simulate_indicator_percentile_combo(df.copy(), sim_params)
                    current_prefix = f"{sim_prefix}_{selected_combo_str.replace(', ', '-')}_{indicator}_{perc}_lt"
                    os.makedirs("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/indicator_percentile/", exist_ok=True)
                    if output_type in ["final_df", "both"]:
                        out_final = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/indicator_percentile/", f"{current_prefix}_final_df.csv")
                        final_df.to_csv(out_final, index=False)
                    if output_type in ["trade_list", "both"]:
                        out_trade = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/indicator_percentile/", f"{current_prefix}_trade_list.csv")
                        norm_trades.to_csv(out_trade, index=False)
                    count += 1
            st.success(f"Indicator Percentile Simulation completed for {count} variant(s)!\nFiles are saved in 'C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/indicator_percentile/' with file names reflecting the parameters.")
        
# 13) Support/Resistance Breakout Simulation Tab – Unified Condition Builder
with setup_tabs[12]:
    import streamlit as st
    import pandas as pd
    import json
    import os
    from itertools import product
    
    fmt = "%.8f"
    all_columns = df.columns.tolist()
    
    st.title("Unified Support/Resistance Simulation Setup")
    
    st.subheader("Initial Entry Conditions")
    col_long, col_short = st.columns(2)
    with col_long:
        st.markdown("**Long Entry Condition Groups**")
        num_long_groups = st.number_input("Number of Long Entry Groups", min_value=1, step=1, value=get_default("num_long_groups", 1), key="num_long_groups")
        long_entry_groups = []
        for g in range(int(num_long_groups)):
            st.markdown(f"--- **Long Group {g+1}** ---")
            group = get_condition_group(f"Long_Entry_Group_{g+1}", all_columns, default=get_default(f"Long_Entry_Group_{g+1}", None))
            long_entry_groups.append(group)
    with col_short:
        st.markdown("**Short Entry Condition Groups**")
        num_short_groups = st.number_input("Number of Short Entry Groups", min_value=1, step=1, value=get_default("num_short_groups", 1), key="num_short_groups")
        short_entry_groups = []
        for g in range(int(num_short_groups)):
            st.markdown(f"--- **Short Group {g+1}** ---")
            group = get_condition_group(f"Short_Entry_Group_{g+1}", all_columns, default=get_default(f"Short_Entry_Group_{g+1}", None))
            short_entry_groups.append(group)
    
    st.subheader("Continuation Conditions")
    col_long_cont, col_short_cont = st.columns(2)
    with col_long_cont:
        st.markdown("**Long Continuation Condition Groups**")
        num_long_cont_groups = st.number_input("Number of Long Continuation Groups", min_value=0, step=1, value=get_default("num_long_cont_groups", 0), key="num_long_cont")
        long_cont_groups = []
        for g in range(int(num_long_cont_groups)):
            st.markdown(f"--- **Long Continuation Group {g+1}** ---")
            group = get_condition_group(f"Long_Continue_Group_{g+1}", all_columns, default=get_default(f"Long_Continue_Group_{g+1}", None))
            long_cont_groups.append(group)
    with col_short_cont:
        st.markdown("**Short Continuation Condition Groups**")
        num_short_cont_groups = st.number_input("Number of Short Continuation Groups", min_value=0, step=1, value=get_default("num_short_cont_groups", 0), key="num_short_cont")
        short_cont_groups = []
        for g in range(int(num_short_cont_groups)):
            st.markdown(f"--- **Short Continuation Group {g+1}** ---")
            group = get_condition_group(f"Short_Continue_Group_{g+1}", all_columns, default=get_default(f"Short_Continue_Group_{g+1}", None))
            short_cont_groups.append(group)
    st.markdown("If a continuation condition group is not built for a side, the corresponding entry group will be used.")
    
    st.subheader("Target/Stop Variant Settings")
    target_stop_variant = st.checkbox("Enable Target/Stop Variant", value=get_default("ts_variant_ui", False), key="ts_variant_ui")
    if target_stop_variant:
        target_mult_input = st.text_input("Enter Target multiplier(s) (comma separated)", value=get_default("target_mult_ui", "1,2,3"), key="target_mult_ui")
        stop_mult_input = st.text_input("Enter Stop multiplier(s) (comma separated)", value=get_default("stop_mult_ui", "1,1.5,2"), key="stop_mult_ui")
        target_multipliers = [float(x.strip()) for x in target_mult_input.split(",") if x.strip()]
        stop_multipliers = [float(x.strip()) for x in stop_mult_input.split(",") if x.strip()]
        ts_preference = st.selectbox("Target/Stop Preference", ["stop", "target"], index=["stop", "target"].index(get_default("ts_pref_ui", "stop")), key="ts_pref_ui")
        ts_mode = st.selectbox("Target/Stop Mode", ["immediate_exit", "mark_exit_only"], index=["immediate_exit", "mark_exit_only"].index(get_default("ts_mode_ui", "immediate_exit")), key="ts_mode_ui")
    else:
        target_multipliers = []
        stop_multipliers = []
        ts_preference = None
        ts_mode = None
    
    st.subheader("Other Settings")
    col_pref = st.columns(2)
    with col_pref[0]:
        pref_high = st.selectbox("Preferred side (High Volatility)", ["Long", "Short"], index=["Long", "Short"].index(get_default("pref_high_ui", "Long")), key="pref_high_ui")
    with col_pref[1]:
        pref_low = st.selectbox("Preferred side (Low Volatility)", ["Long", "Short"], index=["Long", "Short"].index(get_default("pref_low_ui", "Long")), key="pref_low_ui")
    
    col_lots = st.columns(2)
    with col_lots[0]:
        long_high_size = st.number_input("Long, High Lot Size", value=get_default("lot_lh_ui", 1.0), step=1.0, key="lot_lh_ui")
        long_low_size = st.number_input("Long, Low Lot Size", value=get_default("lot_ll_ui", 2.0), step=1.0, key="lot_ll_ui")
    with col_lots[1]:
        short_high_size = st.number_input("Short, High Lot Size", value=get_default("lot_sh_ui", 3.0), step=1.0, key="lot_sh_ui")
        short_low_size = st.number_input("Short, Low Lot Size", value=get_default("lot_sl_ui", 4.0), step=1.0, key="lot_sl_ui")
    fixed_lots = {"Long_High": long_high_size, "Long_Low": long_low_size, "Short_High": short_high_size, "Short_Low": short_low_size}
    
    col_tick = st.columns(2)
    with col_tick[0]:
        tick_size = st.number_input("Tick Size", value=get_default("tick_size_ui", 0.00005), format=fmt, key="tick_size_ui")
    with col_tick[1]:
        tick_value = st.number_input("Tick Value", value=get_default("tick_value_ui", 5.0), format=fmt, key="tick_value_ui")
    
    out_type = st.selectbox("Select Output Type", ["trade_list", "final_df", "both"], index=["trade_list", "final_df", "both"].index(get_default("output_type_ui", "trade_list")), key="output_type_ui")
    sim_prefix = st.text_input("Simulation File Prefix", value=get_default("sim_prefix_ui", "UnifiedSim"), key="sim_prefix_ui")
    
    st.subheader("Run Simulation and Generate Report")
    if st.button("Run Unified Simulation"):
        os.makedirs("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/unified/", exist_ok=True)
        if len(long_cont_groups) == 0:
            long_cont_groups = long_entry_groups
        if len(short_cont_groups) == 0:
            short_cont_groups = short_entry_groups
        variant_combinations = list(product(long_entry_groups, short_entry_groups, long_cont_groups, short_cont_groups))
        total_variants = len(variant_combinations)
        pdf_report_info = []
        count = 0
        from core_processing import simulate_breakout_combo
        for idx, (l_entry, s_entry, l_cont, s_cont) in enumerate(variant_combinations, start=1):
            variant_id = f"V{idx}_L{long_entry_groups.index(l_entry)+1}_S{short_entry_groups.index(s_entry)+1}_LC{long_cont_groups.index(l_cont)+1}_SC{short_cont_groups.index(s_cont)+1}"
            variant_params = {
                "init_entry_condition_long": l_entry,
                "init_entry_condition_short": s_entry,
                "long_continue_condition": l_cont,
                "short_continue_condition": s_cont,
                "preferred_high": pref_high,
                "preferred_low": pref_low,
                "fixed_lots": fixed_lots,
                "tick_size": tick_size,
                "tick_value": tick_value,
                "output_type": out_type,
                "target_stop_variant": target_stop_variant
            }
            pdf_report_info.append((variant_id, variant_params))
            if target_stop_variant:
                for t_mult in target_multipliers:
                    for s_mult in stop_multipliers:
                        sim_variant = variant_params.copy()
                        sim_variant["target_multiplier"] = t_mult
                        sim_variant["stop_multiplier"] = s_mult
                        sim_variant["target_stop_mode"] = ts_mode
                        sim_variant["target_stop_preference"] = ts_preference
                        final_df, norm_trades = simulate_breakout_combo(df.copy(), sim_variant)
                        current_prefix = f"{sim_prefix}_{variant_id}_TS_T{t_mult}_S{s_mult}"
                        if out_type in ["final_df", "both"]:
                            out_final = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/unified/", f"{current_prefix}_final_df.csv")
                            final_df.to_csv(out_final, index=False)
                        if out_type in ["trade_list", "both"]:
                            out_trade = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/unified/", f"{current_prefix}_trade_list.csv")
                            norm_trades.to_csv(out_trade, index=False)
                        count += 1
            else:
                final_df, norm_trades = simulate_breakout_combo(df.copy(), variant_params)
                current_prefix = f"{sim_prefix}_{variant_id}"
                if out_type in ["final_df", "both"]:
                    out_final = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/unified/", f"{current_prefix}_final_df.csv")
                    final_df.to_csv(out_final, index=False)
                if out_type in ["trade_list", "both"]:
                    out_trade = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/unified/", f"{current_prefix}_trade_list.csv")
                    norm_trades.to_csv(out_trade, index=False)
                count += 1
        from core_processing import save_condition_tree_pdf
        pdf_output_path = os.path.join("C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/unified/", f"{sim_prefix}_Variant_Report.pdf")
        save_condition_tree_pdf(pdf_report_info, pdf_output_path)
        st.success(f"Unified Simulation completed for {count} simulation runs (from {total_variants} variants)!\nFiles saved in 'C:/Users/yash.patel/Python Projects/Algo_Builder/data/simulation_results/unified/' including PDF report: {pdf_output_path}")