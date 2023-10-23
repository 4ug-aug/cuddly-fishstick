"""

My main app
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
# plotly
import plotly.express as px

st.set_page_config(
        page_title="Data Profiler",
        page_icon="📊",
)

st.markdown("# Dataset Profiling 📊")
# st.sidebar.markdown("# Profiler")

#....................... DATA ACCESS .......................

if st.session_state.get("data") is not None:
    df = st.session_state.data
else:
    st.warning("No data uploaded")
    st.stop()

#....................... DATA PROFILING .......................

# Save df button
if st.sidebar.button("Save dataframe"):
    # Check if mani_data folder exists
    if not os.path.exists("mani_data"):
        os.makedirs("mani_data")
        st.sidebar.success("mani_data folder created", icon="📁")

    # Save df to csv
    df.to_csv("mani_data/df.csv", index=False)
    st.sidebar.success("Dataframe saved to mani_data folder", icon="💾")

# Add sidebar options
data_mani_exp = st.sidebar.expander("Data Manipulations", expanded=True)

# Make container
st.write("## Data Profiling")
c1 = st.container()

n_variables = len(df.columns)
n_observation = len(df)
missing_values = df.isnull().sum().sum()
missing_cells_percentage = round(missing_values / (n_variables * n_observation) * 100, 2).round(2)
duplicate_rows = df.duplicated().sum()
ducpliate_rows_percentage = round(duplicate_rows / n_observation * 100, 2).round(2)
# Memory size in KiB
total_size = round(df.memory_usage().sum() / 1024, 2)

# Variables
variables_type_cat = df.select_dtypes(include=['object']).columns
variables_type_num = df.select_dtypes(include=['number']).columns
variables_type_bool = df.select_dtypes(include=['bool']).columns

df_variable_types = pd.DataFrame({
    "Variable Type": ["Categorical", "Numerical", "Boolean"],
    "Number of Variables": [len(variables_type_cat), len(variables_type_num), len(variables_type_bool)]
})
# Set cols as index
df_variable_types.set_index("Variable Type", inplace=True)

df_summary = pd.DataFrame({
    " ": ["Number of Variables", "Number of Observations", "Missing Cells", "Missing Cells (%)", 
          "Duplicate Rows", "Duplicate Rows (%)", "Total Size (KiB)"],
    "Value": [n_variables, n_observation, missing_values, missing_cells_percentage, 
              duplicate_rows, ducpliate_rows_percentage, total_size]
})
# remove trailing zeros from values
df_summary["Value"] = df_summary["Value"].astype(str).str.rstrip('0').str.rstrip('.')
# Set cols as index
df_summary.set_index(" ", inplace=True)

# Make columns
col1, col2 = c1.columns(2)

# Display summary
col1.write("### Summary")
col1.table(df_summary)

# Display variable types
col2.write("### Variable Types")
col2.table(df_variable_types)

# IF MISSING VALUES IN COLS

# add button to sidebar to remove cols with missing values
if missing_values > 0:
    if data_mani_exp.button("Remove columns with missing values"):
        df = df.dropna(axis=1)
    # Dropdown with impute options
    impute_options = ["Mean", "Median", "Mode"]
    impute = data_mani_exp.selectbox("Impute missing values with:", impute_options)
    if impute == "Mean":
        df = df.fillna(df.mean())
    elif impute == "Median":
        df = df.fillna(df.median())
    elif impute == "Mode":
        df = df.fillna(df.mode().iloc[0])
    st.rerun()

# if we have one or more categorical variables with more than 2 unique values
# add button to sidebar to convert to dummy variables
if len(variables_type_cat) > 0:
    # Choose categorical variables
    cat_var = data_mani_exp.multiselect("Choose categorical variables to convert to dummies:", variables_type_cat)
    if len(cat_var) > 0:
        if data_mani_exp.button("Convert to dummy variables"):
            # if the categorical variable has more than 2 unique values
            # convert to dummy variables
            if len(df[cat_var].nunique()) > 2:
                df = pd.get_dummies(df, columns=cat_var)
            else:
                # binary encoding
                df[cat_var] = df[cat_var].astype('category')
                df[cat_var] = df[cat_var].apply(lambda x: x.cat.codes)

            # save df
            st.rerun()

# ....................... VARIABLE DESCRIPTIONS .......................

def correct_corr():
    # Correlation matrix
    corr = df.corr()
    # If some have high correlation
    if (corr > 0.8).any().sum() > 0:
        # Get the columns with high correlation
        cols = np.where(corr > 0.8)
        # Create a set to store the columns to drop
        to_drop = set()
        # Loop through the columns with high correlation
        for i in range(len(cols[0])):
            # Get the name of the columns
            col1 = corr.columns[cols[0][i]]
            col2 = corr.columns[cols[1][i]]
            # If the columns are not the same
            if col1 != col2:
                # If the columns are not already in the set
                if col1 not in to_drop and col2 not in to_drop:
                    # Add the column with the highest correlation with the target to the set
                    if corr[col1].sum() > corr[col2].sum():
                        to_drop.add(col2)
                    else:
                        to_drop.add(col1)
        # Drop the columns with high correlation
        df = df.drop(to_drop, axis=1)

@st.cache_data
def check_corr():
    # Correlation matrix
    # remove columns with string values
    df_corr = df.select_dtypes(exclude=['object']).copy()
    corr = df_corr.corr()
    # get columns with high correlation above 0.8
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if corr.iloc[i, j] > 0.8:
                pairs.append([corr.columns[i], corr.columns[j]])
    if len(pairs) > 0:
        return pairs
    else:
        return None


@st.cache_data
def numeric_var_description(var):
    # Create variable description
    var_description = {}
    var_description["mean"] = "Mean: " + str(df[var].mean())
    var_description["std"] = "Standard Deviation: " + str(df[var].std())
    var_description["min"] = "Minimum: " + str(df[var].min())
    var_description["max"] = "Maximum: " + str(df[var].max())
    var_description["missing"] = "Missing values: " + str(df[var].isnull().sum())
    var_description["missing(%)"] = "Missing values (%): " + str(round(df[var].isnull().sum() / len(df[var]) * 100, 2))
    var_description["unique"] = "Unique values: " + str(df[var].nunique())
    # Add variable description
    return var_description

@st.cache_data
def categorical_var_description(var):
    # Create variable description
    var_description = {}
    var_description["missing"] = "Missing values: " + str(df[var].isnull().sum())
    var_description["missing(%)"] = "Missing values (%): " + str(round(df[var].isnull().sum() / len(df[var]) * 100, 2))
    var_description["unique"] = "Unique values: " + str(df[var].nunique())
    var_description["top"] = "Top value: " + str(df[var].value_counts().idxmax())
    var_description["freq"] = "Frequency of top value: " + str(df[var].value_counts().max())
    var_description["balanced"] = "Balanced: " + str(df[var].value_counts().max() / df[var].value_counts().sum() * 100)
    # Add variable description
    return var_description

var_desc, alerts = st.tabs(["Variable Descriptions", "Quality Alerts"])

# Variable descriptions
# Define sidebar
with var_desc:
    st.title("Select a variable")
    variable = st.selectbox("Variable", df.columns)

# Define main content
with var_desc:
    st.title("Variable Descriptions")
    if variable:
        st.markdown(f"### {variable}")
        st.markdown(f"**Type:** {df[variable].dtype}")

        # var_desc_col1, var_desc_col2 = st.columns(2)

        if df[variable].dtype == "object":
            props = categorical_var_description(variable)

            for key, value in props.items():
                st.markdown(f"**{key}:** {value}")
            
            percent = df[variable].value_counts(normalize=True) * 100
            # make bar chart of distribution
            st.plotly_chart(px.bar(x=percent.index, y=percent,
                                        title=f"Distribution of {variable}",
                                        labels={"x": variable, "y": "Count"}))
        else:
            props = numeric_var_description(variable)

            for key, value in props.items():
                st.markdown(f"**{key}:** {value}")
            # make histogram
            st.plotly_chart(px.histogram(df, x=variable, 
                                            marginal="box", 
                                            title=f"Distribution of {variable}").update_layout(bargap=0.1))





# Quality alerts
with alerts:
    st.title("Quality Alerts")

    sidebar_warning_exp = st.sidebar.expander("Quality Alerts", expanded=True)

    alerts_to_print = []

    # run quality checks on all features
    with st.spinner():
        # check for highly correlated features
        highly_corr = check_corr()

        if highly_corr:
            # make bullet list of highly correlated features
            corr_list = []
            for pair in highly_corr:
                st.warning(f"Highly correlated features detected: {pair[0]} and {pair[1]}", icon="🔥")
                sidebar_warning_exp.warning(f"Highly correlated features detected: {pair[0]} and {pair[1]}", icon="🔥")

            if data_mani_exp.button("Correct highly correlated features"):
                correct_corr()
                st.success("Highly correlated features removed")

        for col in df.columns:
            # check for missing values
            missing_values = df[col].isnull().sum()
            if missing_values > 0:
                alerts_to_print.append(f"Missing values in {col}: {missing_values}")
            # check for duplicate rows
            if df[col].dtype != "object":
                duplicate_rows = df[col].duplicated().sum()
                if duplicate_rows > 0:
                    alerts_to_print.append(f"Duplicate rows in {col}: {duplicate_rows}")
            # check balance of categorical features
            if df[col].dtype == "object":
                # check if evenly distributed
                if df[col].value_counts().max() / df[col].value_counts().sum() * 100 > 80:
                    st.warning(f"Unbalanced feature detected: {col}", icon="🔥")
                # if binary check if 0 and 1
                if df[col].nunique() == 2:
                    if 0 not in df[col].unique() or 1 not in df[col].unique():
                        st.warning(f"Binary feature detected but not 0 and 1: {col}")
                # if not binary check if 0 and 1
                else:
                    if 0 in df[col].unique() or 1 in df[col].unique():
                        st.warning(f"Non-binary feature detected but has 0 and 1: {col}")

    if len(alerts_to_print) > 0:
        for alert in alerts_to_print:
            st.warning(alert)
            sidebar_warning_exp.warning(alert)
    else:
        sidebar_warning_exp.success("No quality alerts detected", icon="🎉")


