from io import StringIO
from typing import Callable, Dict, Optional

import numpy as np
import openpyxl
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st


def set_pure_python_theme():
    """Sets a light, professional theme using only st.set_page_config."""
    st.set_page_config(
        page_title="Data Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'mailto:support@example.com',
            'Report a bug': "mailto:bugs@example.com",
            'About': "This is a professional data analysis dashboard built using pure Streamlit Python."
        }
    )

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def init_session_state() -> None:
    defaults = {
        "dataset": None,
        "dataset_name": "",
        "summary_cache": {},
        "cleaning_actions": [],
        "upload_error": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_dataset(
    df: Optional[pd.DataFrame], *, name: str = "", reset_history: bool = True
) -> None:
    st.session_state.dataset = df
    if name:
        st.session_state.dataset_name = name
    st.session_state.summary_cache = {}
    if reset_history:
        st.session_state.cleaning_actions = []


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


@st.cache_data
def build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "Missing Count": df.isna().sum(),
            "Missing %": (df.isna().mean() * 100).round(2),
        }
    )
    summary = summary[summary["Missing Count"] > 0].sort_values(
        "Missing %", ascending=False
    )
    summary.index.name = "Column"
    return summary


def apply_cleaning_action(df: pd.DataFrame, action: str) -> pd.DataFrame:
    cleaned = df.copy()

    if action == "Drop rows with missing values":
        cleaned = df.dropna().reset_index(drop=True)
        st.session_state.cleaning_actions.append("Dropped rows with missing values")

    elif action == "Fill numeric columns with mean":
        numeric_cols = get_numeric_columns(cleaned)
        if not numeric_cols:
            st.warning("No numeric columns available to fill.")
            return df
        means = cleaned[numeric_cols].mean()
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(means)
        st.session_state.cleaning_actions.append(
            "Filled numeric columns with their mean"
        )

    elif action == "Fill numeric columns with median":
        numeric_cols = get_numeric_columns(cleaned)
        if not numeric_cols:
            st.warning("No numeric columns available to fill.")
            return df
        medians = cleaned[numeric_cols].median()
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(medians)
        st.session_state.cleaning_actions.append(
            "Filled numeric columns with their median"
        )

    elif action == "Fill categorical columns with mode":
        categorical_cols = get_categorical_columns(cleaned)
        if not categorical_cols:
            st.warning("No categorical columns available to fill.")
            return df
        for col in categorical_cols:
            mode_val = cleaned[col].mode().iloc[0]
            cleaned[col] = cleaned[col].fillna(mode_val)
        st.session_state.cleaning_actions.append(
            "Filled categorical columns with their mode"
        )

    else:
        return df

    return cleaned


@st.cache_data
def generate_summary_report(df: pd.DataFrame) -> str:
    descriptive = df.describe(include="all").transpose()
    missing_summary = build_missing_summary(df)

    buffer = StringIO()
    buffer.write("# Descriptive Statistics\n")
    descriptive.to_csv(buffer)
    buffer.write("\n# Missing Value Summary\n")
    missing_summary.to_csv(buffer)

    return buffer.getvalue()


def render_header() -> None:
    st.title("📊 Data Analysis Dashboard")
    st.caption(
        "A tool to explore, clean, and visualize your CSV/Excel datasets."
    )
    st.divider()


def handle_file_upload() -> None:
    st.subheader("📤 Load Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Only .csv and .xlsx files are supported. Maximum size 200MB.",
        accept_multiple_files=False,
    )

    if not uploaded_file:
        if st.session_state.dataset is None:
            st.info("Awaiting file upload to begin analysis.")
        return

    if uploaded_file.name == st.session_state.dataset_name:
        return

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            st.error("Unsupported file type. Please upload a CSV or XLSX file.")
            return

    except pd.errors.EmptyDataError:
        update_dataset(None)
        st.error("The uploaded file is empty. Please provide a file with data.")
        return
    except Exception as exc:
        update_dataset(None)
        st.error(f"An unexpected error occurred while reading the file: {exc}")
        return

    if df.empty:
        update_dataset(None)
        st.error("The dataset contains no rows after loading.")
        return

    if df.columns.empty:
        update_dataset(None)
        st.error("The dataset contains no columns. Please verify the file content.")
        return

    update_dataset(df, name=uploaded_file.name)
    st.success(
        f"Successfully loaded {uploaded_file.name} with {len(df)} rows and {len(df.columns)} columns."
    )

    with st.expander("Quick view of the loaded data", expanded=False):
        st.dataframe(df.head(), use_container_width=True)


def render_data_summary(df: pd.DataFrame) -> None:
    st.subheader("📄 Data Summary")
    
    with st.container(border=True):
        st.markdown("#### 🔑 Key Statistics")
        total_rows = len(df)
        total_columns = len(df.columns)
        total_missing = int(df.isna().sum().sum())
        total_duplicates = int(df.duplicated().sum())
    
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{total_rows:,}")
        col2.metric("Total Columns", f"{total_columns:,}")
        col3.metric("Missing Values", f"{total_missing:,}", delta_color="inverse" if total_missing > 0 else "off")
        col4.metric("Duplicate Rows", f"{total_duplicates:,}", delta_color="inverse" if total_duplicates > 0 else "off")
    
    st.divider()

    with st.container(border=True):
        st.markdown("### 📋 Column Types & Missing Data")
        column_info = pd.DataFrame(
            {
                "Column Name": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Count": df.isna().sum().values,
                "Missing %": (df.isna().mean() * 100).round(2).values,
            }
        )
        st.data_editor(column_info, use_container_width=True, hide_index=True)

    st.divider()

    numeric_columns = get_numeric_columns(df)
    if not numeric_columns:
        st.warning(
            "No numeric columns identified. Some statistics and charts may be unavailable."
        )
        
    with st.container(border=True):
        st.markdown("### 📈 Descriptive Statistics (Numeric)")
        if numeric_columns:
            st.dataframe(
                df[numeric_columns].describe().transpose().round(3),
                use_container_width=True,
            )
        else:
            st.info("No numeric columns to display statistics for.")

    st.divider()

    with st.container(border=True):
        st.markdown("### 🔡 Category Counts (Categorical)")
        categorical_columns = get_categorical_columns(df)
        if categorical_columns:
            with st.expander("View detailed category value counts (Top 20)", expanded=False):
                for col in categorical_columns:
                    st.write(f"{col}")
                    st.dataframe(
                        df[col].value_counts().head(20).to_frame(),
                        use_container_width=True
                    )
        else:
            st.info("No categorical columns to display value counts for.")


def render_cleaning_section(df: pd.DataFrame) -> None:
    st.subheader("✨ Data Cleaning & Preprocessing")
    st.divider()

    with st.container(border=True):
        st.markdown("#### 1. Handle Duplicates")
        num_duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows Found", num_duplicates)
    
        if num_duplicates > 0:
            if st.button("Remove Duplicate Rows", key="btn_remove_duplicates", type="primary"):
                cleaned_df = df.drop_duplicates().reset_index(drop=True)
                st.session_state.cleaning_actions.append(
                    f"Removed {num_duplicates} duplicate rows"
                )
                update_dataset(
                    cleaned_df, name=st.session_state.dataset_name, reset_history=False
                )
                st.success("Duplicate rows removed. Dataset updated.")
                st.rerun()
        else:
            st.success("No duplicate rows detected.")

    st.divider()

    with st.container(border=True):
        st.markdown("#### 2. Manage Columns (Drop)")
        cols_to_drop = st.multiselect("Select columns to permanently drop", options=df.columns.tolist())
    
        if st.button("Drop Selected Columns", key="btn_drop_cols", type="secondary"):
            if not cols_to_drop:
                st.warning("Please select at least one column to drop.")
            else:
                cleaned_df = df.drop(columns=cols_to_drop)
                st.session_state.cleaning_actions.append(
                    f"Dropped columns: {', '.join(cols_to_drop)}"
                )
                update_dataset(
                    cleaned_df, name=st.session_state.dataset_name, reset_history=False
                )
                st.success(f"Columns {', '.join(cols_to_drop)} dropped. Dataset updated.")
                st.rerun()

    st.divider()

    with st.container(border=True):
        st.markdown("#### 3. Handle Missing Values")
        summary = build_missing_summary(df)
    
        if summary.empty:
            st.success("No missing values detected in the dataset.")
        else:
            st.write("Columns with missing values:")
            st.dataframe(summary, use_container_width=True)
    
            action = st.radio(
                "Select an imputation or removal strategy",
                (
                    "No action",
                    "Drop rows with missing values",
                    "Fill numeric columns with mean",
                    "Fill numeric columns with median",
                    "Fill categorical columns with mode",
                ),
                key="cleaning_action",
                horizontal=True
            )
            
            st.markdown("##### Confirmation Step")
            
            confirm_apply = st.checkbox(
                f"I confirm applying the action: '{action}'.",
                disabled=(action == "No action"),
                key='confirm_missing_action'
            )
    
            if st.button("Apply Missing Value Action", key="btn_apply_missing", type="primary"):
                if action == "No action":
                    st.info("No changes applied.")
                elif not confirm_apply:
                    st.error("Please check the confirmation box to apply the action.")
                else:
                    cleaned_df = apply_cleaning_action(df, action)
                    if cleaned_df is df:
                        st.warning("Action failed or had no effect. Check data types or console.")
                    else:
                        update_dataset(
                            cleaned_df,
                            name=st.session_state.dataset_name,
                            reset_history=False,
                        )
                        st.success(f"Action '{action}' applied successfully. Dataset updated.")
                        st.rerun()


def render_download_section(df: pd.DataFrame) -> None:
    st.subheader("📥 Download Report & Data")
    st.divider()

    with st.container(border=True):
        st.markdown("#### 1. Summary Report")
        st.write(
            "Generate a CSV report containing descriptive statistics and a missing value summary."
        )
    
        report_csv = generate_summary_report(df)
        st.download_button(
            label="Download Analysis Summary (CSV)",
            data=report_csv,
            file_name=f"summary_report_{st.session_state.dataset_name.split('.')[0]}.csv",
            mime="text/csv",
            type="primary"
        )

    st.divider()

    with st.container(border=True):
        st.markdown("#### 2. Current Dataset")
        st.write(
            "Download the dataset as it currently exists (with all cleaning actions applied)."
        )
    
        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="Download Cleaned Data (CSV)",
            data=csv_data,
            file_name=f"cleaned_{st.session_state.dataset_name.split('.')[0]}.csv",
            mime="text/csv",
            type="secondary"
        )

    if st.session_state.cleaning_actions:
        st.divider()
        with st.container(border=True):
            st.markdown("#### Applied Cleaning Actions Log")
            st.write("The following modifications have been applied to this dataset:")
            
            actions_df = pd.DataFrame({
                "Step": range(1, len(st.session_state.cleaning_actions) + 1),
                "Action": st.session_state.cleaning_actions
            })
            st.table(actions_df.set_index("Step"))


def render_about_section(_: pd.DataFrame) -> None:
    st.subheader("ℹ About This Dashboard")
    st.divider()
    
    with st.container(border=True):
        st.markdown("#### Application Features")
        st.write(
            """
            This interactive Streamlit dashboard provides a streamlined, professional data-analysis workflow:
    
            * Secure Upload: Easily load your .csv or .xlsx files.
            * Data Overview: Instantly grasp dataset dimensions, missing values, and duplicate rows.
            * Interactive Visualization: Leverage Plotly to generate a variety of charts for dynamic exploration.
            * Smart Data Cleaning: Address data quality issues with mean/median/mode imputation and clear confirmation steps.
            * Comprehensive Reporting: Download a detailed summary report and export your cleaned dataset.
            """
        )
    st.divider()
    st.caption("Developed using Streamlit, Pandas, Plotly, and Seaborn.")


def render_visualizations(df: pd.DataFrame) -> None:
    st.subheader("🎨 Visualization Explorer")
    st.divider()
    st.write("Select a plot type and configure options to reveal patterns and distributions in the dataset.")

    with st.container(border=True):
        plot_type = st.selectbox(
            "Choose a visualization type:",
            (
                "Histogram (Distribution)",
                "Bar Chart (Counts)",
                "Box Plot (Outliers)",
                "Heatmap (Correlation)",
                "Scatter Plot (Relationship)",
                "Pairplot (Scatter Matrix)",
            ),
            key="viz_type_selector"
        )
    
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    all_cols = numeric_cols + categorical_cols
    
    st.divider()

    # --- Histogram ---
    if plot_type.startswith("Histogram"):
        if not numeric_cols:
            st.warning("Histogram requires at least one numeric column.")
            return
        
        with st.container(border=True):
            st.markdown("##### 📈 Histogram Configuration")
            col1, col2 = st.columns([3, 1])
            with col1:
                column = st.selectbox("Numeric variable to plot", numeric_cols, key='hist_col')
            with col2:
                bins = st.slider("Number of bins", min_value=5, max_value=100, value=30, key='hist_bins')
            
            fig = px.histogram(df, x=column, nbins=bins, title=f"Distribution of {column}", marginal="box")
            st.plotly_chart(fig, use_container_width=True)

    # --- Bar Chart ---
    elif plot_type.startswith("Bar Chart"):
        if not categorical_cols:
            st.warning("Bar Chart requires at least one categorical column.")
            return
            
        with st.container(border=True):
            st.markdown("##### 📊 Bar Chart Configuration")
            col1, col2 = st.columns([3, 1])
            with col1:
                column = st.selectbox("Categorical variable to plot", categorical_cols, key='bar_col')
            with col2:
                top_n = st.slider("Show Top N categories", min_value=5, max_value=50, value=15, key='bar_n')
            
            counts = df[column].dropna().value_counts().head(top_n).reset_index()
            counts.columns = [column, "Count"]
            
            fig = px.bar(counts, x=column, y="Count", title=f"Bar Chart of Top {top_n} {column} Categories")
            st.plotly_chart(fig, use_container_width=True)

    # --- Box Plot ---
    elif plot_type.startswith("Box Plot"):
        if not numeric_cols:
            st.warning("Box Plot requires at least one numeric column.")
            return
            
        with st.container(border=True):
            st.markdown("##### 📦 Box Plot Configuration")
            column = st.selectbox("Numeric variable to check for outliers", numeric_cols, key='box_col')
            fig = px.box(df, y=column, title=f"Box Plot for {column}")
            st.plotly_chart(fig, use_container_width=True)

    # --- Heatmap ---
    elif plot_type.startswith("Heatmap"):
        if len(numeric_cols) < 2:
            st.warning("Heatmap requires at least two numeric columns.")
            return

        with st.container(border=True):
            st.markdown("##### 🔥 Correlation Heatmap")
            st.write("This map shows the Pearson correlation between numeric columns. Values range from -1 (total negative correlation) to 1 (total positive correlation).")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap",
                            color_continuous_scale='RdBu_r', range_color=[-1, 1])
            st.plotly_chart(fig, use_container_width=True)

    # --- Scatter Plot ---
    elif plot_type.startswith("Scatter"):
        if len(numeric_cols) < 2:
            st.warning("Scatter Plot requires at least two numeric columns.")
            return
            
        with st.container(border=True):
            st.markdown("#####  Scatter Plot Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                x_ax = st.selectbox("X-Axis", numeric_cols, key='scatter_x')
            with col2:
                # Filter out the x-axis column from y-axis options
                y_options = [c for c in numeric_cols if c != x_ax]
                y_ax = st.selectbox("Y-Axis", y_options, key='scatter_y')
            with col3:
                # Add an optional color dimension
                color_options = [None] + all_cols
                color_by = st.selectbox("Color by (Optional)", color_options, key='scatter_color')

            if x_ax and y_ax:
                fig = px.scatter(
                    df, 
                    x=x_ax, 
                    y=y_ax, 
                    color=color_by,
                    title=f"Scatter Plot: {x_ax} vs {y_ax}"
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Pairplot ---
    elif plot_type.startswith("Pairplot"):
        if len(numeric_cols) < 2:
            st.warning("Pairplot requires at least two numeric columns.")
            return
            
        with st.container(border=True):
            st.markdown("##### 🌐 Pairplot (Scatter Matrix) Configuration")
            st.write("A pairplot shows scatter plots for all pairs of numeric columns and histograms on the diagonal.")
            
            col1, col2 = st.columns(2)
            with col1:
                # Let user select a subset of numeric columns to plot (plotting all can be slow)
                default_cols = numeric_cols[:min(len(numeric_cols), 5)] # Default to first 5
                plot_vars = st.multiselect(
                    "Select numeric columns to plot", 
                    numeric_cols, 
                    default=default_cols,
                    key='pairplot_vars'
                )
            with col2:
                # Add an optional hue dimension
                hue_options = [None] + categorical_cols
                hue_by = st.selectbox("Color by (Optional)", hue_options, key='pairplot_hue')
            
            if len(plot_vars) < 2:
                st.info("Please select at least two numeric columns to create a pairplot.")
            else:
                st.write("Generating pairplot... (this may take a moment)")
                # Seaborn's pairplot is a figure-level function, so we use st.pyplot()
                pairplot_fig = sns.pairplot(df, vars=plot_vars, hue=hue_by)
                st.pyplot(pairplot_fig)


def build_sidebar() -> str:
    st.sidebar.title("🧭 Navigation")
    st.sidebar.divider()
    st.sidebar.caption("Explore your data with these tools:")

    section_map = {
        "📄 Data Summary": "Summary",
        "🎨 Visualizations": "Visualizations",
        "✨ Data Cleaning": "Cleaning",
        "📥 Download Report": "Report",
        "ℹ About": "About",
    }
    
    section_label = st.sidebar.radio(
        "Go to",
        section_map.keys(),
        index=0,
    )

    return section_map[section_label]


def main() -> None:
    set_pure_python_theme()
    init_session_state()

    render_header()

    uploader_container = st.container()
    with uploader_container:
        handle_file_upload()

    if st.session_state.dataset is None:
        st.divider()
        st.info("Upload a dataset using the file uploader above to unlock the dashboard features and navigation.")
        return

    section = build_sidebar()

    section_renderers: Dict[str, Callable[[pd.DataFrame], None]] = {
        "Summary": render_data_summary,
        "Visualizations": render_visualizations,
        "Cleaning": render_cleaning_section,
        "Report": render_download_section,
        "About": render_about_section,
    }

    current_df = st.session_state.dataset
    renderer = section_renderers.get(section)

    if renderer:
        renderer(current_df.copy())
    else:
        st.error(f"Error: Renderer for section '{section}' not found.")


if __name__ == "__main__":
    main()