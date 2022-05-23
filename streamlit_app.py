import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from numpy.random import normal
from functools import cache
import plotly.express as px
from plotly.subplots import make_subplots
from itertools import combinations


st.set_page_config(
    layout="centered", page_icon="https://mma.prnewswire.com/media/1638770/CAFEMEDIA_LOGO.jpg", page_title="Data Analysis, CafeMedia"
)
with st.echo("below"):
    st.title("Data Analysis, CafeMedia")
    st.write("By: Joseph Perrotta")

    st.title("Part 1: Data Table")

    def aggrid_interactive_table(df: pd.DataFrame):
        """Creates an st-aggrid interactive table based on a dataframe.

        Args:
            df (pd.DataFrame]): Source dataframe

        Returns:
            dict: The selected row
        """
        options = GridOptionsBuilder.from_dataframe(
            df, enableRowGroup=True, enableValue=True, enablePivot=True
        )

        options.configure_side_bar()

        options.configure_selection("single")
        selection = AgGrid(
            df,
            enable_enterprise_modules=True,
            gridOptions=options.build(),
            theme="light",
            update_mode=GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True,
        )

        return selection

    df = pd.read_csv("Test Data for Data Scientist Screening Homework.csv")

    selection = aggrid_interactive_table(df=df)

    if selection:
        st.write("You selected:")
        st.json(selection["selected_rows"])

    st.write("Part 1A: Time On Page & Revenue:")

    with st.expander("Images & Insight", expanded=False):
        df_rev_top = df[["revenue", "top"]]

        chart1 = alt.Chart(df_rev_top).mark_circle().encode(
             x='top', y='revenue', tooltip=['revenue', 'top',])
        st.altair_chart(chart1, use_container_width=True)

        chart2 = alt.Chart(df_rev_top).mark_circle().encode(
             x='revenue', y='top', tooltip=['revenue', 'top',])
        st.altair_chart(chart2, use_container_width=True)

        st.write("At a glance, the images indicate that as time on page increases revenue decreases ")

    st.write("Part 1B: Time on Page & Revenue When Controlling for Other Variables")

    with st.expander("Image Combinations and Insights", expanded=False):

        st.write("These images unveil a different relationship between ToP and Revenue. When controlling for other variables it shows that Revenue generally increases as ToP increases")

        st.write("Image Plots are Interactive!")


        use_df = df.copy()
        use_df["site"] = use_df["site"].astype(str)  # Help plotly see this as categorical
        x_col = "top"
        y_col = "revenue"
        x_data = df[x_col]
        y_data = df[y_col]
        x_limits = [x_data.min(), x_data.max()]
        y_limits = [y_data.min(), y_data.max()]

        # Produce combinations of columns
        base_cols = ["browser", "platform", "site"]
        cols = base_cols.copy()


        def combine_columns(columns):
            new_col = None
            for col in columns:
                this_part = f"{col}=" + use_df[col]
                if new_col is None:
                    new_col = this_part
                else:
                    new_col += ", " + this_part
            new_col_name = " & ".join(columns)
            use_df[new_col_name] = new_col
            cols.append(new_col_name)


        # Make new column representing each pair of columns
        for col_pair in combinations(base_cols, 2):
            combine_columns(col_pair)

        # Make new column representing each unique set of possibility
        combine_columns(base_cols)

        data = use_df.to_dict()


        def generate_plots(col):
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                labels={
                    x_col: "Time on Page",
                    y_col: "Revenue ($)",
                    col: "",  # col,
                },
                color=col,
                title=f"Time on Page v. Revenue, by {col}",
            )
            fig.update_xaxes(range=x_limits)
            fig.update_yaxes(range=y_limits)
            st.plotly_chart(fig, use_container_width=True)


        for col in cols:
            generate_plots(col)


    st.markdown(
        """
        PandasProfiling EDA and Statistical Analysis:

        Using the Python package PandasProfiling we're able to quickly analyze the dataset and discover some insights.

        The HTML report provides an overview of the data as well as several correlation metrics.

        Time on Page is highly correlated to Platform and Browser is highly correlated to Revenue.

        Open the Expander below to see the full report:
        """
    )

    st.image("correlations.png")

    with st.expander("Show Full Report", expanded=False):

        st.write("The report is interactive! Click in each section to see more details.")

        @cache
        def generate_report():
            return ProfileReport(df, title="PandasProfiling Report", explorative=True)
        pr = generate_report()

        st_profile_report(pr)


    st.title("Part 2: Bimodal Distribution")

    # First normal distribution parameters
    mu1 = 1
    sigma1 = 0.15
    # Second normal distribution parameters
    mu2 = 2
    sigma2 = 0.25
    w1 = 2/3 # Proportion of samples from first distribution
    w2 = 1 - w1 # Proportion of samples from second distribution
    n = 10000 # Total number of samples
    n1 = round(n*w1) # Number of samples from first distribution
    n2 = round(n*w2) # Number of samples from second distribution
    # Generate n1 samples from the first normal distribution and n2 samples from the second normal distribution
    X = np.concatenate((normal(mu1, sigma1, n1), normal(mu2, sigma2, n2)))
    X -= X.min()
    X /= X.max()
    # Determine parameters mu1, mu2, sigma1, sigma2, w1 and w2
    gm = GaussianMixture(n_components=2, random_state=0).fit(X.reshape(-1, 1))
    print(f'mu1={gm.means_[0]}, mu2={gm.means_[1]}')
    print(f'sigma1={np.sqrt(gm.covariances_[0])}, sigma2={np.sqrt(gm.covariances_[1])}')
    print(f'w1={gm.weights_[0]}, w2={gm.weights_[1]}')
    print(f'n1={int(n * gm.weights_[0])} n2={int(n * gm.weights_[1])}')

    import altair as alt
    df = pd.DataFrame({"data": X})
    st.altair_chart(alt.Chart(df).mark_bar().encode(
        x=alt.X(
            "data",
            bin=alt.BinParams(maxbins=50),
            axis=alt.Axis(values=(0,1)),
        ),
        y="count()",
    ))

    @st.cache
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Bimodal_Dist_JP.csv',
        mime='text/csv',
    )

    with st.expander("Data Points for Distribution", expanded=False):
        st.table(df)

    st.title("Code for App")


