import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from pylab import concatenate, normal

st.set_page_config(
    layout="centered", page_icon="JP️", page_title="Data Analysis App"
)
st.title("Data Analysis App")
# st.write(
#     """This app shows how you can use the [streamlit-aggrid](STREAMLIT_AGGRID_URL)
#     Streamlit component in an interactive way so as to display additional content
#     based on user click."""
# )


st.write("Part 1: Data Table")

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

st.title("Time On Page & Revenue:")

df_rev_top = df[["revenue", "top"]]

chart1 = alt.Chart(df_rev_top).mark_circle().encode(
     x='top', y='revenue', tooltip=['revenue', 'top',])
st.altair_chart(chart1, use_container_width=True)

chart2 = alt.Chart(df_rev_top).mark_circle().encode(
     x='revenue', y='top', tooltip=['revenue', 'top',])
st.altair_chart(chart2, use_container_width=True)

st.write("As shown in the images, as time on page increases revenue decreases ")

st.title("PandasProfiling EDA and Statistical Analysis")
st.write("Using the Python package PandasProfiling we're able to quickly analyze the dataset and discover insights. ")
st.write("The HTML report provides an overview of the data as well as several correlation metrics. ")
st.write("Time on Page is highly correlated to Platform and Browser is highly correlated to Revenue. ")
st.write("Other insights include:")
st.image("correlations.png")

pr = ProfileReport(df, title="PandasProfiling Report", explorative=True)
with st.expander("View in Depth Analysis", expanded=False):
    st_profile_report(pr)



# N=10000
# mu, sigma = 1, .005
# mu2, sigma2 = .1, .04
# X1 = np.random.normal(mu, sigma, N)
# X2 = np.random.normal(mu2, sigma2, N)
# X = np.concatenate([X1, X2])
# st.line_chart(X)
# st.altair_chart(X)


# First normal distribution parameters
mu1 = 1
sigma1 = 0.1

# Second normal distribution parameters
mu2 = 2
sigma2 = 0.2

w1 = 2/3 # Proportion of samples from first distribution
w2 = 1/3 # Proportion of samples from second distribution

n = 10000 # Total number of samples

n1 = int(n*w1) # Number of samples from first distribution
n2 = int(n*w2) # Number of samples from second distribution

# Generate n1 samples from the first normal distribution and n2 samples from the second normal distribution
X = concatenate((normal(mu1, sigma1, n1), normal(mu2, sigma2, n2))).reshape(-1, 1)

# Determine parameters mu1, mu2, sigma1, sigma2, w1 and w2
gm = GaussianMixture(n_components=2, random_state=0).fit(X)
print(f'mu1={gm.means_[0]}, mu2={gm.means_[1]}')
print(f'sigma1={np.sqrt(gm.covariances_[0])}, sigma2={np.sqrt(gm.covariances_[1])}')
print(f'w1={gm.weights_[0]}, w2={gm.weights_[1]}')
print(f'n1={int(n * gm.weights_[0])} n2={int(n * gm.weights_[1])}')

st.line_chart(X)
st.altair_chart(X)


with st.expander("## Code", expanded=True):
    st.code(
        '''
    import pandas as pd
    import streamlit as st
  
    ''',
        "python",
    )
