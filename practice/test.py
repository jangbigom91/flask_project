import pandas as pd
import plotly.express as px

df = pd.read_csv('PM10_seoul.csv')

fig = px.line(df, x='Date', y='Seoul', title='PM10 of Seoul(2014~2021)')

fig_seoul = fig.show()