# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:04:42 2024

@author: cutliffed
"""

import streamlit as st
import pandas as pd
# connect to redshift
import sys
import os
#import matplotlib.pylot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import re
from plotly.subplots import make_subplots  # Import make_subplots from plotly.subplots

sys.path.append(os.path.join('//paddystore', 'cit$', 'CIT - Web Analytics', 'Users Sandbox', 'Donal', 'Python'))
from sqlalchemy import create_engine
c = create_engine('postgresql+psycopg2://app_r_commercial:GamingPerformance1@redshift.app.betfair:5439/edcrsdbprod')


import_query = "SELECT * FROM pp_sports_shared.dc_fbref_macth_summary_24"
df = pd.read_sql_query(import_query, c)


#Data Manipulation

df['position_category'] = df['position'].apply(
    lambda x: 'Goalkeeper' if 'GK' in x.split(',')[0]
    else 'Defender' if 'B' in x.split(',')[0]
    else 'Midfielder/Striker'
)
df['xgi'] = df['xa'] + df['npxg']

# Streamlit page settings
st.set_page_config(page_title="FPL Short Term Performance", page_icon=":bar_chart:", layout="wide")
st.markdown(
    '''
    <style>
        .css-1v0mbdj {  
            margin-top: 0rem;
        }
        h1 { 
            font-size: 20px;
            margin-top: 0;
            margin-bottom: 0rem;
        }
        div.block-container {
            padding-top: 1.2rem;
        }
    </style>
    ''',
    unsafe_allow_html=True
)

# Display the title
st.title(":bar_chart: FPL-Short Term Performance Dashboard")

# Sidebar Filters
st.sidebar.header("Choose your filter:")

# 1. Position Filter
position_options = sorted(df["position_category"].unique())  # Clean and sort unique positions
position = st.sidebar.multiselect("Pick the Player Position", options=position_options, default=None)

# 2. Team Filter
team_options = sorted(df["team_name"].str.strip().unique())  # Clean and sort unique team names
team = st.sidebar.multiselect("Pick the Team", options=team_options, default=None)

# 3. Ensure `gw_start_date` is datetime
df["gw_start_date"] = pd.to_datetime(df["gw_start_date"], errors="coerce")

# 4. Clean the `stage` column to keep only "Matchweek X"
df["clean_stage"] = df["stage"].str.extract(r"(Matchweek \d+)")[0]

# 5. Format `gw_start_date` as month/year (e.g., "12/24")
df["month_year"] = df["gw_start_date"].dt.strftime("%m/%y")

# 6. Combine `month_year` and `clean_stage` to create a slider label
df["stage_with_month"] = df["month_year"] + " - " + df["clean_stage"]

# 7. Sort the DataFrame by `gw_start_date` to ensure chronological order
df_sorted = df.sort_values(by="gw_start_date")

# 8. Get unique `stage_with_month` values in the order of `gw_start_date`
available_stages = df_sorted["stage_with_month"].dropna().unique()

# 9. Find the latest `stage_with_month` value (corresponding to the most recent `gw_start_date`)
default_stage = df_sorted.iloc[-1]["stage_with_month"]

# Sidebar: Slider for selecting Matchweek
selected_stage_with_month = st.sidebar.select_slider(
    "Pick the Matchweek",
    options=available_stages,  # Provide sorted stages with month/year
    value=default_stage  # Default to the most recent Matchweek
)

# 10. Split the selected value back into components for filtering
selected_month_year, selected_clean_stage = selected_stage_with_month.split(" - ")

# Build the query dynamically based on user input
query_conditions = []

# Add conditions based on selected filters
query_conditions.append(f"month_year == '{selected_month_year}'")
query_conditions.append(f"clean_stage == '{selected_clean_stage}'")

# Add position filter if selected
if position:
    query_conditions.append(f"position_category in {position}")

# Add team filter if selected
if team:
    query_conditions.append(f"team_name in {team}")

# Combine all conditions into a single query string
query_string = " & ".join(query_conditions)

# Filter the DataFrame using the query method
filtered_df = df.query(query_string)

# Debugging: Show the filtered DataFrame for checking
if st.sidebar.checkbox("Show Filtered DataFrame for Debugging", value=False):
    st.write("Filtered DataFrame:")
    st.write(filtered_df)

# Show a message if no data matches the filters
if filtered_df.empty:
    st.warning("No data matches your filters. Please adjust the filters.")
else:
    # Proceed with visualizations or further analysis using `filtered_df`
    pass  # Replace this with your visualization or analysis code


# Metric selection for scatter plot and trend graph
metrics = ['xgi', 'npxg', 'xa', 'att_pen_touches', 'att_third_touches']
selected_metric = st.selectbox("Select Metric for Analysis", options=metrics)

# Scatter Plot Sectio
x_axis = f"avg_10_week_{selected_metric}"  # Example: "avg_10_week_xgi"
y_axis = f"{selected_metric}_diff_3_10"   # Example: "xgi_diff_3_10"

if x_axis not in filtered_df.columns or y_axis not in filtered_df.columns:
    st.error(f"Selected metric '{selected_metric}' does not exist in the DataFrame.")
else:
    fig = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        hover_name="player",
        color=y_axis,
        color_continuous_scale=[(0, "red"), (0.5, "white"), (1, "blue")],
        range_color=[filtered_df[y_axis].min(), filtered_df[y_axis].max()]
    )
    fig.update_layout(
        title=f"{selected_metric.capitalize()} Change vs 10-Week Average {selected_metric.capitalize()}",
        xaxis=dict(title=f"10-Week Average {selected_metric.capitalize()}", showgrid=False),
        yaxis=dict(title=f"3-10 Week {selected_metric.capitalize()} Difference", showgrid=False),
        width=800,
        height=600,
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
# Trend Graph Section
dynamic_avg_column = f'avg_10_week_{selected_metric}'
dynamic_avg_column_3_week = f'avg_3_week_{selected_metric}'

if dynamic_avg_column not in df.columns or dynamic_avg_column_3_week not in df.columns:
    st.error(f"Columns for the selected metric '{selected_metric}' are missing in the DataFrame.")
else:
    # Filter data by the selected stage
    filtered_df_stage = df[df["stage_with_month"] == selected_stage_with_month]

    # Get the player with the largest absolute change in the metric_diff_3_10
    default_player = filtered_df_stage.loc[filtered_df_stage[f'{selected_metric}_diff_3_10'].idxmax(), 'player']

    # Sidebar: Select the player based on the largest absolute change in the metric_diff_3_10
    selected_player = st.selectbox('Select a Player for Detailed View', options=filtered_df_stage['player'].unique(), index=list(filtered_df_stage['player'].unique()).index(default_player))
    
    # Filter and sort player data for the selected player
    player_data = df[df['player'] == selected_player].copy()
    player_data_sorted = player_data.sort_values(by='gw_start_date', ascending=True)
    last_13_dates = player_data_sorted['gw_start_date'].drop_duplicates().tail(13)
    trend_data = player_data_sorted[player_data_sorted['gw_start_date'].isin(last_13_dates)]

    trend_data['opposition'] = np.where(
        trend_data['team_name'] == trend_data['home_team'], 
        trend_data['away_team'], 
        trend_data['home_team']
    )
    # Average data for trend graph
    avg_trend_data = trend_data.groupby(['stage', 'gw_start_date']).agg({
        selected_metric: 'mean',
        'opposition': lambda x: ' & '.join(x.unique())
    }).reset_index()
    avg_trend_data_sorted = avg_trend_data.sort_values(by='gw_start_date', ascending=True)

    avg_10_week_metric = trend_data[dynamic_avg_column].iloc[-1]
    avg_3_week_metric = trend_data[dynamic_avg_column_3_week].rolling(window=3, min_periods=1).mean()

    # Clean the 'stage' column to remove "Premier League" and brackets
    avg_trend_data_sorted['stage'] = avg_trend_data_sorted['stage'].str.extract(r'(\d+)')[0].apply(lambda x: f"GW{x}")

    def generate_abbreviation(team_name):
        abbreviation = ''.join([word[0].upper() for word in team_name.split()]) + "FC"
        return abbreviation

    avg_trend_data_sorted['abbreviated_opposition'] = avg_trend_data_sorted['opposition'].apply(generate_abbreviation)

    # Create a single plot without dual axes
    fig3 = go.Figure()

    # Add Bar Trace for Selected Metric (Primary y-axis)
    fig3.add_trace(
        go.Bar(
            x=avg_trend_data_sorted['stage'],
            y=avg_trend_data_sorted[selected_metric],
            name=f"{selected_metric.capitalize()} Value",
        )
    )

    # Add area chart for 3-week dynamic average (Primary y-axis)
    fig3.add_trace(
        go.Scatter(
            x=avg_trend_data_sorted['stage'],
            y=avg_3_week_metric,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(0, 100, 200, 0.2)',
            name=f'Avg 3 Week {selected_metric}',
            line=dict(color='rgba(0, 100, 200, 0.8)'),
        )
    )

    # Add annotations for the opposition team
    fig3.add_trace(
        go.Scatter(
            x=avg_trend_data_sorted['stage'],
            y=avg_trend_data_sorted[selected_metric] + avg_trend_data_sorted[selected_metric] * 0.1,
            mode='text',
            text=[f'{abbr}' for abbr in avg_trend_data_sorted['abbreviated_opposition']],
            textposition='top center',
            textfont=dict(size=7.5),
            showlegend=False
        )
    )

    # Add a reference line for 10-week average
    overall_average_metric = avg_10_week_metric.mean()
    fig3.add_hline(
        y=overall_average_metric,
        line_dash="dot",
        annotation_text=f"Avg 10 Week {selected_metric}: {overall_average_metric:.2f}",
        annotation_position="top left",
        line_color="grey",
        annotation_font=dict(size=9),
        annotation_x=1.01,
        annotation_xanchor="left",
        annotation_yanchor="middle"
    )

    # Update layout for the figure
    fig3.update_layout(
        title=dict(
            text=f"{selected_metric.capitalize()} Bar Chart for {selected_player}",
            font=dict(size=14),
            x=0.0,
            y=0.97,
            yanchor='top'
        ),
        xaxis=dict(
            title="Gameweek (Stage)",
            titlefont=dict(size=10),
            tickfont=dict(size=8),
            categoryorder='array',
            categoryarray=avg_trend_data_sorted['stage']
        ),
        yaxis=dict(
            title=f"{selected_metric.capitalize()} Value",
            titlefont=dict(size=10),
            tickfont=dict(size=8)
        ),
        legend=dict(
            font=dict(size=8),
            x=0.5,
            y=-0.2,
            xanchor='center',
            yanchor='top'
        ),
        margin=dict(
            t=50,
            b=50,
            l=50,
            r=150
        ),
        height=400,
        width=600
    )

    # Display the plot
    st.plotly_chart(fig3, use_container_width=True)








































#data1 = px.scatter(filtered_df, x="avg_10_week_xgi", y="xgi_diff_3_10",
#                   hover_name="player" , # Change 'player' to the column that has the player names
#                   color="xgi_diff_3_10",  # Color based on the xgi_diff_3_10 column
#                   color_continuous_scale=[(0, "red"), (0.5, "white"), (1, "blue")],  # Custom scale from red (negative) to blue (positive)
#                   range_color=[filtered_df['xgi_diff_3_10'].min(), filtered_df['xgi_diff_3_10'].max()])  # Ensure full range of colors

#data1['layout'].update(title="Expected Goal Involovement",
#                       titlefont = dict(size =15), xaxis = dict(title="Avg. 10 Week XGI Per 90",titlefont=dict(size=12)),
#                       yaxis = dict(title = "XGI Per 90 Change (3 Wk. Avg. - 10 Wk. Avg.)", titlefont = dict(size=12),showgrid=False),
#                       width=600,
#                       height =500,
#                       coloraxis_showscale=False  
#) 

# Layout for placing the plot in the top-left corner using Streamlit columns
#col1, col2, col3 = st.columns(3)  # Create 3 columns

# Display the scatter plot in the first column (top-left)
#with col1:
#    st.plotly_chart(data1, use_container_width=False)  # Disable container width so custom width is used

                      
#st.plotly_chart(data1,use_container_width=True)   





                    
                       
           #color="xgi_diff_3_10", hover_name="player", facet_col="position_category",
           #log_x=True, size_max=45, range_x=[0.001,1.10], range_y=[-.80,0.80])
#fig.show()























#st.sidebar.header("Choose your filter")


















#st.write(df.describe())
#df['position_category'] = df['position'].apply(
    #lambda x: 'Goalkeeper' if 'GK' in x.split(',')[0]
    #else 'Defender' if 'B' in x.split(',')[0]
    #else 'Midfielder/Striker'
#)


#fig = px.scatter(df, x="avg_10_week_xgi", y="xgi_diff_3_10", animation_frame="gw_start_date", animation_group="player",
           #size="pop", 
           #color="xgi_diff_3_10", hover_name="player", facet_col="position_category",
           #log_x=True, size_max=45, range_x=[0.001,1.10], range_y=[-.80,0.80])
#fig.show()