import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="AQI & Weather Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('EDA_Main_Dataset_Weather_Enriched.csv')
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    return df

df = load_data()

# --- TAB SETUP ---
tab1, tab2, tab3 = st.tabs(["📊 General Visualization", "⚖️ City Comparison", "🔮 Future Prediction"])

# --- TAB 1: GENERAL VISUALIZATION ---
with tab1:
    st.header("General Air Quality & Weather Trends")
    
    # KPIs Calculation
    # We create a reset_index dataframe for easy plotting
    city_stats = df.groupby('City')['AQI Value'].mean().sort_values().reset_index()
    
    most_polluted_city = city_stats.iloc[-1]['City']
    most_polluted_val = city_stats.iloc[-1]['AQI Value']
    
    least_polluted_city = city_stats.iloc[0]['City']
    least_polluted_val = city_stats.iloc[0]['AQI Value']
    
    season_stats = df.groupby('Season')['AQI Value'].mean().sort_values().reset_index()
    most_polluted_season = season_stats.iloc[-1]['Season']
    least_polluted_season = season_stats.iloc[0]['Season']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Worst City", most_polluted_city, f"{most_polluted_val:.1f} AQI", delta_color="inverse")
    col2.metric("Best City", least_polluted_city, f"{least_polluted_val:.1f} AQI")
    col3.metric("Worst Season", most_polluted_season)
    col4.metric("Best Season", least_polluted_season)

    # Visualization Section
    c1, c2 = st.columns(2)
    with c1:
        # Taking top 10 for better visibility
        top_10_cities = city_stats.tail(10)
        fig_city = px.bar(
            top_10_cities, 
            x='AQI Value', 
            y='City', 
            orientation='h', 
            title="Top 10 Most Polluted Cities (Avg AQI)", 
            color='AQI Value',
            color_continuous_scale='Reds'
        )
        fig_city.update_layout(yaxis={'categoryorder':'total ascending'}) 
        st.plotly_chart(fig_city, use_container_width=True)
    
    with c2:
        fig_season = px.pie(season_stats, values='AQI Value', names='Season', title="AQI Distribution by Season", hole=0.4)
        st.plotly_chart(fig_season, use_container_width=True)

# --- TAB 2: CITY COMPARISON ---
with tab2:
    st.header("Side-by-Side Comparison")
    
    col_a, col_b, col_c = st.columns(3)
    city_1 = col_a.selectbox("Select City 1", options=sorted(df['City'].unique()), index=0)
    city_2 = col_b.selectbox("Select City 2", options=sorted(df['City'].unique()), index=1)
    metric = col_c.selectbox("Criteria for Comparison", 
                             options=['AQI Value', 'Avg_Temp_C', 'Humidity_Pct', 'Wind_Speed_kmh', 'Rainfall_mm'])

    comparison_df = df[df['City'].isin([city_1, city_2])]
    
    selected_season = st.multiselect("Filter by Season", options=df['Season'].unique(), default=df['Season'].unique())
    comparison_df = comparison_df[comparison_df['Season'].isin(selected_season)]

    fig_comp = px.histogram(comparison_df, x=metric, color='City', barmode='overlay', 
                            title=f"Distribution of {metric} between {city_1} and {city_2}", marginal="violin")
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Time Series Comparison
    fig_line = px.line(comparison_df.groupby(['date', 'City'])[metric].mean().reset_index(), 
                       x='date', y=metric, color='City', title=f"{metric} Over Time")
    st.plotly_chart(fig_line, use_container_width=True)

# --- TAB 3: FUTURE PREDICTION ---
with tab3:
    st.header("Predictive Insights (AQI & Weather)")
    
    @st.cache_resource
    def train_aqi_model(data):
        le_city = LabelEncoder()
        data['City_Encoded'] = le_city.fit_transform(data['City'])
        
        X = data[['City_Encoded', 'Year', 'Month', 'Day']]
        y = data[['AQI Value', 'Avg_Temp_C', 'Humidity_Pct']]
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model, le_city

    model, le_city = train_aqi_model(df.copy())

    p_col1, p_col2 = st.columns(2)
    predict_city = p_col1.selectbox("Select Region for Prediction", options=sorted(df['City'].unique()))
    predict_date = p_col2.date_input("Select Future Date", value=datetime.now())

    if st.button("Predict Environmental Metrics"):
        city_code = le_city.transform([predict_city])[0]
        prediction = model.predict([[city_code, predict_date.year, predict_date.month, predict_date.day]])
        
        res_aqi, res_temp, res_hum = prediction[0]
        
        st.subheader(f"Results for {predict_city} on {predict_date}")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted AQI", f"{int(res_aqi)}")
        r2.metric("Predicted Temperature", f"{res_temp:.1f} °C")
        r3.metric("Predicted Humidity", f"{res_hum:.1f} %")
        
        if res_aqi <= 50: st.success("Expected Air Quality: Good")
        elif res_aqi <= 100: st.info("Expected Air Quality: Satisfactory")
        elif res_aqi <= 200: st.warning("Expected Air Quality: Moderate")
        else: st.error("Expected Air Quality: Poor/Severe")