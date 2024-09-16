import streamlit as st
import plotly.express as px
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb


st.set_page_config(layout='wide')

@st.cache_data
def get_data():
    
    df = pd.read_csv('master.csv')
    
    df = df.drop(columns=['country-year', 'generation', 'HDI for year', ' gdp_for_year ($) '])
    
    return df

def get_model():
    model = joblib.load('lightgbm_model.pkl')
    return model


st.header(':red[Suicide Rate]')

selected_page = st.sidebar.radio("Select a page", ["Home", "Graphics", "Prediction"])

df = get_data()
df_ = df.sample(250)
df_sorted = df.groupby('country')['suicides/100k pop'].mean().sort_values(ascending=False)
df_sorted = df_sorted.reset_index()
df_sorted.index = df_sorted.index + 1


if selected_page == "Home":
   
    column_sig, column_sig2 = st.columns(2)

    with column_sig:
        st.subheader('Research Data')
        st.markdown('**Content**\n\n'
            'This compiled dataset pulled from four other datasets linked by time and place, and was built to find signals correlated to increased suicide rates among different cohorts globally, across the socio-economic spectrum.\n\n'
            '**References**\n\n'
            'All relevant data sources and references for this dataset are available in the Kaggle dataset: [Suicide Rates Overview 1985 to 2016](https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016)')
        st.subheader('Project Objective')
        st.markdown('The goal of this project is to study suicide rates around the world and address a common misconception about northern countries. Northern countries, especially Finland, are often thought to have very high suicide rates. However, this project aims to use data analysis to show that this belief is not true')
        st.subheader('Features')
        st.markdown('''
- :red[country:] The name of the country where the data was collected.
- :red[year:] The year in which the suicide data was recorded.
- :red[sex:] The gender of the individuals, categorized as male or female.
- :red[age:] The age group of the individuals, typically divided into categories like 15-24, 25-34, etc
- :red[suicides_no:] The total number of suicides in a given year for the specified population.
- :red[population:] The total population of the country or demographic group in the given year.
- :red[suicides/100k pop:] The number of suicides per 100,000 people, used to standardize the rate across different population sizes.
- :red[gdp_per_capita ($):] The average economic output per person, measured in US dollars, for the given country in that year.                 
                    ''')


    with column_sig2:
        st.subheader('Random Sample of Global Suicide Rates Data')
        st.dataframe(df_)
        st.subheader('Top Countries by Suicide Rate per 100k Population')
        st.dataframe(df_sorted)


elif selected_page == "Graphics":

    st.subheader("Suicide Rate of Chosen Country by the Years")
    year_select_for_map = st.slider("Years", min_value=int(df['year'].min()), max_value=int(df['year'].max()), step=1)

    df_filtered = df[df['year'] == year_select_for_map]
    df_grouped = df_filtered.groupby('country', as_index=False).agg({'suicides/100k pop': 'mean'})

    fig = px.choropleth(df_grouped, 
                        locations="country",
                        locationmode="country names",
                        color="suicides/100k pop",
                        hover_name="country",
                        color_continuous_scale=px.colors.sequential.Viridis, 
                        labels={'suicides/100k pop': 'Suicides per 100k'})

   
    st.plotly_chart(fig, use_container_width=True)

    column_sig, column_sig2 = st.columns(2)

    selected_country = column_sig.selectbox("Select a country", df['country'].unique(), index=list(df['country'].unique()).index('Finland'))

    df_country = df[df['country'] == selected_country]
    years = df_country['year'].unique()
    column_sig.subheader(f'Suicide Rates in {selected_country} based on GDP per Capita')


    fig, ax1 = plt.subplots(figsize=(14, 8))

    sns.histplot(data=df_country, x='year', weights='gdp_per_capita ($)', color='blue', label='GDP per Capita', kde=False, bins=30, alpha=0.6, ax=ax1)
    sns.histplot(data=df_country, x='year', weights='suicides/100k pop', color='red', label='Suicide Rate per 100k', kde=False, bins=30, alpha=0.6, ax=ax1)

    ax2 = ax1.twinx()  
    sns.lineplot(data=df_country, x='year', y='suicides/100k pop', color='red', marker='o', label='Suicide Rate (Line)', ax=ax2)

    
    ax1.set_title(f'GDP per Capita vs Suicide Rate over the Years in {selected_country}', fontsize=14)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('GDP per Capita / Suicide Rate (Histogram)', fontsize=12)
    ax2.set_ylabel('Suicide Rate per 100k (Line)', fontsize=12)

    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    column_sig.pyplot(plt)


    df_sorted = df_grouped.sort_values(by='suicides/100k pop', ascending=False)
    column_sig2.subheader(f'Top Countries by Suicide Rate per 100k Population in {year_select_for_map}')
    column_sig2.dataframe(df_sorted)


elif selected_page == "Prediction":
    model = get_model()
    feature_names = joblib.load('feature_names.pkl')
   
    country = st.selectbox("Select Your Country", df['country'].unique(), help="Select the country for which you want to predict the suicide rate.")
    year = st.number_input("Select Year", min_value=2017, max_value=2050, step=1, value=2020, help="Select a year for which you want to predict the suicide rate.")
    sex = st.selectbox("Select Gender", ['male', 'female'], help="The gender of the individuals.")
    age = st.selectbox("Select Age Group", ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years'], help="The age group of the individuals.")
    population = st.number_input("Enter Population", min_value=1000, max_value=500000000, step=10000, value=1000000, help="Enter the total population of the country or demographic group.")
    gdp_per_capita = st.number_input("Enter GDP per Capita ($)", min_value=100.0, max_value=100000.0, step=5000.0, value=20000.0, help="Enter the average economic output per person, in US dollars.")

    
    input_data = {'year': [year], 'population': [population], 'gdp_per_capita ($)': [gdp_per_capita], 'sex': [sex], 'age': [age], 'country': [country]}
    input_df = pd.DataFrame(input_data)

    
    input_df = pd.get_dummies(input_df)

    
    input_df = input_df.reindex(columns=feature_names, fill_value=0)


    if st.button("Predict Suicide Rate"):
        
        prediction = model.predict(input_df)
        st.write(f"Predicted suicide rate for {sex}, {age} age group in {year} in {country}: {prediction[0]:.2f}")

