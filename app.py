import streamlit as st
import pandas as pd
import json
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore

DATA = "welcome_survey_simple_v2.csv"

MODEL_NAME = "welcome_survey_clustering_pipeline_v2"

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters


#st.title("Znajdź podobnych osób")

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ["<18", "25-34", "35-44", "45-54", "55-64", ">=65"])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])


    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])

model=get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
#st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data["description"])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))


#st.write("Wybrane dane:")
#st.dataframe(person_df, hide_index=True)
#st.write("Przykładowe osoby z bazy:")
#st.dataframe(all_df.sample(10), hide_index=True)


#st.header("Osoby z grupy")
#fig = px.histogram(same_cluster_df.sort_values("age"), x="age", color_discrete_sequence=['#636EFA'])
#fig.update_layout(
#    title="Rozkład wieku w grupie",
#    xaxis_title="Wiek",
#    yaxis_title="Liczba osób",
#)
#st.plotly_chart(fig)

st.header("Wiek i Płeć")
col1, col2 = st.columns(2)

# Wiek i płeć
with col1:
    
    fig = px.pie(same_cluster_df, 
        names="age",           # Kategorie (wiek)
        title="Rozkład wieku w grupie",
        hole=0.3,              # Opcjonalnie: robi z tego "donut chart"
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig)

with col2:
    fig = px.pie(
    same_cluster_df,
    names="gender",  # zamiast x
    color="gender",
    color_discrete_sequence=px.colors.qualitative.Safe
)

    fig.update_layout(
        title="Rozkład płci w grupie",
        showlegend=True
    )

    st.plotly_chart(fig)

# Wykształcenie
st.header("Wykształcenie")
fig = px.pie(
    same_cluster_df,
    names="edu_level",
    color="edu_level",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    hole=0.4  #to robi "dziurę"
)

fig.update_layout(
    title="Rozkład wykształcenia w grupie"
)

fig.update_traces(
    textinfo="percent+label"
)

st.plotly_chart(fig)


# Zwierzęta i odpoczynek
st.header("Ulubione Zwierzęta i Odpoczynek")
col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(same_cluster_df, x="fav_animals", color="fav_animals")
    fig.update_layout(
        title="Rozkład ulubionych zwierząt w grupie",
        xaxis_title="Ulubione zwierzęta",
        yaxis_title="Liczba osób",
        bargap=0.01,  # ZMNIEJSZ TEN NUMER (np. 0.1 lub 0.05), aby słupki były szersze
        showlegend=False 
    )

    st.plotly_chart(fig)


with col2:

    fig = px.histogram(same_cluster_df, x="fav_place", color_discrete_sequence=['#636EFA'])
    fig.update_layout(
        title="Rozkład ulubionych miejsc w grupie",
        xaxis_title="Ulubione miejsce",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)


