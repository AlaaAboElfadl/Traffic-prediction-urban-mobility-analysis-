import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


# =========================================================
# Configuration
# =========================================================

st.set_page_config(
    page_title="Urban Mobility Analysis",
    page_icon="🚕",
    layout="wide"
)

ARTIFACTS_PATH = "streamlit_artifacts"
DATA_PATH = "cab_rides.csv"


# =========================================================
# Load Artifacts
# =========================================================

@st.cache_resource
def load_artifacts():

    xgb_model = joblib.load(
        os.path.join(
            ARTIFACTS_PATH,
            "xgb_model.pkl"
        )
    )

    scaler = joblib.load(
        os.path.join(
            ARTIFACTS_PATH,
            "price_scaler.pkl"
        )
    )

    encoder = joblib.load(
        os.path.join(
            ARTIFACTS_PATH,
            "onehot_encoder.pkl"
        )
    )

    source_encoder = joblib.load(
        os.path.join(
            ARTIFACTS_PATH,
            "source_encoder.pkl"
        )
    )

    destination_encoder = joblib.load(
        os.path.join(
            ARTIFACTS_PATH,
            "destination_encoder.pkl"
        )
    )

    kmeans_model = joblib.load(
        os.path.join(
            ARTIFACTS_PATH,
            "kmeans_model.pkl"
        )
    )

    cluster_scaler = joblib.load(
        os.path.join(
            ARTIFACTS_PATH,
            "cluster_scaler.pkl"
        )
    )

    mobility_profiles = joblib.load(
        os.path.join(
            ARTIFACTS_PATH,
            "mobility_profiles.pkl"
        )
    )

    location_coordinates = joblib.load(
        os.path.join(
            ARTIFACTS_PATH,
            "location_coordinates.pkl"
        )
    )

    model_features = joblib.load(
        os.path.join(
            ARTIFACTS_PATH,
            "model_features.pkl"
        )
    )

    return (
        xgb_model,
        scaler,
        encoder,
        source_encoder,
        destination_encoder,
        kmeans_model,
        cluster_scaler,
        mobility_profiles,
        location_coordinates,
        model_features
    )


(
    xgb_model,
    scaler,
    encoder,
    source_encoder,
    destination_encoder,
    kmeans_model,
    cluster_scaler,
    mobility_profiles,
    location_coordinates,
    model_features
) = load_artifacts()


# =========================================================
# Load Data
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_PATH
    )

    data = data.iloc[
        :100000
    ].copy()

    return data


data = load_data()

analysis_data = data.dropna(
    subset=["price"]
).copy()


# =========================================================
# Temporal Features
# =========================================================

analysis_data["time_stamp"] = pd.to_datetime(
    analysis_data["time_stamp"],
    unit="ms"
)

analysis_data["year"] = (
    analysis_data["time_stamp"].dt.year
)

analysis_data["month"] = (
    analysis_data["time_stamp"].dt.month
)

analysis_data["day"] = (
    analysis_data["time_stamp"].dt.day
)

analysis_data["day_of_week"] = (
    analysis_data["time_stamp"].dt.dayofweek
)

analysis_data["hour"] = (
    analysis_data["time_stamp"].dt.hour
)

analysis_data["is_rushhour"] = (
    analysis_data["hour"]
    .isin([7, 8, 9, 16, 17, 18])
    .astype(int)
)

analysis_data["is_weekend"] = (
    analysis_data["day_of_week"]
    .isin([5, 6])
    .astype(int)
)


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title(
    "🚕 Urban Mobility Analysis"
)

section = st.sidebar.radio(
    "Select a Section",
    [
        "🏠 Overview",
        "💰 Price Prediction",
        "🚦 Mobility Analysis",
        "🗺️ Geographic Analysis",
        "⏰ Temporal Analysis",
        "📈 Surge Analysis"
    ]
)


# =========================================================
# Overview
# =========================================================

if section == "🏠 Overview":

    st.title(
        "Traffic Prediction & Urban Mobility Analysis"
    )

    st.write(
        "Uber & Lyft Ride Data Analysis and Price Prediction"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Ride Records",
            "100,000"
        )

    with col2:
        st.metric(
            "Final Model",
            "XGBoost"
        )

    with col3:
        st.metric(
            "Test R²",
            "0.921"
        )

    with col4:
        st.metric(
            "KMeans Clusters",
            "10"
        )

    st.subheader(
        "Final Model Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            "$1.40"
        )

    with col2:
        st.metric(
            "RMSE",
            "$2.61"
        )

    with col3:
        st.metric(
            "R²",
            "92.1%"
        )


# =========================================================
# Price Prediction
# =========================================================

elif section == "💰 Price Prediction":

    st.title(
        "💰 Ride Price Prediction"
    )

    locations = sorted(
        location_coordinates.keys()
    )

    cab_types = list(
        encoder.categories_[0]
    )

    ride_names = list(
        encoder.categories_[1]
    )

    source = st.selectbox(
        "Source",
        locations
    )

    destination = st.selectbox(
        "Destination",
        locations
    )

    cab_type = st.selectbox(
        "Cab Type",
        cab_types
    )

    ride_name = st.selectbox(
        "Ride Type",
        ride_names
    )

    distance = st.number_input(
        "Distance",
        min_value=0.0,
        value=2.0
    )

    hour = st.slider(
        "Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    day_of_week = st.slider(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=2
    )

    month = st.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=12
    )

    day = st.slider(
        "Day",
        min_value=1,
        max_value=31,
        value=15
    )

    is_rushhour = int(
        hour in [7, 8, 9, 16, 17, 18]
    )

    is_weekend = int(
        day_of_week in [5, 6]
    )

    is_holiday = 0


    # -----------------------------------------------------
    # Geographic Coordinates
    # -----------------------------------------------------

    source_latitude, source_longitude = (
        location_coordinates[source]
    )

    destination_latitude, destination_longitude = (
        location_coordinates[destination]
    )


    # -----------------------------------------------------
    # Haversine Distance
    # -----------------------------------------------------

    def haversine(
        lat1,
        lon1,
        lat2,
        lon2
    ):

        lat1, lon1, lat2, lon2 = map(
            np.radians,
            [
                lat1,
                lon1,
                lat2,
                lon2
            ]
        )

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2) ** 2
            +
            np.cos(lat1)
            *
            np.cos(lat2)
            *
            np.sin(dlon / 2) ** 2
        )

        return (
            2
            * 6371
            * np.arcsin(
                np.sqrt(a)
            )
        )


    distance_in_km = haversine(
        source_latitude,
        source_longitude,
        destination_latitude,
        destination_longitude
    )


    # -----------------------------------------------------
    # Encode Source and Destination
    # -----------------------------------------------------

    source_encoded = source_encoder.transform(
        [[source]]
    )[0][0]

    destination_encoded = destination_encoder.transform(
        [[destination]]
    )[0][0]


    # -----------------------------------------------------
    # One-Hot Encode Cab Type and Ride Name
    # -----------------------------------------------------

    categorical_input = encoder.transform(
        [[
            cab_type,
            ride_name
        ]]
    )

    # The encoder uses sparse_output=False,
    # so the result is already a NumPy array.

    categorical_array = categorical_input[0]

    categorical_features = pd.DataFrame(
        [categorical_array],
        columns=encoder.get_feature_names_out(
            [
                "cab_type",
                "name"
            ]
        )
    )


    # -----------------------------------------------------
    # Build Prediction Data
    # -----------------------------------------------------

    prediction_data = pd.DataFrame({
        "distance": [distance],
        "source_latitude": [source_latitude],
        "source_longitude": [source_longitude],
        "destination_latitude": [destination_latitude],
        "destination_longitude": [destination_longitude],
        "month": [month],
        "day": [day],
        "day_of_week": [day_of_week],
        "hour": [hour],
        "is_rushhour": [is_rushhour],
        "is_weekend": [is_weekend],
        "is_holiday": [is_holiday],
        "distance_in_km": [distance_in_km],
        "source_encoded": [source_encoded],
        "destination_encoded": [destination_encoded]
    })

    prediction_data = pd.concat(
        [
            prediction_data,
            categorical_features
        ],
        axis=1
    )

    prediction_data = prediction_data.reindex(
        columns=model_features,
        fill_value=0
    )


    # -----------------------------------------------------
    # Scale Features
    # -----------------------------------------------------

    prediction_scaled = scaler.transform(
        prediction_data
    )

    prediction_scaled = pd.DataFrame(
        prediction_scaled,
        columns=model_features
    )


    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    if st.button(
        "Predict Price"
    ):

        predicted_price = xgb_model.predict(
            prediction_scaled
        )[0]

        st.success(
            f"Predicted Ride Price: ${predicted_price:.2f}"
        )

        st.info(
            f"Haversine Distance: {distance_in_km:.2f} km"
        )


    # =====================================================
    # Price Analysis Charts
    # =====================================================

    st.subheader(
        "📊 Price Analysis"
    )

    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # Price Distribution
    # -----------------------------------------------------

    with col1:

        st.write(
            "Price Distribution"
        )

        price_distribution = (
            analysis_data["price"]
            .value_counts()
            .sort_index()
        )

        st.line_chart(
            price_distribution
        )


    # -----------------------------------------------------
    # Average Price by Cab Type
    # -----------------------------------------------------

    with col2:

        st.write(
            "Average Price by Cab Type"
        )

        avg_price_by_cab = (
            analysis_data
            .groupby("cab_type")["price"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            avg_price_by_cab
        )


    # -----------------------------------------------------
    # Average Price by Ride Type
    # -----------------------------------------------------

    st.write(
        "Average Price by Ride Type"
    )

    avg_price_by_ride = (
        analysis_data
        .groupby("name")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )

    st.bar_chart(
        avg_price_by_ride
    )


# =========================================================
# Mobility Analysis
# =========================================================

elif section == "🚦 Mobility Analysis":

    st.title(
        "🚦 Mobility Analysis"
    )

    st.subheader(
        "Mobility Profiles"
    )

    st.dataframe(
        mobility_profiles,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Average Distance by Cluster"
        )

        st.bar_chart(
            mobility_profiles[
                "average_distance_km"
            ]
        )

    with col2:

        st.subheader(
            "Average Price by Cluster"
        )

        st.bar_chart(
            mobility_profiles[
                "average_price"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Most Common Hour"
        )

        st.bar_chart(
            mobility_profiles[
                "most_common_hour"
            ]
        )

    with col2:

        st.subheader(
            "Number of Rides"
        )

        st.bar_chart(
            mobility_profiles[
                "number_of_rides"
            ]
        )


# =========================================================
# Geographic Analysis
# =========================================================

elif section == "🗺️ Geographic Analysis":

    st.title(
        "🗺️ Geographic Analysis"
    )

    geographic_data = (
        analysis_data
        .groupby("source")
        .agg(
            ride_count=("source", "count"),
            average_price=("price", "mean")
        )
        .reset_index()
    )

    geographic_data = geographic_data.sort_values(
        "ride_count",
        ascending=False
    )

    map_data = []

    for location in geographic_data["source"]:

        if location in location_coordinates:

            latitude, longitude = (
                location_coordinates[location]
            )

            map_data.append({
                "latitude": latitude,
                "longitude": longitude
            })

    map_df = pd.DataFrame(
        map_data
    )

    st.subheader(
        "Pickup Location Map"
    )

    if not map_df.empty:

        st.map(
            map_df
        )

    st.subheader(
        "Top 10 Pickup Locations"
    )

    st.dataframe(
        geographic_data.head(10),
        use_container_width=True
    )


# =========================================================
# Temporal Analysis
# =========================================================

elif section == "⏰ Temporal Analysis":

    st.title(
        "⏰ Temporal Analysis"
    )

    rides_by_hour = (
        analysis_data["hour"]
        .value_counts()
        .sort_index()
    )

    st.subheader(
        "Rides by Hour"
    )

    st.line_chart(
        rides_by_hour
    )

    rides_by_day = (
        analysis_data["day_of_week"]
        .value_counts()
        .sort_index()
    )

    st.subheader(
        "Rides by Day of Week"
    )

    st.bar_chart(
        rides_by_day
    )

    rides_by_month = (
        analysis_data["month"]
        .value_counts()
        .sort_index()
    )

    st.subheader(
        "Rides by Month"
    )

    st.bar_chart(
        rides_by_month
    )


# =========================================================
# Surge Analysis
# =========================================================

elif section == "📈 Surge Analysis":

    st.title(
        "📈 Surge Analysis"
    )

    st.subheader(
        "Surge Multiplier Distribution"
    )

    surge_distribution = (
        analysis_data["surge_multiplier"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        surge_distribution
    )

    avg_surge_by_hour = (
        analysis_data
        .groupby("hour")["surge_multiplier"]
        .mean()
    )

    st.subheader(
        "Average Surge by Hour"
    )

    st.line_chart(
        avg_surge_by_hour
    )

    avg_surge_by_service = (
        analysis_data
        .groupby("cab_type")["surge_multiplier"]
        .mean()
    )

    st.subheader(
        "Average Surge by Service"
    )

    st.bar_chart(
        avg_surge_by_service
    )