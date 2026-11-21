import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="GeoClimate AI",
    page_icon="🌍",
    layout="wide"
)

# -------------------- LOAD DATA --------------------

try:
    df = pd.read_csv("geoclimate_data.csv")
except FileNotFoundError:
    st.error("geoclimate_data.csv not found. Keep it in the project folder.")
    st.stop()

required_columns = [
    "Country", "State", "District", "Year", "Month",
    "Rainfall_mm", "Temperature_C", "NDVI", "Forest_Area_km2"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing CSV columns: {missing_columns}")
    st.stop()

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Month"] = pd.to_numeric(df["Month"], errors="coerce")

# -------------------- CONFIGURATION --------------------

INDIA_STATES = {
    "Andhra Pradesh": ["All districts", "Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna", "Kurnool", "Nellore", "Prakasam", "Srikakulam", "Visakhapatnam", "West Godavari", "YSR Kadapa"],
    "Arunachal Pradesh": ["All districts", "Tawang", "West Kameng", "Papum Pare", "Changlang"],
    "Assam": ["All districts", "Baksa", "Barpeta", "Cachar", "Dibrugarh", "Kamrup", "Sonitpur"],
    "Bihar": ["All districts", "Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga"],
    "Chhattisgarh": ["All districts", "Raipur", "Durg", "Bilaspur", "Bastar"],
    "Goa": ["All districts", "North Goa", "South Goa"],
    "Gujarat": ["All districts", "Ahmedabad", "Surat", "Vadodara", "Rajkot", "Kutch"],
    "Haryana": ["All districts", "Gurugram", "Faridabad", "Hisar", "Karnal"],
    "Himachal Pradesh": ["All districts", "Shimla", "Kangra", "Mandi", "Kullu"],
    "Jharkhand": ["All districts", "Ranchi", "Dhanbad", "Bokaro", "Jamshedpur"],
    "Karnataka": [
        "All districts", "Bagalkot", "Ballari", "Belagavi",
        "Bengaluru Rural", "Bengaluru Urban", "Bidar",
        "Chamarajanagar", "Chikkaballapur", "Chikkamagaluru",
        "Chitradurga", "Dakshina Kannada", "Davanagere",
        "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi",
        "Kodagu", "Kolar", "Koppal", "Mandya", "Mysuru",
        "Raichur", "Ramanagara", "Shivamogga", "Tumakuru",
        "Udupi", "Uttara Kannada", "Vijayanagara",
        "Vijayapura", "Yadgir"
    ],
    "Kerala": ["All districts", "Thiruvananthapuram", "Kollam", "Alappuzha", "Kottayam", "Ernakulam", "Thrissur", "Kozhikode", "Kannur", "Wayanad"],
    "Madhya Pradesh": ["All districts", "Bhopal", "Indore", "Jabalpur", "Gwalior", "Ujjain"],
    "Maharashtra": ["All districts", "Mumbai", "Pune", "Nagpur", "Nashik", "Kolhapur", "Satara"],
    "Manipur": ["All districts", "Imphal East", "Imphal West", "Thoubal", "Churachandpur"],
    "Meghalaya": ["All districts", "East Khasi Hills", "West Garo Hills", "Ri-Bhoi"],
    "Mizoram": ["All districts", "Aizawl", "Lunglei", "Champhai"],
    "Nagaland": ["All districts", "Kohima", "Dimapur", "Mokokchung"],
    "Odisha": ["All districts", "Khordha", "Cuttack", "Ganjam", "Puri", "Sambalpur"],
    "Punjab": ["All districts", "Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda"],
    "Rajasthan": ["All districts", "Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"],
    "Sikkim": ["All districts", "Gangtok", "Namchi", "Mangan"],
    "Tamil Nadu": ["All districts", "Chennai", "Coimbatore", "Madurai", "Salem", "Thanjavur"],
    "Telangana": ["All districts", "Hyderabad", "Warangal", "Karimnagar", "Nizamabad", "Khammam"],
    "Tripura": ["All districts", "West Tripura", "Gomati", "Dhalai"],
    "Uttar Pradesh": ["All districts", "Lucknow", "Kanpur Nagar", "Varanasi", "Agra", "Prayagraj", "Gorakhpur"],
    "Uttarakhand": ["All districts", "Dehradun", "Haridwar", "Nainital", "Almora"],
    "West Bengal": ["All districts", "Kolkata", "Howrah", "Darjeeling", "Jalpaiguri", "Purba Bardhaman"]
}

CROPS = [
    "Rice", "Wheat", "Sugarcane", "Maize", "Cotton", "Ragi",
    "Jowar", "Bajra", "Groundnut", "Soybean", "Tur / Pigeon Pea",
    "Green Gram", "Black Gram", "Chickpea", "Sunflower",
    "Mustard", "Potato", "Tomato", "Onion", "Chilli",
    "Turmeric", "Coffee", "Tea", "Coconut", "Arecanut",
    "Banana", "Mango", "Millets", "Pulses", "Vegetables", "Other"
]

SEASONS = ["Summer", "Winter", "Rainy Season"]

SEASON_MONTHS = {
    "Summer": [3, 4, 5, 6],
    "Winter": [11, 12, 1, 2],
    "Rainy Season": [6, 7, 8, 9]
}

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# -------------------- HELPERS --------------------

def safe_percent_change(current, reference):
    if reference == 0:
        return None
    return ((current - reference) / abs(reference)) * 100


def parse_values(raw, label, expected=12):
    try:
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]

        if len(values) != expected:
            st.error(
                f"{label}: Please enter exactly {expected} monthly values."
            )
            return None

        return values

    except ValueError:
        st.error(f"Enter valid numeric values for {label}.")
        return None


def monthly_chart_data(data, column, value_name):
    result = data.groupby("Month", as_index=False)[column].mean()
    result["Month Name"] = result["Month"].map(MONTH_NAMES)
    result = result.sort_values("Month")
    result = result.rename(columns={column: value_name})
    return result


# -------------------- SIDEBAR --------------------

st.sidebar.title("🌱 GeoClimate AI")

st.sidebar.subheader("Location Selection")

region_level = st.sidebar.selectbox(
    "Select Region Level",
    ["India", "State", "District"]
)

if region_level == "India":
    region_name = "India"
    state_name = "All India"
    district_name = "All districts"

elif region_level == "State":
    state_name = st.sidebar.selectbox(
        "Select State",
        list(INDIA_STATES.keys())
    )
    region_name = state_name
    district_name = "All districts"

else:
    state_name = st.sidebar.selectbox(
        "Select State",
        list(INDIA_STATES.keys())
    )

    district_name = st.sidebar.selectbox(
        "Select District",
        INDIA_STATES[state_name]
    )

    region_name = (
        state_name
        if district_name == "All districts"
        else f"{district_name}, {state_name}"
    )

st.sidebar.subheader("Agriculture Selection")

crop = st.sidebar.selectbox("Select Crop", CROPS)
season = st.sidebar.selectbox("Select Season", SEASONS)

available_years = sorted(
    df["Year"].dropna().astype(int).unique().tolist()
)

if not available_years:
    st.error("No valid year values found in the CSV file.")
    st.stop()

analysis_year = st.sidebar.selectbox(
    "Select Analysis Year",
    available_years
)

st.sidebar.caption(
    "Current environmental input fields are demonstration values. "
    "Replace them with verified satellite and weather data for real analysis."
)

# -------------------- ENVIRONMENTAL INPUTS --------------------

st.sidebar.subheader("Rainfall Inputs")

rainfall_normal = st.sidebar.number_input(
    "Normal seasonal rainfall (mm)",
    min_value=0.0,
    value=800.0,
    step=10.0
)

rainfall_daily_total = st.sidebar.number_input(
    "Current seasonal rainfall total (mm)",
    min_value=0.0,
    value=600.0,
    step=10.0
)

rain_days = st.sidebar.number_input(
    "Number of daily observations",
    min_value=1,
    value=90,
    step=1
)

st.sidebar.subheader("Temperature Inputs")

normal_temp = st.sidebar.number_input(
    "Reference mean temperature (°C)",
    value=28.0,
    step=0.5
)

temperature_values = st.sidebar.text_input(
    "Monthly mean temperatures (°C)",
    "27,28,29,30,29,28,27,27,28,29,28,27"
)

st.sidebar.subheader("Vegetation Inputs")

ndvi_raw = st.sidebar.text_input(
    "Monthly NDVI values",
    "0.42,0.45,0.50,0.55,0.61,0.68,0.72,0.70,0.66,0.60,0.52,0.46"
)

qa_valid = st.sidebar.number_input(
    "QA-valid observations",
    min_value=0,
    value=10,
    step=1
)

qa_total = st.sidebar.number_input(
    "Total QA observations",
    min_value=1,
    value=12,
    step=1
)

ndvi_threshold = st.sidebar.slider(
    "Healthy vegetation NDVI threshold",
    0.0, 1.0, 0.50, 0.05
)

st.sidebar.subheader("Deforestation Inputs")

forest_ndvi = st.sidebar.number_input(
    "Current forest-area NDVI",
    min_value=-1.0,
    max_value=1.0,
    value=0.35,
    step=0.05
)

forest_threshold = st.sidebar.slider(
    "Forest NDVI threshold",
    0.0, 1.0, 0.40, 0.05
)

landcover_is_forest = st.sidebar.checkbox(
    "Land-cover mask identifies forest",
    value=True
)

forest_area = st.sidebar.number_input(
    "Study area (hectares)",
    min_value=0.0,
    value=1000.0,
    step=50.0
)

# -------------------- FILTER DATA --------------------

filtered_df = df.copy()

if state_name != "All India":
    filtered_df = filtered_df[
        filtered_df["State"] == state_name
    ]

if district_name != "All districts":
    filtered_df = filtered_df[
        filtered_df["District"] == district_name
    ]

filtered_df = filtered_df[
    filtered_df["Year"] == analysis_year
]

# -------------------- MAIN PAGE --------------------

st.title("🌍 GeoClimate AI")
st.subheader("Integrated Smart Agriculture Advisory System")

st.write(
    "Environmental monitoring using rainfall, temperature, "
    "vegetation and deforestation indicators."
)

st.info(
    "Prototype mode: CSV records and manually entered environmental "
    "values are demonstration data, not verified real-time measurements."
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Selected Region", region_name)
c2.metric("Crop", crop)
c3.metric("Season", season)
c4.metric("Analysis Year", analysis_year)

if filtered_df.empty:
    st.warning(
        f"Data is not available yet for {region_name} in {analysis_year}. "
        "Currently, the sample CSV contains data only for Belagavi "
        "and Mysuru in Karnataka."
    )

    st.stop()

st.markdown("---")
st.header("Environmental Monitoring")

analyze = st.button(
    "🔍 Analyze Selected Region",
    type="primary",
    use_container_width=True
)

# -------------------- ANALYSIS --------------------

if analyze:

    temp_vals = parse_values(
        temperature_values,
        "Temperature"
    )

    ndvi_vals = parse_values(
        ndvi_raw,
        "NDVI"
    )

    if temp_vals is None or ndvi_vals is None:
        st.stop()

    if qa_valid > qa_total:
        st.error(
            "QA-valid observations cannot exceed total observations."
        )
        st.stop()

    # 1. RAINFALL ANALYSIS

    daily_rainfall = np.full(
        int(rain_days),
        rainfall_daily_total / int(rain_days)
    )

    temporal_total = float(np.sum(daily_rainfall))
    spatial_mean = float(np.mean(daily_rainfall))

    rainfall_change = safe_percent_change(
        temporal_total,
        rainfall_normal
    )

    # 2. TEMPERATURE ANALYSIS

    monthly_c = np.array(temp_vals, dtype=float)
    monthly_k = monthly_c + 273.15

    monthly_days = np.array([
        31, 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
    ])

    day_weighted_mean_k = float(
        np.average(monthly_k, weights=monthly_days)
    )

    annual_mean_c = day_weighted_mean_k - 273.15

    temp_std = float(np.std(monthly_c))
    temp_min = float(np.min(monthly_c))
    temp_max = float(np.max(monthly_c))

    x = np.arange(1, len(monthly_c) + 1)

    temp_slope = float(
        np.polyfit(x, monthly_c, 1)[0]
    )

    if temp_slope > 0.02:
        temp_trend = "Increasing"
    elif temp_slope < -0.02:
        temp_trend = "Decreasing"
    else:
        temp_trend = "Approximately stable"

    # 3. VEGETATION ANALYSIS

    qa_fraction = qa_valid / qa_total

    if qa_valid > 0:
        monthly_ndvi = np.clip(
            np.array(ndvi_vals, dtype=float),
            -1,
            1
        )

        annual_ndvi = float(np.mean(monthly_ndvi))

        vegetation_class = (
            "Healthy vegetation"
            if annual_ndvi >= ndvi_threshold
            else "Low / sparse vegetation"
        )

    else:
        monthly_ndvi = np.array([])
        annual_ndvi = float("nan")
        vegetation_class = "Insufficient QA-valid observations"

    # 4. DEFORESTATION SCREENING

    forest_candidate = forest_ndvi >= forest_threshold

    forest_detected = bool(
        forest_candidate and landcover_is_forest
    )

    forest_status = (
        "Forest-like cover detected"
        if forest_detected
        else "Possible vegetation/forest loss or non-forest land cover"
    )

    # 5. SEASONAL ANALYSIS

    selected_months = SEASON_MONTHS[season]

    selected_month_idx = [
        month - 1 for month in selected_months
    ]

    seasonal_temp = float(
        np.mean(monthly_c[selected_month_idx])
    )

    if monthly_ndvi.size:
        seasonal_ndvi = float(
            np.mean(monthly_ndvi[selected_month_idx])
        )
    else:
        seasonal_ndvi = float("nan")

    # -------------------- RESULTS --------------------

    st.markdown("---")
    st.header("📊 Analysis Results")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Seasonal Rainfall",
        f"{temporal_total:.1f} mm"
    )

    r2.metric(
        "Annual Mean Temperature",
        f"{annual_mean_c:.1f} °C"
    )

    r3.metric(
        "Mean NDVI",
        f"{annual_ndvi:.3f}"
        if np.isfinite(annual_ndvi)
        else "N/A"
    )

    r4.metric(
        "QA-valid Fraction",
        f"{qa_fraction * 100:.1f}%"
    )

    # -------------------- RAINFALL --------------------

    st.subheader("🌧️ 1. Rainfall Monitoring")

    if rainfall_change is None:
        st.warning(
            "Rainfall change cannot be calculated because normal rainfall is zero."
        )
    else:
        direction = (
            "increase"
            if rainfall_change >= 0
            else "decrease"
        )

        st.write(
            f"Rainfall shows a {abs(rainfall_change):.1f}% "
            f"{direction} compared with the entered normal rainfall."
        )

    st.write(
        f"Total entered rainfall: {temporal_total:.1f} mm "
        f"across {int(rain_days)} observations."
    )

    st.write(
        f"Mean rainfall per observation: {spatial_mean:.2f} mm."
    )

    st.write("Monthly rainfall data from CSV:")

    rainfall_df = filtered_df[
        filtered_df["Month"].isin(selected_months)
    ]

    if rainfall_df.empty:
        st.warning(
            f"No rainfall records available for {season}."
        )
    else:
        rain_chart = monthly_chart_data(
            rainfall_df,
            "Rainfall_mm",
            "Rainfall (mm)"
        )

        st.dataframe(
            rain_chart,
            hide_index=True,
            use_container_width=True
        )

        st.plotly_chart(
            px.bar(
                rain_chart,
                x="Month Name",
                y="Rainfall (mm)",
                title=f"Monthly Rainfall - {region_name}"
            ),
            use_container_width=True
        )

    # Annual rainfall

    annual_rainfall_df = (
        filtered_df.groupby("Year", as_index=False)["Rainfall_mm"]
        .sum()
    )

    annual_rainfall_df = annual_rainfall_df.rename(
        columns={"Rainfall_mm": "Annual Rainfall (mm)"}
    )

    st.subheader("Annual Rainfall")

    st.dataframe(
        annual_rainfall_df,
        hide_index=True,
        use_container_width=True
    )

    # -------------------- TEMPERATURE --------------------

    st.subheader("🌡️ 2. Temperature Monitoring")

    st.write(
        f"Day-weighted annual mean: {annual_mean_c:.2f} °C."
    )

    st.write(
        f"Minimum: {temp_min:.1f} °C | "
        f"Maximum: {temp_max:.1f} °C | "
        f"Standard deviation: {temp_std:.2f} °C"
    )

    st.write(
        f"Temperature trend: {temp_trend} "
        f"(slope {temp_slope:.3f} °C/month)."
    )

    temp_df = filtered_df[
        ["Month", "Temperature_C"]
    ].copy()

    if temp_df.empty:
        st.warning("Temperature data is not available.")
    else:
        temp_chart = monthly_chart_data(
            temp_df,
            "Temperature_C",
            "Temperature (°C)"
        )

        st.plotly_chart(
            px.line(
                temp_chart,
                x="Month Name",
                y="Temperature (°C)",
                markers=True,
                title=f"Monthly Temperature - {region_name}"
            ),
            use_container_width=True
        )

    # -------------------- VEGETATION --------------------

    st.subheader("🌿 3. Vegetation Monitoring")

    st.write(
        f"QA masking: {qa_valid} valid observations "
        f"out of {qa_total} ({qa_fraction * 100:.1f}%)."
    )

    st.write(
        f"Vegetation classification: {vegetation_class}."
    )

    ndvi_df = filtered_df[
        ["Month", "NDVI"]
    ].copy()

    if ndvi_df.empty:
        st.warning("Vegetation data is not available.")
    else:
        ndvi_chart = monthly_chart_data(
            ndvi_df,
            "NDVI",
            "NDVI"
        )

        st.plotly_chart(
            px.line(
                ndvi_chart,
                x="Month Name",
                y="NDVI",
                markers=True,
                title=f"Monthly Vegetation - {region_name}"
            ),
            use_container_width=True
        )

    # -------------------- DEFORESTATION --------------------

    st.subheader("🌳 4. Deforestation Detection")

    st.write(
        f"NDVI threshold test: "
        f"{'Pass' if forest_candidate else 'Below threshold'}."
    )

    st.write(
        f"Land-cover mask: "
        f"{'Forest' if landcover_is_forest else 'Not forest'}."
    )

    st.write(
        f"Screening result: {forest_status}."
    )

    forest_screening_area = filtered_df[
        "Forest_Area_km2"
    ].mean()

    st.metric(
        "Recorded Forest Area",
        f"{forest_screening_area:.2f} km²"
    )

    st.caption(
        "This is a screening demonstration. Actual deforestation "
        "detection requires comparing classified forest pixels "
        "across different dates."
    )

    # -------------------- SMART AGRICULTURE ADVISORY --------------------

    st.markdown("---")
    st.header("🌾 Smart Agriculture Advisory")

    advice = []

    if rainfall_change is not None and rainfall_change < -20:
        advice.append(
            "Rainfall is substantially below the reference. "
            "Check local soil moisture and irrigation availability."
        )

    elif rainfall_change is not None and rainfall_change > 20:
        advice.append(
            "Rainfall is substantially above the reference. "
            "Check field drainage and waterlogging conditions."
        )

    else:
        advice.append(
            "Rainfall is near the entered reference range. "
            "Continue monitoring local rainfall."
        )

    if seasonal_temp > normal_temp + 2:
        advice.append(
            "Temperature is above the reference. Monitor possible heat stress."
        )

    elif seasonal_temp < normal_temp - 2:
        advice.append(
            "Temperature is below the reference. Check crop-specific temperature requirements."
        )

    else:
        advice.append(
            "Temperature is close to the entered reference value."
        )

    if np.isfinite(annual_ndvi):
        if annual_ndvi < ndvi_threshold:
            advice.append(
                "Vegetation index is below the selected threshold. "
                "Inspect crop condition and moisture availability."
            )
        else:
            advice.append(
                "Vegetation index is above the selected threshold. "
                "Continue periodic crop monitoring."
            )

    if not forest_detected:
        advice.append(
            "The forest screening test did not identify forest-like cover. "
            "Verify using multi-date satellite imagery."
        )

    for item in advice:
        st.write("• " + item)

    # -------------------- DOWNLOAD REPORT --------------------

    report = pd.DataFrame({
        "Parameter": [
            "Region",
            "State",
            "District",
            "Crop",
            "Season",
            "Analysis Year",
            "Seasonal Rainfall (mm)",
            "Normal Rainfall (mm)",
            "Rainfall Change (%)",
            "Annual Mean Temperature (°C)",
            "Temperature Trend",
            "Annual Mean NDVI",
            "Vegetation Classification",
            "Forest Screening Status"
        ],

        "Value": [
            region_name,
            state_name,
            district_name,
            crop,
            season,
            analysis_year,
            round(temporal_total, 2),
            round(rainfall_normal, 2),
            "Not available"
            if rainfall_change is None
            else round(rainfall_change, 2),
            round(annual_mean_c, 2),
            temp_trend,
            "Not available"
            if not np.isfinite(annual_ndvi)
            else round(annual_ndvi, 3),
            vegetation_class,
            forest_status
        ]
    })

    st.subheader("📥 Download Analysis Report")

    st.download_button(
        "Download CSV Report",
        report.to_csv(index=False).encode("utf-8"),
        file_name="GeoClimateAI_Advisory_Report.csv",
        mime="text/csv"
    )

    st.dataframe(
        report.astype(str),
        hide_index=True,
        use_container_width=True
    )

else:

    st.info(
        "Choose the region, crop, season and environmental inputs "
        "in the sidebar, then click Analyze Selected Region."
    )

    st.subheader("Project Modules")

    m1, m2, m3, m4 = st.columns(4)

    m1.markdown(
        "*🌧️ Rainfall*\n\n"
        "Rainfall aggregation and change calculation."
    )

    m2.markdown(
        "*🌡️ Temperature*\n\n"
        "Temperature conversion, statistics and trend analysis."
    )

    m3.markdown(
        "*🌿 Vegetation*\n\n"
        "NDVI monitoring and vegetation classification."
    )

    m4.markdown(
        "*🌳 Deforestation*\n\n"
        "NDVI threshold and land-cover screening."
    )