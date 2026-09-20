import streamlit as st
import requests
import pandas as pd
import io


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "http://localhost:8000"


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Network Security",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #e8f5e9;
        color: #2e7d32;
        margin-bottom: 20px;
    }

    .danger-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffebee;
        color: #c62828;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🛡️ Network Security</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Network traffic classification using Machine Learning</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Controls")

    st.write("Backend API")

    st.code(API_URL)

    st.divider()

    st.subheader("Model Training")

    if st.button(
        "🚀 Train Model",
        use_container_width=True
    ):

        with st.spinner("Training model..."):

            try:

                response = requests.get(
                    f"{API_URL}/train",
                    timeout=600
                )

                if response.status_code == 200:

                    st.success(
                        "Model training completed successfully!"
                    )

                else:

                    st.error(
                        f"Training failed: {response.text}"
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Could not connect to FastAPI: {e}"
                )


# --------------------------------------------------
# Prediction Section
# --------------------------------------------------

st.header("📊 Network Traffic Prediction")

st.write(
    "Upload a CSV file containing network traffic features "
    "to generate predictions."
)


uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


# --------------------------------------------------
# File Preview
# --------------------------------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write(
        f"Rows: **{df.shape[0]}**  |  "
        f"Columns: **{df.shape[1]}**"
    )


    # --------------------------------------------------
    # Prediction Button
    # --------------------------------------------------

    if st.button(
        "🔍 Predict",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Running prediction..."):

            try:

                # Reset file pointer
                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv"
                    )
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=300
                )


                # ------------------------------------------
                # Successful response
                # ------------------------------------------

                if response.status_code == 200:

                    prediction_data = response.json()

                    result_df = pd.DataFrame(
                        prediction_data
                    )

                    st.success(
                        "Prediction completed successfully!"
                    )


                    # ------------------------------------------
                    # Results
                    # ------------------------------------------

                    st.subheader("🎯 Prediction Results")

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )


                    # ------------------------------------------
                    # Prediction Statistics
                    # ------------------------------------------

                    if "predicted_column" in result_df.columns:

                        st.subheader(
                            "📈 Prediction Summary"
                        )

                        prediction_counts = (
                            result_df[
                                "predicted_column"
                            ]
                            .value_counts()
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Total Samples",
                            len(result_df)
                        )

                        col2.metric(
                            "Unique Predictions",
                            len(prediction_counts)
                        )

                        col3.metric(
                            "Most Common",
                            str(
                                prediction_counts
                                .index[0]
                            )
                        )


                    # ------------------------------------------
                    # Download
                    # ------------------------------------------

                    csv_data = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Predictions",
                        data=csv_data,
                        file_name="predicted_output.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


                # ------------------------------------------
                # API Error
                # ------------------------------------------

                else:

                    st.error(
                        f"Prediction failed.\n\n"
                        f"Status Code: {response.status_code}\n\n"
                        f"{response.text}"
                    )


            except requests.exceptions.RequestException as e:

                st.error(
                    f"Could not connect to FastAPI: {e}"
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Network Security ML Pipeline • FastAPI + Streamlit"
)