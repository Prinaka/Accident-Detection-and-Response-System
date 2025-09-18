import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

LOG_FILE = "collision_log.csv"

st.set_page_config(page_title="Collision Dashboard", layout="wide")
st.title("Vehicle Collision Detection Dashboard")

@st.cache_data
def load_logs():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    else:
        return pd.DataFrame(columns=["time", "vehicle1", "vehicle2", "bbox1", "bbox2", "frame_path"])

logs = load_logs()

if 'selected_frame' not in st.session_state:
    st.session_state.selected_frame = None

left_col, mid_col, right_col = st.columns([1, 0.2, 1])

with left_col:
    st.subheader("Recent Collision History")
    if not logs.empty:
        history = logs.tail(10).copy()
        history = history.reset_index(drop=True)
        cols_to_display = ["time", "vehicle1", "vehicle2"]
        history_df = history[cols_to_display].copy()
        
        st.dataframe(
            history_df.style.set_properties(**{'font-size': '10pt'}),
            height=300,
            use_container_width=True,
            )

        st.subheader("Select a frame to view:")
        option_list = history.apply(lambda row: f"Collision at {row['time']} ({row['vehicle1']} vs {row['vehicle2']})", axis=1).tolist()
        selected_option = st.selectbox("Choose a collision:", options=option_list, index=0)
        selected_index = option_list.index(selected_option)
        st.session_state.selected_frame = history.iloc[selected_index]['frame_path']
        
    else:
        st.info("No collision history available yet.")


with right_col:
    st.subheader("Collision Frame Viewer")
    if st.session_state.selected_frame and os.path.exists(str(st.session_state.selected_frame)):
        st.image(st.session_state.selected_frame, caption="Selected Collision Frame", use_container_width=True)
    else:
        st.warning("⚠️Frame not found or not selected.")
        
    st.subheader("Collision Analytics")
    if not logs.empty:
        logs["hour"] = pd.to_datetime(logs["time"]).dt.hour
        counts = logs.groupby("hour").size()
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax, color="crimson", alpha=0.7)
        ax.set_title("Collisions per Hour", fontsize=14, weight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Number of Collisions")
        st.pyplot(fig)
    else:
        st.info("No analytics to show yet.")