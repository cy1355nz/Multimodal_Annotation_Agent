"""
Streamlit web interface for Multimodal Annotation Agent.
Provides interactive UI for annotators to input descriptions and images.
"""
import os
from PIL import Image
import streamlit as st
from agent.annotation_agent import AnnotationAgent
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

# Page configuration
st.set_page_config(
    page_title="Multimodal Annotation Agent",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 Multimodal Annotation Agent for Autonomous Driving")
st.markdown("---")

# Initialize session state
if "agent" not in st.session_state:
    st.session_state["agent"] = AnnotationAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if "pending_state" not in st.session_state:
    st.session_state["pending_state"] = None

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")

    # Output directory setting
    output_dir = st.text_input(
        "Output Directory",
        value=get_abs_path(agent_conf.get("output_dir", "data/output"))
    )

    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state["messages"] = []
        st.session_state["last_result"] = None
        st.rerun()

    st.divider()
    st.markdown("""
    ### 📖 Instructions
    1. Enter natural language description of the driving scene
    2. Upload 1-3 images (optional)
    3. Generate a structured draft
    4. Review, approve, or provide feedback
    5. Download the approved JSON result
    """)

# Main input area
st.header("📝 Scene Description")

# Text input for description
description = st.text_area(
    "Enter your scene description:",
    placeholder="Example: Sunny day, ego vehicle driving on urban expressway with a white foam box in the right lane ahead. Ego vehicle should slow down and avoid.",
    height=150
)

st.header("🎞️ Optional Clip Metadata")
clip_cols = st.columns(2)
with clip_cols[0]:
    clip_id = st.text_input(
        "Clip ID",
        placeholder="Example: clip_urban_red_light_001",
    )
with clip_cols[1]:
    timestamp = st.text_input(
        "Timestamp",
        placeholder="Example: 1521055565282267",
    )

# File uploader for images
st.header("🖼️ Upload Images")
uploaded_files = st.file_uploader(
    "Upload up to 3 images",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
    help="Upload 1-3 images of the driving scene"
)

# Validate file count
if uploaded_files and len(uploaded_files) > 3:
    st.error("❌ Maximum 3 images allowed!")
    uploaded_files = uploaded_files[:3]

# Display uploaded images
if uploaded_files:
    cols = st.columns(min(len(uploaded_files)*3, 9))
    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx]:
            st.image(uploaded_file, caption=f"Image {idx + 1}", width='content')

# Submit button
if st.button("🚀 Generate Draft", type="primary", disabled=not description):
    if description:
        # Add user message to history
        st.session_state["messages"].append({
            "role": "user",
            "content": description
        })

        # Save uploaded files temporarily
        image_paths = []
        if uploaded_files:
            temp_dir = get_abs_path("data/temp")
            os.makedirs(temp_dir, exist_ok=True)

            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_paths.append(temp_path)

        # Generate annotation draft for review
        with st.spinner("🤖 Agent is analyzing the scene..."):
            try:
                with st.chat_message("assistant"):
                    def capture(generator):
                        for chunk in generator:
                            if isinstance(chunk, str) and chunk:
                                yield chunk

                    stream = st.session_state["agent"].execute_stream(
                        description=description,
                        image_paths=image_paths if image_paths else None,
                        clip_id=clip_id.strip() or None,
                        timestamp=timestamp.strip() or None,
                    )
                    st.write_stream(capture(stream))

                    state = st.session_state["agent"].last_state
                    st.session_state["pending_state"] = state

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "Draft generated and waiting for human review."
                })
                st.success("✅ Draft generated. Please review before saving.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"Sorry, an error occurred: {str(e)}"
                })

pending_state = st.session_state.get("pending_state")
if pending_state:
    st.markdown("---")
    st.header("🔎 Human Review")
    st.caption(f"Review status: {pending_state.review_status}")
    st.json(pending_state.result.model_dump(mode="json"))

    feedback = st.text_area(
        "Reviewer feedback",
        placeholder="Example: The traffic light should be green, and ego_vehicle.longitudinal_action should be maintain_speed.",
        height=100,
        key="review_feedback"
    )

    review_cols = st.columns(2)
    with review_cols[0]:
        if st.button("✅ Approve & Save", type="primary"):
            try:
                saved_state = st.session_state["agent"].approve_and_save(pending_state)
                st.session_state["pending_state"] = None
                st.session_state["last_result"] = saved_state.output_path
                st.success(f"✅ Approved and saved to: {saved_state.output_path}")
            except Exception as e:
                st.error(f"❌ Save failed: {str(e)}")

    with review_cols[1]:
        if st.button("🛠️ Revise with Feedback", disabled=not feedback.strip()):
            status = st.status("Feedback received. Revising annotation...", expanded=True)
            status.write("MemoryAgent recorded your feedback.")
            try:
                with st.spinner("AnnotationWriterAgent is revising the JSON and QualityAgent is validating it..."):
                    revised_state = st.session_state["agent"].revise_with_feedback(pending_state, feedback)
                st.session_state["pending_state"] = revised_state
                status.write("QualityAgent validation passed. Draft is ready for review again.")
                status.update(label="Revision completed", state="complete")
                st.success("✅ Revised draft generated. Please review again.")
                st.rerun()
            except Exception as e:
                status.update(label="Revision failed", state="error")
                st.error(f"❌ Revision failed: {str(e)}")

if st.session_state.get("last_result") and os.path.exists(st.session_state["last_result"]):
    with open(st.session_state["last_result"], "rb") as f:
        st.download_button(
            label="⬇️ Download latest JSON",
            data=f,
            file_name=os.path.basename(st.session_state["last_result"]),
            mime="application/json"
        )
