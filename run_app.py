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
    3. Click 'Submit' to generate structured annotation
    4. Download the JSON result
    """)

# Main input area
st.header("📝 Scene Description")

# Text input for description
description = st.text_area(
    "Enter your scene description:",
    placeholder="Example: Sunny day, ego vehicle driving on urban expressway with a white foam box in the right lane ahead. Ego vehicle should slow down and avoid.",
    height=150
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
if st.button("🚀 Generate Annotation", type="primary", disabled=not description):
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

        # Generate annotation
        with st.spinner("🤖 Agent is analyzing the scene..."):
            # try:
            # capture weird empty list
            def capture(generator):
                for chunk in generator:
                    if not isinstance(chunk, str):
                        continue
                    if not chunk or "=====" in chunk or chunk.strip() == "[]":
                        continue
                    response_messages.append(chunk)
                    yield chunk

            # Execute agent
            response_messages = []
            res_stream = st.session_state["agent"].execute_stream(
                description=description,
                image_paths=image_paths if image_paths else None
            )
            with st.chat_message("assistant"):
                response_text = st.write_stream(capture(res_stream))

            # Add assistant message to history
            st.session_state["messages"].append({
                "role": "assistant",
                "content": response_messages[-1] if response_messages else "Annotation completed"
            })
            st.success("✅ Annotation completed successfully!")

            # except Exception as e:
            #     st.error(f"❌ Error: {str(e)}")
            #     st.session_state["messages"].append({
            #         "role": "assistant",
            #         "content": f"Sorry, an error occurred: {str(e)}"
            #     })
            #     st.rerun()
