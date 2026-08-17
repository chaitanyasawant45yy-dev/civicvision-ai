import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="CivicVision AI",
    page_icon="🏙️",
    layout="wide"
)

# Title
st.title("🏙️ CivicVision AI")
st.subheader("AI-Powered Public Infrastructure Monitoring System")

st.write(
    "Upload an infrastructure image to detect potential public infrastructure issues using computer vision."
)

# Load AI model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "📷 Upload an infrastructure image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Infrastructure Image",
        use_container_width=True
    )

    if st.button("🔍 Analyze Infrastructure"):

        with st.spinner("AI is analyzing the infrastructure..."):

            results = model.predict(
                source=image,
                conf=0.25
            )

        st.success("✅ Analysis completed!")

        # Detection image
        result_image = results[0].plot()

        st.image(
            result_image,
            caption="AI Detection Result",
            use_container_width=True
        )

        st.subheader("📊 Detection Results")

        detections = 0

        for box in results[0].boxes:

            detections += 1

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            st.write(
                f"**{class_name}** — "
                f"Confidence: {confidence:.2%}"
            )

        if detections == 0:
            st.info(
                "No objects were detected by the current AI model."
            