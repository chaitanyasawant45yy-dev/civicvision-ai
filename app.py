import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="CivicVision AI",
    page_icon="🏙️",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🏙️ CivicVision AI")
st.subheader("AI-Powered Public Infrastructure Monitoring System")

st.write(
    "Upload an image and CivicVision AI will analyze the scene, "
    "estimate civic impact, and recommend an appropriate action."
)

st.divider()

# -----------------------------
# LOAD AI MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best (1).pt")

model = load_model()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📷 Upload an infrastructure image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Analyze Image"):

        with st.spinner("CivicVision AI is analyzing the image..."):

            results = model.predict(
                source=image,
                conf=0.25
            )

        st.success("✅ Analysis completed")

        # -----------------------------
        # AI DETECTION RESULT
        # -----------------------------
        result_image = results[0].plot()

        st.image(
            result_image,
            caption="AI Detection Result",
            use_container_width=True
        )

        # -----------------------------
        # DETECTIONS
        # -----------------------------
        detections = []

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            detections.append({
                "name": class_name,
                "confidence": confidence
            })

        st.divider()

        # -----------------------------
        # CIVIC IMPACT ENGINE
        # -----------------------------
        st.header("🧠 Civic Impact Assessment")

        detection_count = len(detections)

        if detection_count == 0:

            impact_score = 20
            severity = "Low"
            priority = "Normal"

        elif detection_count <= 2:

            impact_score = 45
            severity = "Medium"
            priority = "High"

        else:

            impact_score = 75
            severity = "High"
            priority = "Urgent"

        # -----------------------------
        # SCORE DISPLAY
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Civic Impact Score",
                f"{impact_score}/100"
            )

        with col2:
            st.metric(
                "Severity",
                severity
            )

        with col3:
            st.metric(
                "Priority",
                priority
            )

        # -----------------------------
        # EXPLANATION
        # -----------------------------
        st.subheader("💡 Why this score?")

        st.write(
            f"CivicVision detected {detection_count} object(s) "
            "in the uploaded image. The current prototype uses "
            "the number of detected objects as an initial indicator "
            "for civic impact."
        )

        # -----------------------------
        # DETECTION DETAILS
        # -----------------------------
        st.subheader("📊 Detection Details")

        if detections:

            for item in detections:

                st.write(
                    f"**{item['name']}** — "
                    f"Confidence: {item['confidence']:.2%}"
                )

        else:

            st.info(
                "No objects were detected by the current AI model."
            )

        # -----------------------------
        # RECOMMENDATION
        # -----------------------------
        st.subheader("🛠️ Recommended Action")

        if priority == "Urgent":

            st.warning(
                "Conduct an immediate physical inspection "
                "and prioritize the location for maintenance review."
            )

        elif priority == "High":

            st.info(
                "Schedule a physical inspection and evaluate "
                "whether maintenance action is required."
            )

        else:

            st.success(
                "No immediate action is suggested by this prototype. "
                "Continue routine monitoring."
            )

        st.divider()

        st.caption(
            "CivicVision AI is a prototype decision-support system. "
            "AI results should be verified by qualified personnel "
            "before real-world action."
        )
            