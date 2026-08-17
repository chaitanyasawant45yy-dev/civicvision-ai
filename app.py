import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="CivicVision AI",
    page_icon="🏙️",
    layout="wide"
)


# ---------------------------------------------------------
# FIND YOLO MODEL AUTOMATICALLY
# ---------------------------------------------------------

model_files = list(Path(".").glob("*.pt"))

if not model_files:
    st.error(
        "YOLO model not found. Please upload your trained .pt model "
        "to the repository."
    )
    st.stop()

# Prefer files containing "best"
best_models = [f for f in model_files if "best" in f.name.lower()]

if best_models:
    MODEL_PATH = str(best_models[0])
else:
    MODEL_PATH = str(model_files[0])


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🏙️ CivicVision AI")

st.subheader(
    "AI-Powered Public Infrastructure Monitoring & Risk Assessment"
)

st.write(
    "Upload an infrastructure image and CivicVision AI will detect "
    "visible issues, estimate civic risk and recommend maintenance priority."
)

st.divider()


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload an infrastructure image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Infrastructure Image",
        use_container_width=True
    )

    st.divider()

    with st.spinner("🤖 CivicVision AI is analysing the image..."):

        results = model.predict(
            source=np.array(image),
            conf=0.25,
            verbose=False
        )

        result = results[0]

        annotated_image = result.plot()

        boxes = result.boxes

        detection_count = len(boxes)

        # -------------------------------------------------
        # DETECTION INFORMATION
        # -------------------------------------------------

        confidences = []

        if detection_count > 0:
            confidences = boxes.conf.cpu().numpy().tolist()

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0
        )

        highest_confidence = (
            max(confidences)
            if confidences
            else 0
        )

        # -------------------------------------------------
        # ESTIMATE DETECTED AREA
        # -------------------------------------------------

        image_width, image_height = image.size
        image_area = image_width * image_height

        total_detected_area = 0

        if detection_count > 0:

            xyxy = boxes.xyxy.cpu().numpy()

            for box in xyxy:

                x1, y1, x2, y2 = box

                box_width = max(0, x2 - x1)
                box_height = max(0, y2 - y1)

                total_detected_area += box_width * box_height

        damage_percentage = (
            total_detected_area / image_area * 100
            if image_area > 0
            else 0
        )

        # -------------------------------------------------
        # CIVIC IMPACT SCORE
        # -------------------------------------------------

        confidence_score = average_confidence * 40

        area_score = min(damage_percentage * 2, 30)

        object_score = min(detection_count * 10, 20)

        impact_score = round(
            confidence_score +
            area_score +
            object_score
        )

        impact_score = max(0, min(100, impact_score))

        # -------------------------------------------------
        # SEVERITY
        # -------------------------------------------------

        if impact_score >= 75:
            severity = "Critical"
        elif impact_score >= 55:
            severity = "High"
        elif impact_score >= 30:
            severity = "Medium"
        else:
            severity = "Low"

        # -------------------------------------------------
        # MAINTENANCE PRIORITY
        # -------------------------------------------------

        if severity == "Critical":
            priority = "Immediate"
            response = "Dispatch maintenance team immediately."

        elif severity == "High":
            priority = "High"
            response = "Schedule repair within 24–48 hours."

        elif severity == "Medium":
            priority = "Medium"
            response = "Schedule physical inspection and plan maintenance."

        else:
            priority = "Low"
            response = "Monitor the location during the next inspection."

        # -------------------------------------------------
        # SAFETY RISK
        # -------------------------------------------------

        if impact_score >= 75:
            safety_risk = "Very High"
        elif impact_score >= 55:
            safety_risk = "High"
        elif impact_score >= 30:
            safety_risk = "Moderate"
        else:
            safety_risk = "Low"


    # -----------------------------------------------------
    # AI DETECTION RESULT
    # -----------------------------------------------------

    st.subheader("🎯 AI Detection Result")

    st.image(
        annotated_image,
        caption="YOLO Infrastructure Detection",
    channels="RGB",
    use_container_width=True
)


    # -----------------------------------------------------
    # CIVIC DASHBOARD
    # -----------------------------------------------------

    st.divider()

    st.subheader("🧠 Civic Risk Assessment")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Civic Impact",
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

    with col4:
        st.metric(
            "Safety Risk",
            safety_risk
        )


    # -----------------------------------------------------
    # DETECTION DETAILS
    # -----------------------------------------------------

    st.subheader("📊 Detection Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Objects Detected",
            detection_count
        )

    with col2:
        st.metric(
            "Best Confidence",
            f"{highest_confidence * 100:.1f}%"
        )

    with col3:
        st.metric(
            "Estimated Damage Area",
            f"{damage_percentage:.2f}%"
        )


    # -----------------------------------------------------
    # PRIORITY EXPLANATION
    # -----------------------------------------------------

    st.subheader("💡 Why this score?")

    if detection_count == 0:

        st.info(
            "No infrastructure issue was detected with the current "
            "confidence threshold. Consider checking the image quality "
            "or using another image."
        )

    else:

        st.write(
            f"CivicVision AI detected **{detection_count} visible issue(s)** "
            f"with an average confidence of "
            f"**{average_confidence * 100:.1f}%**."
        )

        st.write(
            f"The estimated detected area is approximately "
            f"**{damage_percentage:.2f}%** of the image."
        )

        st.write(
            f"Based on these indicators, the current civic impact score "
            f"is **{impact_score}/100**, resulting in a "
            f"**{severity} severity** assessment."
        )


    # -----------------------------------------------------
    # RECOMMENDED ACTION
    # -----------------------------------------------------

    st.subheader("🛠️ Recommended Action")

    if priority == "Immediate":

        st.error(
            f"🚨 {response}"
        )

    elif priority == "High":

        st.warning(
            f"⚠️ {response}"
        )

    else:

        st.info(
            f"ℹ️ {response}"
        )


    # -----------------------------------------------------
    # CIVIC DECISION SUMMARY
    # -----------------------------------------------------

    st.divider()

    st.subheader("📋 Civic Decision Summary")

    summary = f"""
**Issue Status:** {"Detected" if detection_count > 0 else "Not Detected"}

**Civic Impact Score:** {impact_score}/100

**Severity:** {severity}

**Maintenance Priority:** {priority}

**Safety Risk:** {safety_risk}

**Detection Confidence:** {highest_confidence * 100:.1f}%

**Estimated Visible Damage:** {damage_percentage:.2f}%

**Recommended Response:** {response}
"""

    st.markdown(summary)

else:

    st.info(
        "👆 Upload a road or public infrastructure image to begin analysis."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "CivicVision AI • AI-assisted infrastructure monitoring prototype"
)
            