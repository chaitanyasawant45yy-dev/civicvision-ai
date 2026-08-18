import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CivicVision AI",
    page_icon="🏙️",
    layout="wide"
)


# =========================================================
# MODEL FINDER
# =========================================================

model_files = list(Path(".").glob("*.pt"))

if not model_files:
    st.error("YOLO model not found. Please upload a .pt model.")
    st.stop()

best_models = [
    f for f in model_files
    if "best" in f.name.lower()
]

MODEL_PATH = str(best_models[0] if best_models else model_files[0])


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# =========================================================
# HEADER
# =========================================================

st.title("🏙️ CivicVision AI")

st.subheader(
    "AI-Powered Public Infrastructure Monitoring & Smart Maintenance"
)

st.write(
    "Detect visible infrastructure problems, estimate civic risk, "
    "and prioritize maintenance using AI-assisted decision support."
)

st.divider()


# =========================================================
# INPUTS
# =========================================================

st.subheader("📍 Infrastructure Context")

col1, col2 = st.columns(2)

with col1:
    road_type = st.selectbox(
        "🛣️ Road Importance",
        [
            "Local Road",
            "Main Road",
            "Highway / Major Road"
        ]
    )

with col2:
    traffic_level = st.selectbox(
        "🚗 Traffic Level",
        [
            "Low",
            "Medium",
            "High"
        ]
    )


uploaded_file = st.file_uploader(
    "📷 Upload an infrastructure image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Infrastructure Image",
        use_container_width=True
    )

    st.divider()

    with st.spinner("🤖 CivicVision AI is analysing the infrastructure..."):

        results = model.predict(
            source=np.array(image),
            conf=0.25,
            verbose=False
        )

        result = results[0]

        # YOLO returns BGR image
        annotated_image = result.plot()

        # Convert BGR → RGB
        annotated_image = annotated_image[:, :, ::-1]

        boxes = result.boxes

        detection_count = len(boxes)

        # -------------------------------------------------
        # CONFIDENCE
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
        # DETECTED AREA
        # -------------------------------------------------

        image_width, image_height = image.size
        image_area = image_width * image_height

        total_detected_area = 0

        if detection_count > 0:

            xyxy = boxes.xyxy.cpu().numpy()

            for box in xyxy:

                x1, y1, x2, y2 = box

                width = max(0, x2 - x1)
                height = max(0, y2 - y1)

                total_detected_area += width * height

        damage_percentage = (
            total_detected_area / image_area * 100
            if image_area > 0
            else 0
        )

        # -------------------------------------------------
        # TRAFFIC SCORE
        # -------------------------------------------------

        traffic_score = {
            "Low": 5,
            "Medium": 15,
            "High": 25
        }[traffic_level]

        # -------------------------------------------------
        # ROAD IMPORTANCE SCORE
        # -------------------------------------------------

        road_score = {
            "Local Road": 5,
            "Main Road": 15,
            "Highway / Major Road": 25
        }[road_type]

        # -------------------------------------------------
        # AI DAMAGE SCORE
        # -------------------------------------------------

        confidence_score = average_confidence * 30

        area_score = min(damage_percentage * 2, 20)

        detection_score = min(detection_count * 5, 10)

        # -------------------------------------------------
        # SMART MAINTENANCE PRIORITY
        # -------------------------------------------------

        priority_score = round(
            confidence_score
            + area_score
            + detection_score
            + traffic_score
            + road_score
        )

        priority_score = max(
            0,
            min(100, priority_score)
        )

        # -------------------------------------------------
        # SEVERITY
        # -------------------------------------------------

        if priority_score >= 80:
            severity = "Critical"

        elif priority_score >= 60:
            severity = "High"

        elif priority_score >= 40:
            severity = "Medium"

        else:
            severity = "Low"

        # -------------------------------------------------
        # SAFETY RISK
        # -------------------------------------------------

        if priority_score >= 80:
            safety_risk = "Very High"

        elif priority_score >= 60:
            safety_risk = "High"

        elif priority_score >= 40:
            safety_risk = "Moderate"

        else:
            safety_risk = "Low"

        # -------------------------------------------------
        # RESPONSE TIME
        # -------------------------------------------------

        if priority_score >= 80:

            priority = "IMMEDIATE"

            response_time = "Within 6 hours"

            recommendation = (
                "Dispatch an inspection or emergency maintenance team "
                "immediately."
            )

        elif priority_score >= 60:

            priority = "HIGH"

            response_time = "Within 24–48 hours"

            recommendation = (
                "Schedule a physical inspection and maintenance "
                "within 24–48 hours."
            )

        elif priority_score >= 40:

            priority = "MEDIUM"

            response_time = "Within 7 days"

            recommendation = (
                "Add the location to the upcoming maintenance schedule."
            )

        else:

            priority = "LOW"

            response_time = "Monitor"

            recommendation = (
                "Monitor the location during the next infrastructure "
                "inspection."
            )


    # =====================================================
    # DETECTION RESULT
    # =====================================================

    st.subheader("🎯 AI Detection Result")

    st.image(
        annotated_image,
        caption="YOLO Infrastructure Detection",
        channels="RGB",
        use_container_width=True
    )


    # =====================================================
    # SMART PRIORITY DASHBOARD
    # =====================================================

    st.divider()

    st.subheader("🧠 Smart Civic Risk Assessment")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Maintenance Score",
            f"{priority_score}/100"
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


    # =====================================================
    # DETECTION DETAILS
    # =====================================================

    st.subheader("📊 AI Detection Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Issues Detected",
            detection_count
        )

    with col2:
        st.metric(
            "AI Confidence",
            f"{highest_confidence * 100:.1f}%"
        )

    with col3:
        st.metric(
            "Visible Damage",
            f"{damage_percentage:.2f}%"
        )


    # =====================================================
    # CONTEXT ANALYSIS
    # =====================================================

    st.subheader("🌐 Infrastructure Context")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("🛣️ **Road Importance**")
        st.write(road_type)

    with col2:
        st.write("🚗 **Traffic Level**")
        st.write(traffic_level)

    with col3:
        st.write("⏱️ **Recommended Response**")
        st.write(response_time)


    # =====================================================
    # WHY THE SCORE
    # =====================================================

    st.subheader("💡 Why this priority score?")

    st.write(
        f"CivicVision AI detected **{detection_count} visible issue(s)** "
        f"with an AI confidence of **{highest_confidence * 100:.1f}%**."
    )

    st.write(
        f"The estimated visible detected area is "
        f"**{damage_percentage:.2f}%** of the image."
    )

    st.write(
        f"The selected road importance is **{road_type}** and "
        f"traffic level is **{traffic_level}**."
    )

    st.write(
        f"Combining AI detection evidence with infrastructure context "
        f"produced a maintenance priority score of "
        f"**{priority_score}/100**."
    )


    # =====================================================
    # RECOMMENDED ACTION
    # =====================================================

    st.subheader("🛠️ Recommended Action")

    if priority == "IMMEDIATE":

        st.error(
            f"🚨 {recommendation}"
        )

    elif priority == "HIGH":

        st.warning(
            f"⚠️ {recommendation}"
        )

    else:

        st.info(
            f"ℹ️ {recommendation}"
        )


    # =====================================================
    # DECISION CARD
    # =====================================================

    st.divider()

    st.subheader("📋 Civic Decision Card")

    st.markdown(
        f"""
### 🚨 Infrastructure Issue Detected

**Maintenance Priority:** {priority}

**Priority Score:** {priority_score}/100

**Severity:** {severity}

**Safety Risk:** {safety_risk}

**AI Confidence:** {highest_confidence * 100:.1f}%

**Road Importance:** {road_type}

**Traffic Level:** {traffic_level}

**Visible Damage:** {damage_percentage:.2f}%

**Recommended Response:** {response_time}

### 🛠️ Action

{recommendation}
"""
    )


else:

    st.info(
        "👆 Upload a road or public infrastructure image to begin analysis."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CivicVision AI • AI-assisted infrastructure monitoring prototype"
)
            