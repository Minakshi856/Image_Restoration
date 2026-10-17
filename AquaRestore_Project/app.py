
import io
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AquaRestore | Image Restoration",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CUSTOM DESIGN
# ==================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap'
    );

    .stApp {
        background:
          radial-gradient(ellipse at 15% 0%, #103A4A 0%, transparent 36%),
          linear-gradient(145deg, #071722, #0A1D2A 55%, #0B2635);
        color: #E9F4F8;
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #F0FAFF !important;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #0A202D;
        border-right: 1px solid #20404D;
    }

    [data-testid="stSidebar"] * {
        color: #E3F2F7;
    }

    .hero {
        padding: 30px;
        border: 1px solid #245363;
        border-radius: 24px;
        background: linear-gradient(
            125deg,
            rgba(14, 91, 101, .45),
            rgba(13, 36, 53, .90) 65%
        );
        margin-bottom: 25px;
    }

    .eyebrow {
        color: #64E8D0;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero h1 {
        font-size: clamp(32px, 5vw, 52px);
        line-height: 1.12;
        margin: 12px 0;
    }

    .hero p {
        color: #B8D0DA;
        font-size: 16px;
        max-width: 760px;
        line-height: 1.7;
    }

    .panel {
        background: rgba(16, 43, 58, .84);
        border: 1px solid #234653;
        padding: 18px;
        border-radius: 18px;
        margin-bottom: 14px;
    }

    .panel-label {
        font-size: 12px;
        color: #86AAB9;
        letter-spacing: 1.5px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .metric-value {
        font-size: 23px;
        color: #70E8D5;
        font-weight: 700;
    }

    .muted {
        color: #A7C0CC;
        font-size: 13px;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(110deg, #14B8A6, #0D9488);
        color: #031B23;
        border: 0;
        border-radius: 11px;
        font-weight: 700;
        min-height: 44px;
        width: 100%;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #5EEAD4;
        color: #06232D;
        border: 0;
    }

    [data-testid="stFileUploader"] {
        background: rgba(11, 35, 49, .75);
        border: 1px dashed #3C7482;
        border-radius: 15px;
        padding: 10px;
    }

    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    hr {
        border-color: #234653;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# IMAGE ENHANCEMENT ALGORITHM
# ==================================================

def restore_underwater_image(
    image_rgb,
    color_strength=0.5,
    contrast_strength=2.0,
    sharpening_strength=0.4,
):
    """
    Enhance an RGB image using:
    1. Gray-world white balance.
    2. CLAHE local contrast enhancement.
    3. Unsharp masking.

    Returns an RGB NumPy array.
    """

    img = np.asarray(image_rgb, dtype=np.uint8)

    # Stage 1: Color correction
    if color_strength > 0:
        img_float = img.astype(np.float32)
        means = img_float.mean(axis=(0, 1))
        target = float(means.mean())

        gains = target / np.maximum(means, 1.0)
        gains = np.clip(gains, 0.7, 1.4)

        balanced = img_float * (
            1.0 - color_strength
            + color_strength * gains.reshape(1, 1, 3)
        )

        balanced = np.clip(
            balanced, 0, 255
        ).astype(np.uint8)
    else:
        balanced = img.copy()

    # Stage 2: CLAHE contrast enhancement
    lab = cv2.cvtColor(
        balanced, cv2.COLOR_RGB2LAB
    )
    lightness, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(contrast_strength),
        tileGridSize=(8, 8),
    )

    enhanced_lightness = clahe.apply(lightness)

    enhanced_lab = cv2.merge(
        [enhanced_lightness, channel_a, channel_b]
    )

    enhanced = cv2.cvtColor(
        enhanced_lab, cv2.COLOR_LAB2RGB
    )

    # Stage 3: Sharpening
    if sharpening_strength > 0:
        blurred = cv2.GaussianBlur(
            enhanced, (0, 0), sigmaX=1.0
        )

        enhanced = cv2.addWeighted(
            enhanced,
            1.0 + sharpening_strength,
            blurred,
            -sharpening_strength,
            0,
        )

    return np.clip(
        enhanced, 0, 255
    ).astype(np.uint8)


# ==================================================
# APPLICATION HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">
        🌊 COMPUTATIONAL PHOTOGRAPHY STUDIO
      </div>
      <h1>Bring the ocean<br>back to life.</h1>
      <p>
        Enhance underwater photographs affected by color distortion,
        haze and low contrast. Upload your image, tune the settings,
        compare results and download the restored photograph.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# SIDEBAR CONTROLS
# ==================================================

with st.sidebar:
    st.markdown("## 🌊 AquaRestore")
    st.caption("UNDERWATER IMAGE ENHANCEMENT")
    st.divider()

    st.markdown("### Enhancement controls")

    color_strength = st.slider(
        "Color correction",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
    )

    contrast_strength = st.slider(
        "Local contrast",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.5,
    )

    sharpening_strength = st.slider(
        "Detail sharpening",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
    )

    st.divider()
    st.markdown("### Supported formats")
    st.caption("JPG · JPEG · PNG")
    st.caption("Maximum recommended size: 20 megapixels")


# ==================================================
# IMAGE UPLOAD
# ==================================================

st.markdown("## 01 — Upload your photograph")

st.write(
    "Choose an underwater photograph to start the enhancement process."
)

uploaded_file = st.file_uploader(
    "Drop an image here or browse files",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-label">Ready for restoration</div>
          <h3>Explore the colors beneath the surface.</h3>
          <p class="muted">
            Upload a photograph to compare the original and enhanced
            versions. Use the controls in the sidebar to personalize
            your result.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-label">01 · Color</div>
              <h3>Balance</h3>
              <p class="muted">
                Reduce some blue-green color casts.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-label">02 · Contrast</div>
              <h3>Reveal</h3>
              <p class="muted">
                Improve local contrast with CLAHE.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-label">03 · Detail</div>
              <h3>Refine</h3>
              <p class="muted">
                Apply adjustable detail sharpening.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    try:
        with Image.open(uploaded_file) as opened:
            image = ImageOps.exif_transpose(
                opened
            ).convert("RGB")

        original = np.array(image)
        height, width = original.shape[:2]
        megapixels = height * width / 1_000_000

        if megapixels > 20:
            st.error(
                "This image exceeds 20 megapixels. "
                "Please resize it and upload it again."
            )
            st.stop()

        with st.spinner("Restoring your photograph..."):
            start_time = time.perf_counter()

            restored = restore_underwater_image(
                original,
                color_strength=color_strength,
                contrast_strength=contrast_strength,
                sharpening_strength=sharpening_strength,
            )

            processing_time = time.perf_counter() - start_time

        st.markdown("## 02 — Restoration results")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Image width", f"{width} px")

        with m2:
            st.metric("Image height", f"{height} px")

        with m3:
            st.metric("Image size", f"{megapixels:.2f} MP")

        with m4:
            st.metric("Processing time", f"{processing_time:.2f} s")

        original_col, restored_col = st.columns(2, gap="large")

        with original_col:
            st.markdown("### Original image")
            st.image(original, use_container_width=True)
            st.caption("Original photograph · Unmodified")

        with restored_col:
            st.markdown("### Restored image")
            st.image(restored, use_container_width=True)
            st.caption("Color correction · CLAHE · Sharpening")

        st.markdown("## 03 — Download your result")

        output = io.BytesIO()
        Image.fromarray(restored).save(
            output, format="PNG"
        )

        st.download_button(
            label="⬇ Download restored image",
            data=output.getvalue(),
            file_name="aquarestored_image.png",
            mime="image/png",
            use_container_width=True,
        )

        st.caption(
            "Traditional image processing improves appearance but "
            "cannot guarantee recovery of true colors or lost details."
        )

    except Exception as exc:
        st.error(
            "Unable to process this image. "
            "Please try a different JPG or PNG photograph."
        )

        with st.expander("Technical details"):
            st.code(str(exc))


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:16px 0;">
      <div style="color:#64E8D0; font-weight:700;">
        AQUARESTORE
      </div>
      <div style="color:#91ADBA; font-size:12px; margin-top:6px;">
        Underwater Image Restoration · Assignment 2
      </div>
      <div style="color:#7897A5; font-size:11px; margin-top:5px;">
        Traditional computer vision · Educational project
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
