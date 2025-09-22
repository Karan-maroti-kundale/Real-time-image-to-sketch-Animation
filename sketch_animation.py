import streamlit as st
from PIL import Image
import cv2
import numpy as np
import imageio

st.title("🖌️ Real-time Image to Pencil Sketch Animation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded_file:
    # Convert uploaded file to OpenCV format
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    max_dim = 800
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    sketch = np.ones_like(image) * 255
    frames = []

    for contour in contours:
        cv2.drawContours(sketch, [contour], -1, (0,0,0), 1)
        shading = cv2.divide(gray, 255 - cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY), scale=256)
        shading_bgr = cv2.cvtColor(shading, cv2.COLOR_GRAY2RGB)
        frames.append(shading_bgr)

    # Display final sketch
    st.image([Image.fromarray(image), Image.fromarray(frames[-1])], caption=["Original", "Sketch"], use_column_width=True)

    # Save GIF and provide download
    gif_path = "sketch_animation.gif"
    imageio.mimsave(gif_path, frames, duration=0.05)
    with open(gif_path, "rb") as f:
        st.download_button("📥 Download Sketch Animation GIF", f, file_name="sketch_animation.gif")
