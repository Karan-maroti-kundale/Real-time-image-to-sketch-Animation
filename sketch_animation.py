import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imageio

st.title("🖌️ Real-time Image to Pencil Sketch Animation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    # Preprocessing
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    sketch = np.ones_like(img_np) * 255
    frames = []

    for contour in contours:
        cv2.drawContours(sketch, [contour], -1, (0,0,0), 1)
        shading = cv2.divide(gray, 255 - cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY), scale=256)
        shading_bgr = cv2.cvtColor(shading, cv2.COLOR_GRAY2BGR)
        frames.append(shading_bgr)

    # Display final sketch
    st.image([Image.fromarray(img_np), Image.fromarray(frames[-1])], caption=["Original","Sketch"], use_column_width=True)

    # Save GIF for download
    gif_path = "sketch_animation.gif"
    imageio.mimsave(gif_path, frames, fps=15)
    with open(gif_path,"rb") as f:
        st.download_button("📥 Download Sketch Animation GIF", f, file_name="sketch_animation.gif")
