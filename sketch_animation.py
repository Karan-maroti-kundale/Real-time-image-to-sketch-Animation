import cv2
import numpy as np

image = cv2.imread('Demo_input.jpg')
if image is None:
    print("Image not found! Check the filename and path.")
    exit()

max_dim = 800
h, w = image.shape[:2]
if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    image = cv2.resize(image, (int(w*scale), int(h*scale)))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

edges = cv2.Canny(blur, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

sketch = np.ones_like(image) * 255

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Demo_animation.mp4', fourcc, 30, (sketch.shape[1], sketch.shape[0]))

for i, contour in enumerate(contours):
    cv2.drawContours(sketch, [contour], -1, (0,0,0), 1)
    shading = cv2.divide(gray, 255 - cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY), scale=256)
    shading_bgr = cv2.cvtColor(shading, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Pencil Sketch Animation', shading_bgr)
    out.write(shading_bgr)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.imshow('Final Pencil Sketch', shading_bgr)
cv2.imwrite('Demo_output.jpg', shading_bgr)
out.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
