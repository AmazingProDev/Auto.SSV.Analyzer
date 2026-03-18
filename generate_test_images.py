import os
from PIL import Image, ImageDraw, ImageFont

img_dir = "test_images"
os.makedirs(img_dir, exist_ok=True)

colors = [
    (255, 0, 0), (255, 127, 0), (255, 255, 0), (127, 255, 0),
    (0, 255, 0), (0, 255, 127), (0, 255, 255), (0, 127, 255),
    (0, 0, 255), (127, 0, 255), (255, 0, 255), (255, 0, 127)
]

for i in range(12):
    img = Image.new('RGB', (800, 600), color=colors[i])
    d = ImageDraw.Draw(img)
    # Just draw text simply
    text = f"Image {i+1} : Angle {i*30}°"
    # To center without custom font, we'll just draw it a bit large
    d.text((300, 280), text, fill=(255, 255, 255))
    img.save(f"{img_dir}/img_{i+1:02d}.jpg")

print("Generated 12 test images.")
