import os

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

# ---------- Config ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = os.path.join(BASE_DIR, "sam_vit_h_4b8939.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

if not os.path.isfile(SAM_CHECKPOINT):
    raise FileNotFoundError(
        f"SAM checkpoint not found at {SAM_CHECKPOINT}. "
        "Download it from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    )

# ---------- Load models ONCE at startup ----------
yolo_model = YOLO("yolo12x.pt")

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)


# ---------- Core function ----------
def remove_background(input_image: Image.Image) -> Image.Image:
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    # PIL -> RGB numpy array
    image_rgb = np.array(input_image.convert("RGB"))

    # Detect people with YOLO
    results = yolo_model(image_rgb)
    person_boxes = []
    for box in results[0].boxes:
        if int(box.cls[0].item()) == 0:  # class 0 = person in COCO
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append([x1, y1, x2, y2])

    if not person_boxes:
        raise gr.Error("No person detected in the image!")

    # Segment the first detected person with SAM
    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict(
        box=np.array(person_boxes[0]), multimask_output=False
    )
    mask = masks[0]

    # Apply mask as transparency
    rgba_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
    rgba_image[~mask] = (0, 0, 0, 0)

    result = Image.fromarray(rgba_image)

    return result


# ---------- Gradio UI ----------
demo = gr.Interface(
    fn=remove_background,
    inputs=gr.Image(type="pil", label="Drag & Drop or Upload Image Here"),
    outputs=gr.Image(type="pil", image_mode="RGBA", label="Background Removed Result"),
    title="AI Image Background Remover",
    description="Drop any image into the box on the left, and download the transparent PNG result from the right.",
)

if __name__ == "__main__":
    demo.launch()
