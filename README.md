# AI Image Background Remover

A simple web app that removes the background from photos of people. It uses **YOLO12** to find the person and **Segment Anything (SAM)** to cut them out precisely, then returns a transparent PNG. The interface is built with **Gradio**.

## How it works

1. **Upload** an image in the browser.
2. **YOLO12x** detects people in the image (COCO class 0).
3. The first detected person's bounding box is passed to **SAM** as a prompt.
4. SAM produces a pixel-accurate mask of the person.
5. Everything outside the mask is made transparent and the result is returned as an RGBA PNG.

## Features

- Drag-and-drop web interface
- Automatic person detection, no manual clicking needed
- High-quality masks from SAM
- Transparent PNG output, with a copy saved to `output/output.png`
- Automatic GPU (CUDA) use, with CPU fallback

## Requirements

- Python 3.10 to 3.12 recommended (newer versions may lack PyTorch wheels)
- An NVIDIA GPU is strongly recommended. CPU works but is slow with the ViT-H model.

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install torch torchvision opencv-python numpy pillow gradio ultralytics
pip install git+https://github.com/facebookresearch/segment-anything.git
```

> For GPU support, install PyTorch using the command for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

## Download the SAM checkpoint

The SAM weights are not included in this repo. Download one and place it in the project root:

| Model | File | Size | Notes |
|-------|------|------|-------|
| ViT-H (default) | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | ~2.5 GB | Best quality |
| ViT-B | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | ~375 MB | Faster, CPU-friendly |

If you use ViT-B, update these lines in `main.py`:

```python
MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = os.path.join(BASE_DIR, "sam_vit_b_01ec64.pth")
```

The YOLO weights (`yolo12x.pt`) download automatically on first run.

## Usage

```bash
python main.py
```

Open the local URL printed in the terminal (usually `http://127.0.0.1:7860`), upload an image, and download the transparent result.

## Project structure

```
.
├── main.py                  # App: detection, segmentation, Gradio UI
├── sam_vit_h_4b8939.pth     # SAM checkpoint (download separately)
└── README.md
```

## Limitations

- Only the **first detected person** is segmented.
- Works on people only (YOLO class 0). To support other objects, change the class ID filter in `main.py`.
- Large models need significant VRAM (ViT-H is roughly 6 GB or more).

## Credits

- [Segment Anything](https://github.com/facebookresearch/segment-anything) by Meta AI
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Gradio](https://www.gradio.app/)

## License

MIT (or your choice). Note that SAM and Ultralytics YOLO have their own licenses (Apache 2.0 and AGPL-3.0 respectively), so check them if you plan to use this commercially.
