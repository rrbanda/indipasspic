# Indian Passport Photo Processor

A Python tool to process photos for upload to the **Indian Passport Seva** portal, following **ICAO guidelines** and Passport Seva specifications.

## What It Does

| Step | Description |
|------|-------------|
| **Background removal** | Removes cluttered backgrounds and replaces with clean white using natural blending (no ugly cutout edges) |
| **Smart cropping** | Adapts to well-framed or loosely framed photos; outputs 630:810 aspect ratio |
| **Red-eye correction** | Uses OpenCV face/eye detection to precisely target iris pixels only |
| **Enhancement** | Subtle contrast, sharpness, and brightness improvements |
| **Resize & compress** | Outputs exactly 630x810 pixels, JPEG, under 250 KB |

## Passport Seva Photo Requirements

Based on ICAO guidelines and Passport Seva upload instructions:

- **Dimensions:** 630 x 810 pixels
- **File size:** Under 250 KB
- **Format:** JPEG
- **Background:** Plain white
- **Face coverage:** ~70-80% of the photo
- **Expression:** Neutral, eyes open
- **No red-eye**, no shadows, no filters
- **Forehead visible** — no hair, cap, or head covering
- **Ears visible**
- **Head straight**, not tilted

### Photo Capture Tips

- Use a **mobile device** with **flash OFF**
- Shoot from at least **1.5 meters** distance
- Wear **dark clothes**
- Stand against a **light-colored wall**
- Ensure **even lighting** with no shadows on face or background
- Image resolution should be **higher than 2500 x 2500 pixels**

## Installation

```bash
# Clone the repo
git clone https://github.com/rrbanda/indipasspic.git
cd indipasspic

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage (outputs <input>_passport.jpg)
python process_passport_photo.py photo.jpg

# Specify output path
python process_passport_photo.py photo.jpg -o passport_ready.jpg

# Lower quality for smaller file size
python process_passport_photo.py photo.jpg --quality 85
```

## How It Works

### 1. Background Removal (rembg + U2-Net)
Uses the `rembg` library to detect the subject, then creates a **natural blend** to white — instead of a hard cutout, it preserves the original image near the subject and smoothly fades to white only in distant areas.

### 2. Smart Cropping
Detects how much of the frame the subject fills:
- **Well-framed photos** (>50% fill): Only adjusts aspect ratio to 630:810
- **Loosely framed photos**: Crops tighter so face covers ~70% of the frame

### 3. Red-Eye Correction
Uses OpenCV Haar cascades for face and eye detection to find exact iris positions. Applies a gentle red channel desaturation (30%) only within feathered circular masks on the iris/pupil area — no artifacts on skin, eyebrows, or other areas.

### 4. Output
Resizes to exactly 630x810 pixels and saves as optimized JPEG. Automatically reduces quality if file exceeds 250 KB.

## Dependencies

- `pillow` — Image processing
- `rembg` — AI background removal (U2-Net)
- `onnxruntime` — ML inference for rembg
- `opencv-python-headless` — Face/eye detection
- `numpy` — Array operations

## Signature Upload (Manual)

For the signature upload on Passport Seva:
1. Sign on **white paper** with a **black or blue bold pen**
2. Scan and crop into a **rectangular shape**
3. File size must not exceed **100 KB**
4. Upload only the signature, not the entire page

## Important Notes

- You can upload photo and signature **up to 12 times only** on the portal
- Photos not meeting ICAO specifications may be **rejected by immigration authorities**
- This tool fixes background, cropping, red-eye, and sizing — but **cannot fix**: hair covering forehead, ears not visible, head tilt, or other issues requiring a retake

## License

MIT
