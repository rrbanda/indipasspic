#!/usr/bin/env python3
"""
Indian Passport Photo Processor

Processes a photo to meet Indian Passport Seva portal upload requirements
based on ICAO guidelines and Passport Seva specifications.

Features:
  - Background removal & replacement with clean white
  - Smart cropping (adapts to well-framed or loose photos)
  - Red-eye correction using OpenCV face/eye detection
  - Resize to 630x810 pixels
  - JPEG output under 250 KB

Usage:
  python process_passport_photo.py input.jpg
  python process_passport_photo.py input.jpg -o output.jpg
  python process_passport_photo.py input.jpg --quality 85

Requirements:
  pip install pillow rembg onnxruntime opencv-python-headless
"""

import argparse
import io
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove


# ──────────────────────────────────────────────
# Step 1: Background Removal
# ──────────────────────────────────────────────

def get_subject_mask(img):
    """Get a binary mask of the subject using rembg (U2-Net)."""
    print("  [1/6] Detecting subject...")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    result_bytes = remove(img_bytes.read())
    result = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    _, _, _, alpha = result.split()
    return alpha


def create_natural_blend(original_img, mask):
    """
    Replace background with white using a natural blend.

    Instead of hard cutout (which looks ugly), this:
    - Keeps the original image near the subject (preserving natural edges)
    - Smoothly fades to white only in areas far from the subject
    """
    print("  [2/6] Blending background to white...")
    mask_pil = mask.copy()

    # Dilate the mask to keep a generous region around the subject
    for _ in range(8):
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(7))

    # Heavy Gaussian blur for gradual blend transition
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=40))

    # Strengthen center, soften edges
    blend_np = np.array(mask_pil, dtype=np.float32) / 255.0
    blend_np = np.clip(blend_np * 1.8 - 0.2, 0, 1)
    blend_pil = Image.fromarray((blend_np * 255).astype(np.uint8), mode="L")
    blend_pil = blend_pil.filter(ImageFilter.GaussianBlur(radius=15))
    blend_np = np.array(blend_pil, dtype=np.float32) / 255.0

    # Composite: result = original * blend + white * (1 - blend)
    orig_np = np.array(original_img, dtype=np.float32)
    white = np.full_like(orig_np, 255.0)
    blend_3d = blend_np[:, :, np.newaxis]
    result_np = orig_np * blend_3d + white * (1 - blend_3d)

    return Image.fromarray(np.clip(result_np, 0, 255).astype(np.uint8))


# ──────────────────────────────────────────────
# Step 2: Smart Cropping
# ──────────────────────────────────────────────

def get_subject_bounds(mask):
    """Get bounding box of the subject from the alpha mask."""
    mask_np = np.array(mask)
    rows = np.any(mask_np > 128, axis=1)
    cols = np.any(mask_np > 128, axis=0)

    if not rows.any():
        h, w = mask_np.shape
        return 0, 0, w, h

    top = int(np.argmax(rows))
    bottom = int(len(rows) - np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = int(len(cols) - np.argmax(cols[::-1]))

    return left, top, right, bottom


def smart_crop(img, mask):
    """
    Smart crop that adapts based on input framing:
    - Well-framed photos (subject fills >50%): only adjust aspect ratio
    - Loosely framed photos: tighter crop to ~55% of subject height

    Always outputs 630:810 aspect ratio (width:height).
    """
    print("  [3/6] Smart cropping...")

    w, h = img.size
    left, top, right, bottom = get_subject_bounds(mask)

    subject_w = right - left
    subject_h = bottom - top
    subject_fill = (subject_w * subject_h) / (w * h)
    center_x = (left + right) // 2

    target_ratio = 630.0 / 810.0  # ~0.778

    print(f"         Image: {w}x{h}, Subject fill: {subject_fill*100:.0f}%")

    if subject_fill > 0.5:
        # Already well-framed — just adjust aspect ratio
        print("         Well-framed, adjusting aspect ratio...")
        current_ratio = w / h

        if current_ratio > target_ratio:
            # Too wide — crop sides
            new_w = int(h * target_ratio)
            crop_left = max(0, center_x - new_w // 2)
            crop_right = crop_left + new_w
            if crop_right > w:
                crop_right = w
                crop_left = w - new_w
            crop_top = 0
            crop_bottom = h
        else:
            # Too tall — crop bottom
            new_h = int(w / target_ratio)
            crop_left = 0
            crop_right = w
            crop_top = max(0, top - int(new_h * 0.04))
            crop_bottom = crop_top + new_h
            if crop_bottom > h:
                crop_bottom = h
                crop_top = max(0, crop_bottom - new_h)
    else:
        # Loosely framed — tighter crop
        print("         Loosely framed, cropping tighter...")
        visible_height = int(subject_h * 0.55)
        margin_above = int(visible_height * 0.04)

        crop_top = max(0, top - margin_above)
        crop_bottom = crop_top + visible_height
        if crop_bottom > h:
            crop_bottom = h
            crop_top = max(0, crop_bottom - visible_height)

        crop_height = crop_bottom - crop_top
        crop_width = int(crop_height * target_ratio)
        crop_left = max(0, center_x - crop_width // 2)
        crop_right = crop_left + crop_width
        if crop_right > w:
            crop_right = w
            crop_left = max(0, w - crop_width)

    print(f"         Crop: ({crop_left},{crop_top}) to ({crop_right},{crop_bottom})")
    return img.crop((crop_left, crop_top, crop_right, crop_bottom))


# ──────────────────────────────────────────────
# Step 3: Red-Eye Correction
# ──────────────────────────────────────────────

def fix_red_eye(img):
    """
    Red-eye correction using OpenCV Haar cascade eye detection.

    Detects exact eye positions, then desaturates the red channel
    only in the iris/pupil pixels. Uses feathered circular masks
    to avoid visible artifacts.
    """
    print("  [4/6] Fixing red-eye...")

    pixels = np.array(img, dtype=np.uint8)
    h, w, _ = pixels.shape

    bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
    if len(faces) == 0:
        print("         No face detected, skipping red-eye fix.")
        return img

    fx, fy, fw, fh = faces[0]
    print(f"         Face: ({fx},{fy}) {fw}x{fh}")

    # Detect eyes in upper 60% of face
    face_upper = gray[fy : fy + int(fh * 0.6), fx : fx + fw]
    eyes = eye_cascade.detectMultiScale(face_upper, 1.1, 5, minSize=(15, 15))
    if len(eyes) < 2:
        eyes = eye_cascade.detectMultiScale(face_upper, 1.05, 3, minSize=(10, 10))

    if len(eyes) == 0:
        print("         No eyes detected, skipping.")
        return img

    print(f"         Eyes found: {len(eyes)}")
    pixels_f = pixels.astype(np.float32)

    for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
        # Convert to full-image coordinates
        iris_cx = fx + ex + ew // 2
        iris_cy = fy + ey + eh // 2
        iris_r = int(ew * 0.28)

        eye_name = "left" if iris_cx < fx + fw // 2 else "right"
        print(f"         {eye_name} iris: ({iris_cx},{iris_cy}), r={iris_r}")

        # Extract region around iris
        pad = max(3, int(iris_r * 0.3))
        x1 = max(0, iris_cx - iris_r - pad)
        x2 = min(w, iris_cx + iris_r + pad)
        y1 = max(0, iris_cy - iris_r - pad)
        y2 = min(h, iris_cy + iris_r + pad)

        region = pixels_f[y1:y2, x1:x2].copy()

        # Feathered circular mask
        ry, rx = np.ogrid[0 : y2 - y1, 0 : x2 - x1]
        local_cx = iris_cx - x1
        local_cy = iris_cy - y1
        dist = np.sqrt((rx - local_cx) ** 2 + (ry - local_cy) ** 2).astype(
            np.float32
        )
        feather = max(iris_r * 0.3, 2)
        strength = np.clip((iris_r - dist) / feather, 0, 1)

        r_ch = region[:, :, 0]
        g_ch = region[:, :, 1]
        b_ch = region[:, :, 2]
        brightness = (r_ch + g_ch + b_ch) / 3.0

        # Only correct dark pixels (iris/pupil, not sclera/skin)
        iris_mask = (brightness < 140) & (brightness > 5) & (strength > 0.05)
        cs = strength * iris_mask.astype(np.float32)

        if np.any(cs > 0):
            gray_val = brightness
            # Gentle red desaturation (30%) — enough to remove reddish
            # tint without darkening or discoloring the eyes
            region[:, :, 0] = r_ch * (1 - cs * 0.30) + gray_val * (cs * 0.30)
            # Leave green/blue untouched to preserve natural eye color
            print(f"           → {int(np.sum(cs > 0.1))} pixels corrected")

        pixels_f[y1:y2, x1:x2] = np.clip(region, 0, 255)

    return Image.fromarray(pixels_f.astype(np.uint8))


# ──────────────────────────────────────────────
# Step 4: Enhancement
# ──────────────────────────────────────────────

def enhance_photo(img):
    """Subtle professional enhancements: contrast, sharpness, brightness."""
    print("  [5/6] Enhancing...")
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = ImageEnhance.Sharpness(img).enhance(1.1)
    img = ImageEnhance.Brightness(img).enhance(1.02)
    return img


# ──────────────────────────────────────────────
# Step 5: Resize & Save
# ──────────────────────────────────────────────

def resize_and_save(img, output_path, target_size=(630, 810), max_kb=250, quality=95):
    """Resize to exact passport dimensions and save as JPEG under size limit."""
    print(f"  [6/6] Saving {target_size[0]}x{target_size[1]}...")
    resized = img.resize(target_size, Image.LANCZOS)

    q = quality
    while q >= 50:
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=q, optimize=True, subsampling=0)
        size_kb = buffer.tell() / 1024
        if size_kb <= max_kb:
            resized.save(
                output_path, format="JPEG", quality=q, optimize=True, subsampling=0
            )
            print(f"         Quality: {q}, Size: {size_kb:.1f} KB")
            return
        q -= 5

    resized.save(output_path, format="JPEG", quality=50, optimize=True)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"         Quality: 50, Size: {size_kb:.1f} KB")


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def process(input_path, output_path, quality=95):
    """Full passport photo processing pipeline."""
    print(f"\n{'='*50}")
    print(f"  Indian Passport Photo Processor")
    print(f"{'='*50}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}\n")

    img = Image.open(input_path).convert("RGB")
    print(f"  Original: {img.size[0]}x{img.size[1]}")
    print()

    # Pipeline
    mask = get_subject_mask(img)
    img = create_natural_blend(img, mask)
    img = smart_crop(img, mask)
    img = fix_red_eye(img)
    img = enhance_photo(img)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    resize_and_save(img, output_path, quality=quality)

    # Summary
    final = Image.open(output_path)
    file_size = os.path.getsize(output_path) / 1024
    print(f"\n{'='*50}")
    print(f"  OUTPUT: {output_path}")
    print(f"  {final.size[0]}x{final.size[1]} | {file_size:.1f} KB | JPEG")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process a photo for Indian Passport Seva portal upload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_passport_photo.py photo.jpg
  python process_passport_photo.py photo.jpg -o passport_ready.jpg
  python process_passport_photo.py photo.jpg --quality 85

Specifications (auto-applied):
  • Dimensions: 630 x 810 pixels
  • Format: JPEG
  • Max size: 250 KB
  • Background: White
  • Red-eye: Corrected
        """,
    )
    parser.add_argument("input", help="Input photo path (JPG, PNG, etc.)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: <input>_passport.jpg)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 50-95 (default: 95, auto-reduces if >250KB)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = f"{base}_passport.jpg"

    process(args.input, args.output, quality=args.quality)


if __name__ == "__main__":
    main()
