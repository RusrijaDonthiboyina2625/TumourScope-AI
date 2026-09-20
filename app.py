from flask import Flask, render_template, request, send_file
import os
import uuid
import math
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

app = Flask(__name__)

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "static" / "uploads"
GENERATED_DIR = BASE / "static" / "generated"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Demo calibration only. Replace with the real microscope calibration for real measurements.
PIXELS_PER_UM = 2.0
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def quality_check(gray, contour, mask):
    h, w = gray.shape[:2]
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    resolution_ok = h >= 128 and w >= 128

    if brightness < 45:
        brightness_status = "Too dark"
    elif brightness > 215:
        brightness_status = "Too bright"
    else:
        brightness_status = "Good"

    if blur < 40:
        blur_status = "Blurry"
    elif blur < 120:
        blur_status = "Acceptable"
    else:
        blur_status = "Sharp"

    if contrast < 18:
        contrast_status = "Low contrast"
    else:
        contrast_status = "Good"

    spheroid_present = contour is not None and cv2.contourArea(contour) > max(80, h*w*0.0002)
    score_parts = [
        1 if resolution_ok else 0,
        1 if 45 <= brightness <= 215 else 0,
        1 if blur >= 40 else 0,
        1 if contrast >= 18 else 0,
        1 if spheroid_present else 0,
    ]
    quality_score = round(100 * sum(score_parts) / len(score_parts))
    return {
        "resolution": f"{w} × {h}",
        "resolution_ok": resolution_ok,
        "brightness": round(brightness, 1),
        "brightness_status": brightness_status,
        "blur": round(blur, 1),
        "blur_status": blur_status,
        "contrast": round(contrast, 1),
        "contrast_status": contrast_status,
        "spheroid_present": spheroid_present,
        "quality_score": quality_score,
    }


def choose_contour(gray):
    # Explainable CV detector: denoise -> threshold -> morphology -> contour scoring.
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smooth)

    candidates = []
    for polarity in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, th = cv2.threshold(enhanced, 0, 255, polarity + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape[:2]
        image_area = h * w
        for c in contours:
            area = cv2.contourArea(c)
            if area < max(80, image_area * 0.0002) or area > image_area * 0.75:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            x, y, cw, ch = cv2.boundingRect(c)
            extent = area / max(1, cw * ch)
            cx, cy = x + cw/2, y + ch/2
            center_dist = math.hypot(cx - w/2, cy - h/2) / math.hypot(w/2, h/2)

            # Spheroid-like objects tend to be compact, fairly round and not touch the image edge.
            score = (
                0.50 * min(1.0, max(0.0, circularity)) +
                0.25 * min(1.0, extent / 0.78) +
                0.20 * max(0.0, 1.0 - center_dist) +
                0.05 * min(1.0, math.log10(area + 1) / 5)
            )
            candidates.append((score, c, circularity))

    if not candidates:
        return None, 0.0, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, contour, circularity = candidates[0]
    confidence = int(round(max(0.0, min(1.0, score)) * 100))
    return contour, confidence, circularity


def analyze_image(path, label):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read {label} image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contour, confidence, _ = choose_contour(gray)

    overlay = img.copy()
    mask = np.zeros(gray.shape, dtype=np.uint8)

    if contour is not None:
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        cv2.drawContours(overlay, [contour], -1, (255, 180, 0), 3)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 0), 2)

        area_px = float(cv2.contourArea(contour))
        perimeter_px = float(cv2.arcLength(contour, True))
        diameter_px = float(max(w, h))
        circularity = float(4 * math.pi * area_px / (perimeter_px**2)) if perimeter_px else 0.0
        area_um2 = area_px / (PIXELS_PER_UM ** 2)
        diameter_um = diameter_px / PIXELS_PER_UM
    else:
        area_um2 = diameter_um = circularity = 0.0

    q = quality_check(gray, contour, mask)

    # If the image is questionable, keep the measurement but make the warning explicit.
    warnings = []
    if confidence < 60:
        warnings.append("Low-confidence spheroid detection")
    if not q["resolution_ok"]:
        warnings.append("Low image resolution")
    if q["blur"] < 40:
        warnings.append("Image may be blurry")
    if q["brightness_status"] != "Good":
        warnings.append(q["brightness_status"])
    if q["contrast_status"] != "Good":
        warnings.append(q["contrast_status"])
    if not q["spheroid_present"]:
        warnings.append("Spheroid-like region not confidently found")

    uid = uuid.uuid4().hex[:10]
    out_name = f"{uid}_{label}_detected.png"
    out_path = GENERATED_DIR / out_name
    cv2.imwrite(str(out_path), overlay)

    return {
        "label": label,
        "original_url": "/" + str(path.relative_to(BASE)).replace("\\", "/"),
        "processed_url": "/" + str(out_path.relative_to(BASE)).replace("\\", "/"),
        "diameter": round(diameter_um, 2),
        "area": round(area_um2, 2),
        "circularity": round(circularity, 3),
        "confidence": confidence,
        "quality": q,
        "warnings": warnings,
        "detected": contour is not None,
    }


def save_upload(file, prefix):
    if not file or not file.filename:
        return None
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise ValueError(f"Unsupported file type: {ext}")
    name = f"{uuid.uuid4().hex}_{prefix}{ext}"
    path = UPLOAD_DIR / name
    file.save(path)
    return path


def pct_change(a, b):
    if a == 0:
        return 0.0
    return ((b - a) / a) * 100.0


def build_summary(days):
    if not days:
        return "No measurable time point was submitted."
    valid = [d for d in days if d["earth"]["detected"] and d["micro"]["detected"]]
    if not valid:
        return "The submitted images did not produce a confident pair of spheroid-like detections. Review image quality and microscope calibration."

    first = valid[0]
    last = valid[-1]
    earth_growth = pct_change(first["earth"]["area"], last["earth"]["area"])
    micro_growth = pct_change(first["micro"]["area"], last["micro"]["area"])
    avg_conf = np.mean([first["earth"]["confidence"], first["micro"]["confidence"]])
    return (
        f"Across {len(valid)} analyzable time point(s), the measured spheroid-like region "
        f"area changed by {earth_growth:+.1f}% for Earth control and {micro_growth:+.1f}% "
        f"for Microgravity. The first analyzable pair had an average detection confidence "
        f"of {avg_conf:.0f}%. These are image-derived research measurements, not a medical diagnosis."
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    days = []
    for day in range(1, 6):
        earth = request.files.get(f"earth_{day}")
        micro = request.files.get(f"micro_{day}")
        time_point = request.form.get(f"time_{day}", f"Day {day}").strip() or f"Day {day}"

        if not earth or not micro or not earth.filename or not micro.filename:
            continue

        try:
            earth_path = save_upload(earth, f"day{day}_earth")
            micro_path = save_upload(micro, f"day{day}_micro")
            earth_result = analyze_image(earth_path, f"day{day}_earth")
            micro_result = analyze_image(micro_path, f"day{day}_micro")
        except Exception as exc:
            return f"<h2>Analysis error</h2><p>{exc}</p><p><a href='/'>Back</a></p>", 400

        growth = pct_change(earth_result["area"], micro_result["area"])
        days.append({
            "day": day,
            "time_point": time_point,
            "earth": earth_result,
            "micro": micro_result,
            "area_change": round(growth, 2),
            "diameter_change": round(pct_change(earth_result["diameter"], micro_result["diameter"]), 2),
        })

    if not days:
        return "<h2>Please upload at least one complete Earth + Microgravity pair.</h2><p><a href='/'>Back</a></p>", 400

    summary = build_summary(days)
    report_token = uuid.uuid4().hex
    # Keep results in a simple server-side in-memory store for this demo.
    RESULTS[report_token] = {"days": days, "summary": summary}
    return render_template("result.html", days=days, summary=summary, report_token=report_token)


RESULTS = {}


@app.route("/report/<token>")
def report(token):
    data = RESULTS.get(token)
    if not data:
        return "Report session expired. Please analyze again.", 404

    pdf_path = GENERATED_DIR / f"TumourScope_AI_Report_{token[:8]}.pdf"
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title2", parent=styles["Title"], fontSize=20, leading=24, textColor=colors.HexColor("#00e5ff"))
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, leading=10)

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        rightMargin=14*mm, leftMargin=14*mm, topMargin=14*mm, bottomMargin=14*mm
    )
    story = [
        Paragraph("TumourScope AI — Research Analysis Report", title),
        Spacer(1, 5*mm),
        Paragraph(
            "Research prototype. Image-derived measurements only; not a validated medical diagnostic system.",
            small
        ),
        Spacer(1, 5*mm),
        Paragraph(data["summary"], styles["BodyText"]),
        Spacer(1, 5*mm),
    ]

    table_data = [["Day", "Earth diameter (µm)", "Micro diameter (µm)", "Earth area (µm²)", "Micro area (µm²)", "Area change %"]]
    for d in data["days"]:
        table_data.append([
            d["time_point"],
            f'{d["earth"]["diameter"]:.2f}',
            f'{d["micro"]["diameter"]:.2f}',
            f'{d["earth"]["area"]:.2f}',
            f'{d["micro"]["area"]:.2f}',
            f'{d["area_change"]:+.2f}%'
        ])

    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b3d66")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef8ff")]),
    ]))
    story += [t, Spacer(1, 6*mm)]

    for d in data["days"]:
        story.append(Paragraph(f'{d["time_point"]}: detection and quality notes', styles["Heading3"]))
        notes = []
        for r in (d["earth"], d["micro"]):
            notes.append(
                f'{r["label"]}: confidence {r["confidence"]}%, quality {r["quality"]["quality_score"]}%'
                + (f'; warnings: {", ".join(r["warnings"])}' if r["warnings"] else '; no major warnings')
            )
        story.append(Paragraph("<br/>".join(notes), small))
        story.append(Spacer(1, 3*mm))

    doc.build(story)
    return send_file(pdf_path, as_attachment=True, download_name="TumourScope_AI_Report.pdf")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
