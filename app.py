from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
import uuid
import math
import csv
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
GENERATED_DIR = BASE_DIR / "static" / "generated"
DB_PATH = BASE_DIR / "tumourscope.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

app.secret_key = "tumourscope-ai-research-prototype"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "bmp",
    "webp",
}

DEFAULT_PIXELS_PER_UM = 2.0


# =========================================================
# DATABASE
# =========================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            researcher TEXT,
            sample_id TEXT,
            cell_line TEXT,
            experiment_date TEXT,
            microscope_magnification TEXT,
            pixels_per_um REAL NOT NULL,
            research_notes TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS timepoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            day INTEGER NOT NULL,
            time_label TEXT,

            earth_original TEXT,
            earth_processed TEXT,

            micro_original TEXT,
            micro_processed TEXT,

            earth_area_px REAL,
            earth_area_um2 REAL,
            earth_diameter_px REAL,
            earth_diameter_um REAL,
            earth_circularity REAL,
            earth_confidence REAL,
            earth_quality_score REAL,
            earth_quality_notes TEXT,

            micro_area_px REAL,
            micro_area_um2 REAL,
            micro_diameter_px REAL,
            micro_diameter_um REAL,
            micro_circularity REAL,
            micro_confidence REAL,
            micro_quality_score REAL,
            micro_quality_notes TEXT,

            area_change_percent REAL,
            diameter_change_percent REAL,

            created_at TEXT NOT NULL,

            UNIQUE(experiment_id, day),

            FOREIGN KEY(experiment_id)
                REFERENCES experiments(id)
                ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()


init_db()


# =========================================================
# HELPERS
# =========================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def safe_float(value, default=0.0):
    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return default


def clamp(value, low, high):
    return max(low, min(high, value))


# =========================================================
# IMAGE QUALITY
# =========================================================

def quality_check(image, selected_contour):

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    blur_value = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    score = 100.0

    notes = []

    if width < 128 or height < 128:
        score -= 25
        notes.append("Low resolution")

    if brightness < 45:
        score -= 20
        notes.append("Very dark image")

    elif brightness > 215:
        score -= 20
        notes.append("Very bright image")

    if contrast < 18:
        score -= 20
        notes.append("Low contrast")

    if blur_value < 40:
        score -= 20
        notes.append("Image may be blurry")

    elif blur_value < 120:
        score -= 8
        notes.append("Moderate sharpness")

    if selected_contour is None:
        score -= 20
        notes.append(
            "No suitable spheroid-like region detected"
        )

    else:
        notes.append(
            "Spheroid-like region detected"
        )

    score = clamp(score, 0, 100)

    return {
        "score": round(score, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "blur": round(blur_value, 1),
        "notes": notes,
    }


# =========================================================
# CONTOUR DETECTION
# =========================================================

def choose_contour(gray):

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(blurred)

    _, binary = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, inverse = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    masks = []

    for mask in (binary, inverse):

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        masks.append(mask)

    height, width = gray.shape[:2]

    image_area = float(
        width * height
    )

    image_center = np.array(
        [
            width / 2.0,
            height / 2.0
        ]
    )

    candidates = []

    for mask in masks:

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if (
                area < image_area * 0.002
                or area > image_area * 0.85
            ):
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter <= 0:
                continue

            circularity = (
                4.0
                * math.pi
                * area
                / (perimeter * perimeter)
            )

            x, y, w, h = cv2.boundingRect(
                contour
            )

            rect_area = max(
                1.0,
                float(w * h)
            )

            extent = area / rect_area

            moments = cv2.moments(
                contour
            )

            if moments["m00"] != 0:

                cx = (
                    moments["m10"]
                    / moments["m00"]
                )

                cy = (
                    moments["m01"]
                    / moments["m00"]
                )

            else:

                cx = x + w / 2.0
                cy = y + h / 2.0

            distance = np.linalg.norm(
                np.array([cx, cy])
                - image_center
            )

            max_distance = max(
                1.0,
                np.linalg.norm(image_center)
            )

            center_score = (
                1.0
                - min(
                    1.0,
                    distance / max_distance
                )
            )

            circularity_score = clamp(
                circularity,
                0,
                1
            )

            extent_score = clamp(
                extent / 0.8,
                0,
                1
            )

            area_score = clamp(
                math.log10(area + 1)
                / math.log10(image_area + 1),
                0,
                1
            )

            score = (
                0.50 * circularity_score
                + 0.25 * extent_score
                + 0.20 * center_score
                + 0.05 * area_score
            )

            candidates.append(
                (score, contour)
            )

    if not candidates:
        return None, 0.0

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best_score, best_contour = candidates[0]

    return (
        best_contour,
        clamp(
            best_score * 100.0,
            0,
            100
        )
    )


# =========================================================
# IMAGE ANALYSIS
# =========================================================

def analyze_image(
    image_path,
    output_path,
    pixels_per_um
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            "Could not read the uploaded image."
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    contour, confidence = choose_contour(
        gray
    )

    quality = quality_check(
        image,
        contour
    )

    processed = image.copy()

    metrics = {
        "area_px": 0.0,
        "area_um2": 0.0,
        "diameter_px": 0.0,
        "diameter_um": 0.0,
        "circularity": 0.0,
        "confidence": round(
            confidence,
            1
        ),
        "quality_score": quality["score"],
        "quality_notes": "; ".join(
            quality["notes"]
        ),
    }

    if contour is not None:

        area_px = float(
            cv2.contourArea(contour)
        )

        perimeter = float(
            cv2.arcLength(
                contour,
                True
            )
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        diameter_px = float(
            max(w, h)
        )

        if perimeter > 0:

            circularity = (
                4.0
                * math.pi
                * area_px
                / (perimeter * perimeter)
            )

        else:
            circularity = 0.0

        circularity = clamp(
            circularity,
            0,
            1
        )

        ppm = max(
            pixels_per_um,
            0.000001
        )

        metrics.update(
            {
                "area_px": round(
                    area_px,
                    2
                ),

                "area_um2": round(
                    area_px
                    / (ppm * ppm),
                    2
                ),

                "diameter_px": round(
                    diameter_px,
                    2
                ),

                "diameter_um": round(
                    diameter_px / ppm,
                    2
                ),

                "circularity": round(
                    circularity,
                    4
                ),
            }
        )

        cv2.drawContours(
            processed,
            [contour],
            -1,
            (0, 255, 255),
            3
        )

        cv2.rectangle(
            processed,
            (x, y),
            (x + w, y + h),
            (255, 180, 0),
            2
        )

        label = (
            f"Detected | conf "
            f"{confidence:.1f}%"
        )

        cv2.putText(
            processed,
            label,
            (
                x,
                max(
                    25,
                    y - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    else:

        cv2.putText(
            processed,
            "No suitable spheroid-like region detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 80, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imwrite(
        str(output_path),
        processed
    )

    return metrics


# =========================================================
# CALCULATIONS
# =========================================================

def percentage_change(
    new_value,
    old_value
):

    if (
        old_value is None
        or abs(old_value) < 1e-12
    ):
        return None

    return round(
        (
            (new_value - old_value)
            / old_value
        ) * 100.0,
        2
    )


# =========================================================
# FILE SAVE
# =========================================================

def save_uploaded(
    file_storage,
    prefix,
    day
):

    if (
        not file_storage
        or not file_storage.filename
    ):
        return None

    if not allowed_file(
        file_storage.filename
    ):
        raise ValueError(
            f"Unsupported file type for "
            f"{prefix} Day {day}."
        )

    extension = (
        file_storage.filename
        .rsplit(".", 1)[1]
        .lower()
    )

    filename = (
        f"{uuid.uuid4().hex}_"
        f"{prefix}_day{day}."
        f"{extension}"
    )

    path = UPLOAD_DIR / filename

    file_storage.save(path)

    return f"uploads/{filename}"


# =========================================================
# DATABASE LOAD
# =========================================================

def load_experiment(
    experiment_id
):

    conn = get_db()

    experiment = conn.execute(
        """
        SELECT *
        FROM experiments
        WHERE id = ?
        """,
        (experiment_id,)
    ).fetchone()

    timepoints = conn.execute(
        """
        SELECT *
        FROM timepoints
        WHERE experiment_id = ?
        ORDER BY day
        """,
        (experiment_id,)
    ).fetchall()

    conn.close()

    return experiment, timepoints


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        default_pixels_per_um=
            DEFAULT_PIXELS_PER_UM
    )


# =========================================================
# ANALYZE
# =========================================================

@app.post("/analyze")
def analyze():

    experiment_name = (
        request.form
        .get(
            "experiment_name",
            ""
        )
        .strip()
    )

    researcher = (
        request.form
        .get(
            "researcher",
            ""
        )
        .strip()
    )

    sample_id = (
        request.form
        .get(
            "sample_id",
            ""
        )
        .strip()
    )

    cell_line = (
        request.form
        .get(
            "cell_line",
            ""
        )
        .strip()
    )

    experiment_date = (
        request.form
        .get(
            "experiment_date",
            ""
        )
        .strip()
    )

    microscope_magnification = (
        request.form
        .get(
            "microscope_magnification",
            ""
        )
        .strip()
    )

    research_notes = (
        request.form
        .get(
            "research_notes",
            ""
        )
        .strip()
    )

    pixels_per_um = safe_float(
        request.form.get(
            "pixels_per_um"
        ),
        DEFAULT_PIXELS_PER_UM
    )

    if not experiment_name:

        flash(
            "Please enter an Experiment Name.",
            "error"
        )

        return redirect(
            url_for("index")
        )

    if pixels_per_um <= 0:

        flash(
            "Pixels per micrometre must be greater than 0.",
            "error"
        )

        return redirect(
            url_for("index")
        )

    days = []

    for day in range(1, 6):

        earth = request.files.get(
            f"earth_{day}"
        )

        micro = request.files.get(
            f"micro_{day}"
        )

        time_label = (
            request.form
            .get(
                f"time_{day}",
                f"Day {day}"
            )
            .strip()
            or f"Day {day}"
        )

        if (
            earth
            and earth.filename
            and micro
            and micro.filename
        ):

            days.append(
                (
                    day,
                    time_label,
                    earth,
                    micro
                )
            )

    if not days:

        flash(
            "Upload at least one complete Earth-control + Microgravity image pair.",
            "error"
        )

        return redirect(
            url_for("index")
        )

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO experiments
        (
            experiment_name,
            researcher,
            sample_id,
            cell_line,
            experiment_date,
            microscope_magnification,
            pixels_per_um,
            research_notes,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            experiment_name,
            researcher,
            sample_id,
            cell_line,
            experiment_date,
            microscope_magnification,
            pixels_per_um,
            research_notes,
            datetime.now().isoformat(
                timespec="seconds"
            ),
        )
    )

    experiment_id = cursor.lastrowid

    try:

        for (
            day,
            time_label,
            earth_file,
            micro_file
        ) in days:

            earth_rel = save_uploaded(
                earth_file,
                "earth",
                day
            )

            micro_rel = save_uploaded(
                micro_file,
                "micro",
                day
            )

            earth_input = (
                BASE_DIR
                / "static"
                / earth_rel
            )

            micro_input = (
                BASE_DIR
                / "static"
                / micro_rel
            )

            earth_output_name = (
                f"{uuid.uuid4().hex}"
                f"_earth_day{day}_processed.jpg"
            )

            micro_output_name = (
                f"{uuid.uuid4().hex}"
                f"_micro_day{day}_processed.jpg"
            )

            earth_output = (
                GENERATED_DIR
                / earth_output_name
            )

            micro_output = (
                GENERATED_DIR
                / micro_output_name
            )

            earth_metrics = analyze_image(
                earth_input,
                earth_output,
                pixels_per_um
            )

            micro_metrics = analyze_image(
                micro_input,
                micro_output,
                pixels_per_um
            )

            area_change = percentage_change(
                micro_metrics["area_um2"],
                earth_metrics["area_um2"]
            )

            diameter_change = percentage_change(
                micro_metrics["diameter_um"],
                earth_metrics["diameter_um"]
            )

            cursor.execute(
                """
                INSERT INTO timepoints
                (
                    experiment_id,
                    day,
                    time_label,

                    earth_original,
                    earth_processed,

                    micro_original,
                    micro_processed,

                    earth_area_px,
                    earth_area_um2,
                    earth_diameter_px,
                    earth_diameter_um,
                    earth_circularity,
                    earth_confidence,
                    earth_quality_score,
                    earth_quality_notes,

                    micro_area_px,
                    micro_area_um2,
                    micro_diameter_px,
                    micro_diameter_um,
                    micro_circularity,
                    micro_confidence,
                    micro_quality_score,
                    micro_quality_notes,

                    area_change_percent,
                    diameter_change_percent,

                    created_at
                )
                VALUES
                (
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?,
                    ?
                )
                """,
                (
                    experiment_id,
                    day,
                    time_label,

                    earth_rel,
                    f"generated/{earth_output_name}",

                    micro_rel,
                    f"generated/{micro_output_name}",

                    earth_metrics["area_px"],
                    earth_metrics["area_um2"],
                    earth_metrics["diameter_px"],
                    earth_metrics["diameter_um"],
                    earth_metrics["circularity"],
                    earth_metrics["confidence"],
                    earth_metrics["quality_score"],
                    earth_metrics["quality_notes"],

                    micro_metrics["area_px"],
                    micro_metrics["area_um2"],
                    micro_metrics["diameter_px"],
                    micro_metrics["diameter_um"],
                    micro_metrics["circularity"],
                    micro_metrics["confidence"],
                    micro_metrics["quality_score"],
                    micro_metrics["quality_notes"],

                    area_change,
                    diameter_change,

                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
                )
            )

        conn.commit()

    except Exception:

        conn.rollback()
        conn.close()

        raise

    conn.close()

    return redirect(
        url_for(
            "result",
            experiment_id=experiment_id
        )
    )


# =========================================================
# RESULT
# =========================================================

@app.get("/result/<int:experiment_id>")
def result(experiment_id):

    experiment, timepoints = load_experiment(
        experiment_id
    )

    if experiment is None:
        return "Experiment not found", 404

    return render_template(
        "result.html",
        experiment=experiment,
        timepoints=timepoints
    )


# =========================================================
# HISTORY
# =========================================================

@app.get("/history")
def history():

    conn = get_db()

    experiments = conn.execute(
        """
        SELECT
            e.*,
            COUNT(t.id) AS day_count
        FROM experiments e
        LEFT JOIN timepoints t
            ON t.experiment_id = e.id
        GROUP BY e.id
        ORDER BY e.id DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "history.html",
        experiments=experiments
    )


# =========================================================
# CSV EXPORT
# =========================================================

@app.get("/export/<int:experiment_id>.csv")
def export_csv(experiment_id):

    experiment, timepoints = load_experiment(
        experiment_id
    )

    if experiment is None:
        return "Experiment not found", 404

    filename = (
        GENERATED_DIR
        / f"tumourscope_experiment_{experiment_id}.csv"
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "Experiment",
                "Researcher",
                "Sample ID",
                "Cell Line",
                "Date",
                "Microscope Magnification",
                "Pixels per um",
                "Day",
                "Time Point",
                "Earth Diameter um",
                "Microgravity Diameter um",
                "Earth Area um2",
                "Microgravity Area um2",
                "Area Change %",
                "Earth Circularity",
                "Microgravity Circularity",
                "Earth Confidence %",
                "Microgravity Confidence %",
                "Earth Quality",
                "Microgravity Quality",
            ]
        )

        for row in timepoints:

            writer.writerow(
                [
                    experiment["experiment_name"],
                    experiment["researcher"],
                    experiment["sample_id"],
                    experiment["cell_line"],
                    experiment["experiment_date"],
                    experiment[
                        "microscope_magnification"
                    ],
                    experiment[
                        "pixels_per_um"
                    ],
                    row["day"],
                    row["time_label"],
                    row["earth_diameter_um"],
                    row["micro_diameter_um"],
                    row["earth_area_um2"],
                    row["micro_area_um2"],
                    row["area_change_percent"],
                    row["earth_circularity"],
                    row["micro_circularity"],
                    row["earth_confidence"],
                    row["micro_confidence"],
                    row["earth_quality_score"],
                    row["micro_quality_score"],
                ]
            )

    return send_file(
        filename,
        as_attachment=True,
        download_name=filename.name,
        mimetype="text/csv"
    )


# =========================================================
# PDF REPORT
# =========================================================

@app.get("/report/<int:experiment_id>.pdf")
def report(experiment_id):

    experiment, timepoints = load_experiment(
        experiment_id
    )

    if experiment is None:
        return "Experiment not found", 404

    filename = (
        GENERATED_DIR
        / f"tumourscope_experiment_{experiment_id}.pdf"
    )

    pdf = canvas.Canvas(
        str(filename),
        pagesize=A4
    )

    width, height = A4

    y = height - 45

    def line(
        text,
        size=10,
        gap=16
    ):
        nonlocal y

        pdf.setFont(
            "Helvetica",
            size
        )

        pdf.drawString(
            42,
            y,
            str(text)[:115]
        )

        y -= gap

        if y < 55:

            pdf.showPage()

            y = height - 45

    pdf.setFont(
        "Helvetica-Bold",
        18
    )

    pdf.drawString(
        42,
        y,
        "TumourScope AI - Research Analysis Report"
    )

    y -= 28

    line(
        f"Experiment: {experiment['experiment_name']}"
    )

    line(
        f"Researcher: {experiment['researcher'] or '-'}"
    )

    line(
        f"Sample ID: {experiment['sample_id'] or '-'}"
    )

    line(
        f"Cell Line: {experiment['cell_line'] or '-'}"
    )

    line(
        f"Experiment Date: {experiment['experiment_date'] or '-'}"
    )

    line(
        f"Microscope: {experiment['microscope_magnification'] or '-'}"
    )

    line(
        f"Calibration: {experiment['pixels_per_um']} pixels per micrometre"
    )

    line("")

    line(
        "Important: This is a research prototype, not a medical diagnostic system.",
        9
    )

    line(
        "Measurements depend on image quality, contour detection and microscope calibration.",
        9
    )

    line("")

    for row in timepoints:

        pdf.setFont(
            "Helvetica-Bold",
            12
        )

        pdf.drawString(
            42,
            y,
            f"Day {row['day']} - {row['time_label']}"
        )

        y -= 20

        line(
            f"Earth: diameter {row['earth_diameter_um']} um | "
            f"area {row['earth_area_um2']} um2 | "
            f"circularity {row['earth_circularity']}"
        )

        line(
            f"Microgravity: diameter {row['micro_diameter_um']} um | "
            f"area {row['micro_area_um2']} um2 | "
            f"circularity {row['micro_circularity']}"
        )

        line(
            f"Area change Microgravity vs Earth: "
            f"{row['area_change_percent']}%"
        )

        line(
            f"Detection confidence: "
            f"Earth {row['earth_confidence']}% | "
            f"Microgravity {row['micro_confidence']}%"
        )

        line(
            f"Quality score: "
            f"Earth {row['earth_quality_score']} | "
            f"Microgravity {row['micro_quality_score']}"
        )

        line("")

    if experiment["research_notes"]:

        pdf.setFont(
            "Helvetica-Bold",
            11
        )

        pdf.drawString(
            42,
            y,
            "Research Notes"
        )

        y -= 18

        for note_line in experiment[
            "research_notes"
        ].splitlines():

            line(
                note_line,
                9,
                14
            )

    pdf.save()

    return send_file(
        filename,
        as_attachment=True,
        download_name=filename.name,
        mimetype="application/pdf"
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "database": DB_PATH.name
    }


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print(
        f"TumourScope AI database: {DB_PATH}"
    )

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=True
    )