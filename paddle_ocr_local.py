from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from paddleocr import PaddleOCR


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "ocr_images"
DEFAULT_OUTPUT = BASE_DIR / "paddleocr_output"
DEFAULT_MANIFEST = "processed_manifest.json"
TABLE_ROW_COUNT = 31
MIN_CONTOUR_AREA_RATIO = 0.2

# Normalized regions tuned for the Monthly Timesheet Record Card layout.
HEADER_BOXES = {
    "document_title": (0.22, 0.045, 0.75, 0.12),
    "employee_name": (0.12, 0.115, 0.46, 0.185),
    "wp_fin_no": (0.12, 0.155, 0.44, 0.23),
    "month_year": (0.60, 0.10, 0.97, 0.165),
    "employee_id": (0.53, 0.145, 0.98, 0.22),
}

COLUMN_BOXES = {
    "day": (0.0, 0.17, 0.045, 0.91),
    "regular_hours": (0.045, 0.17, 0.145, 0.91),
    "ot_hours": (0.145, 0.17, 0.252, 0.91),
    "rope_access_allowance": (0.252, 0.17, 0.427, 0.91),
    "transport_allowance": (0.427, 0.17, 0.545, 0.91),
    "night_shift_allowance": (0.545, 0.17, 0.655, 0.91),
    "food_allowance": (0.655, 0.17, 0.763, 0.91),
    "job_site": (0.763, 0.17, 0.873, 0.91),
    "supervisor_signature": (0.873, 0.17, 0.995, 0.91),
}

TOTAL_BOXES = {
    "regular_hours": (0.045, 0.91, 0.145, 0.965),
    "ot_hours": (0.145, 0.91, 0.252, 0.965),
    "rope_access_allowance": (0.252, 0.91, 0.427, 0.965),
    "transport_allowance": (0.427, 0.91, 0.545, 0.965),
    "night_shift_allowance": (0.545, 0.91, 0.655, 0.965),
    "food_allowance": (0.655, 0.91, 0.763, 0.965),
}

FOOTER_BOXES = {
    "employee_signature": (0.18, 0.945, 0.43, 0.995),
    "prepare_by": (0.17, 0.955, 0.35, 0.995),
    "supervisor_name": (0.43, 0.955, 0.68, 0.995),
    "approved_by": (0.69, 0.955, 0.84, 0.995),
    "project_engineer_name": (0.74, 0.955, 0.98, 0.995),
}

MONTH_MAP = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


@dataclass
class OCRLine:
    source_file: str
    page_index: int
    line_index: int
    text: str
    score: float
    box: list[list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured timesheet JSON from images using PaddleOCR."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=f"Image file or folder to scan. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help=f"Folder for OCR output. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="OCR language, for example en, ch, or chinese_cht.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="If input is a folder, only scan the top level.",
    )
    parser.add_argument(
        "--disable-angle-cls",
        action="store_true",
        help="Disable angle classification.",
    )
    parser.add_argument(
        "--det-model-dir",
        default="",
        help="Optional local detection model directory.",
    )
    parser.add_argument(
        "--rec-model-dir",
        default="",
        help="Optional local recognition model directory.",
    )
    parser.add_argument(
        "--cls-model-dir",
        default="",
        help="Optional local angle classification model directory.",
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Show PaddleOCR internal logs.",
    )
    parser.add_argument(
        "--rescan-all",
        action="store_true",
        help="Ignore the processed manifest and rescan all matching images.",
    )
    return parser.parse_args()


def iter_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    iterator = path.rglob("*") if recursive else path.glob("*")
    images = [
        candidate
        for candidate in sorted(iterator)
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise FileNotFoundError(f"No image files found in: {path}")
    return images


def build_ocr_engine(args: argparse.Namespace) -> PaddleOCR:
    engine_options: dict[str, Any] = {
        "use_angle_cls": not args.disable_angle_cls,
        "lang": args.lang,
        "show_log": args.show_log,
    }
    if args.det_model_dir:
        engine_options["det_model_dir"] = str(Path(args.det_model_dir).resolve())
    if args.rec_model_dir:
        engine_options["rec_model_dir"] = str(Path(args.rec_model_dir).resolve())
    if args.cls_model_dir:
        engine_options["cls_model_dir"] = str(Path(args.cls_model_dir).resolve())
    return PaddleOCR(**engine_options)


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def orientation_candidates(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("original", image),
        ("rot90_cw", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ("rot90_ccw", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("rot180", cv2.rotate(image, cv2.ROTATE_180)),
    ]


def crop_to_foreground(image: np.ndarray, padding: int = 20) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < 245
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return image

    x, y, width, height = cv2.boundingRect(points)
    start_x = max(0, x - padding)
    start_y = max(0, y - padding)
    end_x = min(image.shape[1], x + width + padding)
    end_y = min(image.shape[0], y + height + padding)
    return image[start_y:end_y, start_x:end_x]


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    summed = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    rect[0] = points[np.argmin(summed)]
    rect[2] = points[np.argmax(summed)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def find_card_contour(image: np.ndarray) -> np.ndarray | None:
    image_area = image.shape[0] * image.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 40, 140)
    edged = cv2.dilate(edged, np.ones((3, 3), dtype=np.uint8), iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < image_area * MIN_CONTOUR_AREA_RATIO:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4:
            return approximation.reshape(4, 2)

    return None


def warp_card(image: np.ndarray) -> np.ndarray:
    contour = find_card_contour(image)
    if contour is None:
        return image

    rect = order_points(contour.astype(np.float32))
    width_top = np.linalg.norm(rect[1] - rect[0])
    width_bottom = np.linalg.norm(rect[2] - rect[3])
    height_left = np.linalg.norm(rect[3] - rect[0])
    height_right = np.linalg.norm(rect[2] - rect[1])
    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, transform, (max_width, max_height))


def normalize_card_image(image: np.ndarray) -> np.ndarray:
    cropped = crop_to_foreground(image)
    warped = warp_card(cropped)
    return crop_to_foreground(warped)


def image_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    cropped = crop_to_foreground(image)
    warped = normalize_card_image(image)
    return [
        ("raw", image),
        ("cropped", cropped),
        ("warped", warped),
        ("cropped_binary", preprocess_for_ocr(cropped)),
        ("warped_binary", preprocess_for_ocr(warped)),
    ]


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def crop_box(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    start_x = max(0, min(width, int(round(x1 * width))))
    end_x = max(0, min(width, int(round(x2 * width))))
    start_y = max(0, min(height, int(round(y1 * height))))
    end_y = max(0, min(height, int(round(y2 * height))))

    if end_x <= start_x or end_y <= start_y:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    return image[start_y:end_y, start_x:end_x]


def split_column_into_rows(column_image: np.ndarray, rows: int) -> list[np.ndarray]:
    if column_image.size == 0:
        return [np.zeros((0, 0, 3), dtype=np.uint8) for _ in range(rows)]

    height = column_image.shape[0]
    row_height = height / rows
    cells: list[np.ndarray] = []
    for index in range(rows):
        start = int(round(index * row_height))
        end = int(round((index + 1) * row_height))
        start = max(0, min(height, start))
        end = max(0, min(height, end))
        if end <= start:
            cells.append(np.zeros((0, 0, 3), dtype=np.uint8))
        else:
            cells.append(column_image[start:end, :])
    return cells


def normalize_result(raw_result: Any, source_file: str) -> list[OCRLine]:
    pages = raw_result if isinstance(raw_result, list) else [raw_result]
    lines: list[OCRLine] = []

    for page_index, page in enumerate(pages, start=1):
        if page is None:
            continue
        if not isinstance(page, list):
            continue
        for line_index, entry in enumerate(page, start=1):
            parsed_line = parse_line_entry(
                entry=entry,
                source_file=source_file,
                page_index=page_index,
                line_index=line_index,
            )
            if parsed_line is not None:
                lines.append(parsed_line)
    return lines


def parse_line_entry(
    entry: Any,
    source_file: str,
    page_index: int,
    line_index: int,
) -> OCRLine | None:
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None

    box, payload = entry[0], entry[1]
    text = ""
    score = 0.0

    if isinstance(payload, (list, tuple)) and payload:
        text = str(payload[0]).strip()
        if len(payload) > 1:
            try:
                score = float(payload[1])
            except (TypeError, ValueError):
                score = 0.0

    normalized_box: list[list[float]] = []
    if isinstance(box, (list, tuple)):
        for point in box:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    normalized_box.append([float(point[0]), float(point[1])])
                except (TypeError, ValueError):
                    continue

    return OCRLine(
        source_file=source_file,
        page_index=page_index,
        line_index=line_index,
        text=text,
        score=round(score, 6),
        box=normalized_box,
    )


def line_left(line: OCRLine) -> float:
    return min(point[0] for point in line.box) if line.box else 0.0


def line_right(line: OCRLine) -> float:
    return max(point[0] for point in line.box) if line.box else 0.0


def line_top(line: OCRLine) -> float:
    return min(point[1] for point in line.box) if line.box else 0.0


def line_bottom(line: OCRLine) -> float:
    return max(point[1] for point in line.box) if line.box else 0.0


def line_x_center(line: OCRLine) -> float:
    return (line_left(line) + line_right(line)) / 2


def line_y_center(line: OCRLine) -> float:
    return (line_top(line) + line_bottom(line)) / 2


def group_lines_by_rows(lines: list[OCRLine], tolerance: float) -> list[list[OCRLine]]:
    groups: list[list[OCRLine]] = []
    sorted_lines = sorted(lines, key=lambda item: (line_y_center(item), line_x_center(item)))

    for line in sorted_lines:
        if not groups:
            groups.append([line])
            continue

        current_group = groups[-1]
        current_center = sum(line_y_center(item) for item in current_group) / len(current_group)
        if abs(line_y_center(line) - current_center) <= tolerance:
            current_group.append(line)
        else:
            groups.append([line])

    for group in groups:
        group.sort(key=line_x_center)
    return groups


def group_to_text(group: list[OCRLine]) -> str:
    return clean_text(" ".join(line.text for line in sorted(group, key=line_x_center)))


def box_to_pixel_range(
    width: int,
    height: int,
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return x1 * width, y1 * height, x2 * width, y2 * height


def lines_in_box(
    lines: list[OCRLine],
    width: int,
    height: int,
    box: tuple[float, float, float, float],
) -> list[OCRLine]:
    start_x, start_y, end_x, end_y = box_to_pixel_range(width, height, box)
    selected: list[OCRLine] = []
    for line in lines:
        x_center = line_x_center(line)
        y_center = line_y_center(line)
        if start_x <= x_center <= end_x and start_y <= y_center <= end_y:
            selected.append(line)
    return selected


def average_score(lines: list[OCRLine]) -> float:
    if not lines:
        return 0.0
    return round(sum(line.score for line in lines) / len(lines), 6)


def normalize_ascii_text(text: str) -> str:
    replacements = {
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "＆": "&",
        "：": ":",
    }
    normalized = text
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return clean_text(normalized)


def tokenize_upper_words(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Z0-9-]+", text.upper()) if token]


def looks_like_name_token(token: str) -> bool:
    if token in {"NAME", "WP", "FIN", "NO", "MONTH", "EMPLOYEE", "ID", "RECORD", "CARD"}:
        return False
    if re.fullmatch(r"\d+", token):
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z-]*", token))


def extract_name_value(text: str) -> str:
    upper = normalize_ascii_text(text).upper()
    if "NAME" not in upper:
        return ""
    candidate = re.sub(r".*?NAME\s*[:.]?\s*", "", upper)
    candidate = re.split(r"\bWP\s*/?\s*FIN\b|\bMONTH\b|\bEMPLOYEE\s*ID\b", candidate)[0]
    tokens = [token for token in tokenize_upper_words(candidate) if looks_like_name_token(token)]
    if not tokens:
        return ""
    return " ".join(tokens[:3]).strip()


def extract_wp_fin_value(text: str) -> str:
    upper = normalize_ascii_text(text).upper()
    match = re.search(r"WP\s*/?\s*FIN(?:\s*NO)?\s*[:.]?\s*([A-Z0-9-]+)", upper)
    if match:
        return match.group(1)
    tokens = tokenize_upper_words(upper)
    skip = {"WP", "FIN", "NO", "NAME", "MONTH", "EMPLOYEE", "ID"}
    for token in tokens:
        if token in skip:
            continue
        if re.fullmatch(r"[A-Z0-9-]{3,10}", token):
            return token
    return ""


def extract_employee_id_value(text: str) -> str:
    upper = normalize_ascii_text(text).upper()
    match = re.search(r"EMPLOYEE\s*ID(?:\s*NO)?\s*[:.]?\s*([A-Z0-9-]+)", upper)
    if match:
        return match.group(1)
    tokens = tokenize_upper_words(upper)
    skip = {"EMPLOYEE", "ID", "NO", "MONTH", "NAME", "WP", "FIN"}
    for token in tokens:
        if token in skip:
            continue
        if re.fullmatch(r"[A-Z]{1,4}-\d{2,6}", token):
            return token
    return ""


def header_quality(header: dict[str, Any]) -> int:
    score = 0
    if "TIMESHEET" in header.get("document_title", "").upper():
        score += 20
    if header.get("employee_name"):
        score += 15
    if header.get("wp_fin_no"):
        score += 15
    if header.get("employee_id"):
        score += 15
    if header.get("month"):
        score += 10
    if header.get("year"):
        score += 10
    return score


def parse_header_from_lines(lines: list[OCRLine], width: int, height: int) -> dict[str, Any]:
    title_lines = lines_in_box(lines, width, height, HEADER_BOXES["document_title"])
    month_year_lines = lines_in_box(lines, width, height, HEADER_BOXES["month_year"])
    header_lines = [line for line in lines if line_y_center(line) <= height * 0.26]
    row_groups = group_lines_by_rows(header_lines, tolerance=max(12.0, height * 0.012))
    row_texts = [normalize_ascii_text(group_to_text(group)) for group in row_groups if group]
    combined_text = " | ".join(row_texts)

    title_text = normalize_ascii_text(group_to_text(title_lines)) or next(
        (text for text in row_texts if "TIMESHEET" in text.upper() or "RECORD CARD" in text.upper()),
        "",
    )
    name_text = next((text for text in row_texts if "NAME" in text.upper()), "")
    wp_text = next((text for text in row_texts if "WP" in text.upper() or "FIN" in text.upper()), "")
    employee_id_text = next(
        (text for text in row_texts if "EMPLOYEE" in text.upper() and "ID" in text.upper()),
        "",
    )
    month_year_text = normalize_ascii_text(group_to_text(month_year_lines)) or next(
        (text for text in row_texts if "MONTH" in text.upper() or re.search(r"20\d{2}", text)),
        "",
    )

    month, year = extract_month_year(month_year_text or combined_text)
    employee_name = extract_name_value(name_text) or extract_name_value(combined_text)
    wp_fin_no = extract_wp_fin_value(wp_text) or extract_wp_fin_value(combined_text)
    employee_id = extract_employee_id_value(employee_id_text) or extract_employee_id_value(combined_text)

    return {
        "document_title": title_text,
        "employee_name": employee_name,
        "wp_fin_no": wp_fin_no,
        "employee_id": employee_id,
        "month": month,
        "year": year,
        "raw_header_text": {
            "document_title": title_text,
            "employee_name": name_text,
            "wp_fin_no": wp_text,
            "month_year": month_year_text,
            "employee_id": employee_id_text,
            "combined_header_text": combined_text,
        },
        "header_scores": {
            "document_title": average_score(title_lines),
            "employee_name": average_score(
                [line for line in header_lines if "NAME" in normalize_ascii_text(line.text).upper()]
            ),
            "wp_fin_no": average_score(
                [line for line in header_lines if "WP" in normalize_ascii_text(line.text).upper() or "FIN" in normalize_ascii_text(line.text).upper()]
            ),
            "month_year": average_score(month_year_lines),
            "employee_id": average_score(
                [line for line in header_lines if "EMPLOYEE" in normalize_ascii_text(line.text).upper() or "ID" in normalize_ascii_text(line.text).upper()]
            ),
        },
    }


def collect_row_cells(
    row_lines: list[OCRLine],
    width: int,
    height: int,
) -> dict[str, str]:
    row: dict[str, str] = {}
    for field_name, box in COLUMN_BOXES.items():
        start_x, _, end_x, _ = box_to_pixel_range(width, height, box)
        cell_lines = [
            line
            for line in row_lines
            if start_x <= line_x_center(line) <= end_x
        ]
        row[field_name] = normalize_cell_value(field_name, group_to_text(cell_lines))
    return row


def parse_table_from_lines(lines: list[OCRLine], width: int, height: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    table_lines = [
        line
        for line in lines
        if height * 0.18 <= line_y_center(line) <= height * 0.92
    ]
    row_groups = group_lines_by_rows(table_lines, tolerance=max(12.0, height * 0.012))

    rows: list[dict[str, str]] = []
    review_flags: list[dict[str, str]] = []
    seen_days: set[str] = set()

    for row_group in row_groups:
        row_text = group_to_text(row_group).lower().replace(" ", "")
        if "date" in row_text and "regular" in row_text:
            continue
        row = collect_row_cells(row_group, width, height)
        day = normalize_cell_value("day", row.get("day", ""))
        if not day:
            continue
        if not day.isdigit():
            continue
        if not 1 <= int(day) <= 31:
            continue

        day = day.zfill(2)
        if day in seen_days:
            continue
        seen_days.add(day)
        row["day"] = day
        row["allowance_total"] = calculate_allowance_total(row)
        rows.append(row)

        non_day_values = [
            row[key]
            for key in row
            if key not in {"day", "allowance_total", "supervisor_signature"}
        ]
        if not any(value for value in non_day_values):
            review_flags.append(
                {
                    "issue": "empty_row",
                    "details": f"Detected day {row['day']} but the work cells were blank.",
                }
            )

    rows.sort(key=lambda item: int(item["day"]))
    return rows, review_flags


def parse_totals_from_lines(lines: list[OCRLine], width: int, height: int) -> dict[str, str]:
    total_lines = [
        line
        for line in lines
        if height * 0.90 <= line_y_center(line) <= height * 0.97
    ]
    if not total_lines:
        return {field_name: "" for field_name in TOTAL_BOXES}

    totals = {}
    for field_name, box in TOTAL_BOXES.items():
        start_x, _, end_x, _ = box_to_pixel_range(width, height, box)
        cell_lines = [
            line
            for line in total_lines
            if start_x <= line_x_center(line) <= end_x
        ]
        totals[field_name] = normalize_cell_value(field_name, group_to_text(cell_lines))
    return totals


def parse_footer_from_lines(lines: list[OCRLine], width: int, height: int) -> dict[str, str]:
    footer: dict[str, str] = {}
    for field_name, box in FOOTER_BOXES.items():
        footer[field_name] = group_to_text(lines_in_box(lines, width, height, box))
    return footer


def ocr_lines(ocr: PaddleOCR, image: np.ndarray, use_cls: bool) -> list[OCRLine]:
    raw_result = ocr.ocr(image, cls=use_cls)
    return normalize_result(raw_result, source_file="")


def ocr_text(ocr: PaddleOCR, image: np.ndarray, use_cls: bool) -> tuple[str, float]:
    if image.size == 0:
        return "", 0.0
    try:
        raw_result = ocr.ocr(image, cls=use_cls)
    except Exception:
        return "", 0.0
    lines = normalize_result(raw_result, source_file="")
    texts = [clean_text(line.text) for line in lines if clean_text(line.text)]
    if not texts:
        return "", 0.0
    avg_score = sum(line.score for line in lines) / len(lines) if lines else 0.0
    return " ".join(texts).strip(), round(avg_score, 6)


def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.strip("|")
    return text


def digits_only(text: str) -> str:
    return "".join(char for char in text if char.isdigit())


def safe_file_stem(value: str, fallback: str) -> str:
    candidate = clean_text(value).strip()
    if not candidate:
        candidate = fallback
    candidate = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", candidate)
    candidate = re.sub(r"\s+", "_", candidate)
    candidate = candidate.strip(" ._")
    return candidate or fallback


def unique_output_path(output_dir: Path, base_name: str, suffix: str) -> Path:
    candidate = output_dir / f"{base_name}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{counter}{suffix}"
        counter += 1
    return candidate


def build_manifest_key(image_path: Path) -> str:
    return str(image_path.resolve()).lower()


def image_signature(image_path: Path) -> dict[str, Any]:
    stat = image_path.stat()
    return {
        "path": str(image_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / DEFAULT_MANIFEST
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = output_dir / DEFAULT_MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def should_skip_image(image_path: Path, manifest: dict[str, Any]) -> bool:
    entry = manifest.get(build_manifest_key(image_path))
    if not isinstance(entry, dict):
        return False
    signature = image_signature(image_path)
    return (
        entry.get("size") == signature["size"]
        and entry.get("mtime_ns") == signature["mtime_ns"]
    )


def update_manifest_entry(
    manifest: dict[str, Any],
    image_path: Path,
    payload: dict[str, Any],
    json_path: Path,
    txt_path: Path,
) -> None:
    signature = image_signature(image_path)
    manifest[build_manifest_key(image_path)] = {
        **signature,
        "employee_name": payload.get("header", {}).get("employee_name", ""),
        "employee_id": payload.get("header", {}).get("employee_id", ""),
        "month": payload.get("header", {}).get("month", ""),
        "year": payload.get("header", {}).get("year", ""),
        "json_output": json_path.name,
        "txt_output": txt_path.name,
    }


def extract_number_like(text: str) -> str:
    match = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return match.group(0) if match else ""


def extract_month_year(text: str) -> tuple[str, str]:
    lowered = text.lower()
    month = ""
    for key, value in MONTH_MAP.items():
        if key in lowered:
            month = value
            break
    years = re.findall(r"(20\d{2}|19\d{2})", text)
    year = years[0] if years else ""
    return month, year


def strip_label(text: str, patterns: list[str]) -> str:
    value = text
    for pattern in patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    value = re.sub(r"^[\s:._-]+", "", value)
    value = re.sub(r"[\s:._-]+$", "", value)
    return clean_text(value)


def normalize_cell_value(field_name: str, text: str) -> str:
    value = normalize_ascii_text(text)
    if not value:
        return ""

    compact = value.replace(" ", "")
    dash_only_pattern = r"[-_=~|/\\]+"
    weird_dash_values = {"-", "_", "=", "一", '"', "''", "."}
    if value in weird_dash_values or re.fullmatch(dash_only_pattern, compact):
        return ""

    if field_name == "day":
        digits = digits_only(value)
        return digits[:2]

    if field_name in {
        "regular_hours",
        "ot_hours",
        "rope_access_allowance",
        "transport_allowance",
        "night_shift_allowance",
        "food_allowance",
    }:
        upper = value.upper().replace(" ", "")
        if upper in {"PH", "P.H", "P/H"}:
            return "P.H"
        if field_name == "rope_access_allowance":
            upper = upper.replace("$", "")
            value = value.replace("$", "")
        if field_name in {"regular_hours", "ot_hours"} and upper in {"", "-", "--"}:
            return ""
        numeric = extract_number_like(value)
        if numeric:
            if field_name == "regular_hours" and numeric not in {"0", "4", "8", "12"}:
                return ""
            return numeric
        if field_name == "regular_hours":
            if upper in {"PH", "P.H"}:
                return "P.H"
            return ""
        if field_name == "ot_hours":
            return ""
        if field_name == "rope_access_allowance":
            return ""
        return value

    if field_name == "job_site":
        return re.sub(r"[^A-Za-z0-9/-]", "", value.upper())

    return value


def normalized_text_for_match(text: str) -> str:
    return normalize_ascii_text(text).upper().replace(" ", "")


def detect_table_body_bounds(
    lines: list[OCRLine],
    width: int,
    height: int,
) -> tuple[float, float]:
    default_top = height * 0.23
    default_bottom = height * 0.90

    header_candidates = [
        line
        for line in lines
        if any(
            keyword in normalized_text_for_match(line.text)
            for keyword in ("DATE", "REGULAR", "OTHOURS", "OTHOURS/OT", "ROPEACCESS")
        )
    ]
    total_candidates = [
        line
        for line in lines
        if "TOTAL" in normalized_text_for_match(line.text)
    ]

    top = default_top
    if header_candidates:
        top = max(default_top, max(line_bottom(line) for line in header_candidates) + height * 0.01)

    bottom = default_bottom
    if total_candidates:
        bottom = min(default_bottom, min(line_top(line) for line in total_candidates) - height * 0.01)

    if bottom <= top:
        return default_top, default_bottom
    return top, bottom


def build_table_column_boxes(table_top: float, table_bottom: float, height: int) -> dict[str, tuple[float, float, float, float]]:
    normalized_top = max(0.0, min(1.0, table_top / height))
    normalized_bottom = max(0.0, min(1.0, table_bottom / height))
    boxes: dict[str, tuple[float, float, float, float]] = {}
    for field_name, (x1, _, x2, _) in COLUMN_BOXES.items():
        boxes[field_name] = (x1, normalized_top, x2, normalized_bottom)
    return boxes


def postprocess_row_values(row: dict[str, str]) -> dict[str, str]:
    day_int = int(row["day"])

    if row["regular_hours"] == str(day_int):
        row["regular_hours"] = ""
    if row["ot_hours"] == str(day_int):
        row["ot_hours"] = ""

    if row["rope_access_allowance"] in {"0", ""}:
        row["rope_access_allowance"] = ""

    if row["rope_access_allowance"] in {"810", "910", "016", "010"}:
        row["rope_access_allowance"] = row["rope_access_allowance"][-2:].lstrip("0") or "0"

    if row["ot_hours"] and row["ot_hours"] not in {"1", "2", "3", "4", "5", "6", "7", "8", "10", "12"}:
        row["ot_hours"] = ""

    if row["regular_hours"] not in {"", "4", "8", "12", "P.H"}:
        row["regular_hours"] = ""

    return row


def parse_table_fixed_grid(
    ocr: PaddleOCR,
    card_image: np.ndarray,
    use_cls: bool,
    raw_lines: list[OCRLine],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    height, width = card_image.shape[:2]
    table_top, table_bottom = detect_table_body_bounds(raw_lines, width, height)
    table_boxes = build_table_column_boxes(table_top, table_bottom, height)

    split_columns: dict[str, list[np.ndarray]] = {}
    for field_name, box in table_boxes.items():
        split_columns[field_name] = split_column_into_rows(
            crop_box(card_image, box),
            TABLE_ROW_COUNT,
        )

    rows: list[dict[str, str]] = []
    review_flags: list[dict[str, str]] = []

    for row_index in range(TABLE_ROW_COUNT):
        row = {"day": str(row_index + 1).zfill(2)}
        for field_name, cells in split_columns.items():
            if field_name == "day":
                continue
            cell_text, _ = ocr_text(ocr, cells[row_index], use_cls)
            row[field_name] = normalize_cell_value(field_name, cell_text)

        row = postprocess_row_values(row)
        row["allowance_total"] = calculate_allowance_total(row)
        rows.append(row)

        if row["regular_hours"] not in {"", "8", "4", "12", "P.H"}:
            review_flags.append(
                {
                    "issue": "regular_hours_needs_review",
                    "details": f"Day {row['day']} regular hours OCR value: {row['regular_hours'] or '[blank]'}",
                }
            )
        if row["ot_hours"] and not re.fullmatch(r"\d+(?:\.\d+)?", row["ot_hours"]):
            review_flags.append(
                {
                    "issue": "ot_hours_needs_review",
                    "details": f"Day {row['day']} OT OCR value: {row['ot_hours']}",
                }
            )
        if row["rope_access_allowance"] and not re.fullmatch(r"\d+(?:\.\d+)?", row["rope_access_allowance"]):
            review_flags.append(
                {
                    "issue": "rope_access_needs_review",
                    "details": f"Day {row['day']} rope access OCR value: {row['rope_access_allowance']}",
                }
            )

    return rows, review_flags


def parse_header(ocr: PaddleOCR, card_image: np.ndarray, use_cls: bool) -> dict[str, Any]:
    title_text, title_score = ocr_text(ocr, crop_box(card_image, HEADER_BOXES["document_title"]), use_cls)
    name_text, name_score = ocr_text(ocr, crop_box(card_image, HEADER_BOXES["employee_name"]), use_cls)
    wp_text, wp_score = ocr_text(ocr, crop_box(card_image, HEADER_BOXES["wp_fin_no"]), use_cls)
    month_year_text, month_year_score = ocr_text(ocr, crop_box(card_image, HEADER_BOXES["month_year"]), use_cls)
    employee_id_text, employee_id_score = ocr_text(ocr, crop_box(card_image, HEADER_BOXES["employee_id"]), use_cls)

    month, year = extract_month_year(month_year_text)
    employee_id = strip_label(
        employee_id_text,
        [r"employee\s*id\s*no", r"employee\s*id", r"id\s*no", r"id"],
    )
    wp_fin_no = strip_label(wp_text, [r"wp\s*/?\s*fin\s*no", r"wp\s*fin", r"no"])
    employee_name = strip_label(name_text, [r"name"])

    return {
        "document_title": clean_text(title_text),
        "employee_name": employee_name,
        "wp_fin_no": wp_fin_no,
        "employee_id": employee_id,
        "month": month,
        "year": year,
        "raw_header_text": {
            "document_title": title_text,
            "employee_name": name_text,
            "wp_fin_no": wp_text,
            "month_year": month_year_text,
            "employee_id": employee_id_text,
        },
        "header_scores": {
            "document_title": title_score,
            "employee_name": name_score,
            "wp_fin_no": wp_score,
            "month_year": month_year_score,
            "employee_id": employee_id_score,
        },
    }


def parse_table(ocr: PaddleOCR, card_image: np.ndarray, use_cls: bool) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    split_columns: dict[str, list[np.ndarray]] = {}
    for field_name, box in COLUMN_BOXES.items():
        split_columns[field_name] = split_column_into_rows(
            crop_box(card_image, box),
            TABLE_ROW_COUNT,
        )

    rows: list[dict[str, str]] = []
    review_flags: list[dict[str, str]] = []

    for row_index in range(TABLE_ROW_COUNT):
        row: dict[str, str] = {"day": ""}
        for field_name, cells in split_columns.items():
            cell_text, _ = ocr_text(ocr, cells[row_index], use_cls)
            row[field_name] = normalize_cell_value(field_name, cell_text)

        day = row.get("day", "")
        if not day:
            continue

        row["day"] = day.zfill(2)
        row["allowance_total"] = calculate_allowance_total(row)
        rows.append(row)

        non_day_values = [
            row[key]
            for key in row
            if key not in {"day", "allowance_total", "supervisor_signature"}
        ]
        if not any(value for value in non_day_values):
            review_flags.append(
                {
                    "issue": "empty_row",
                    "details": f"Detected day {row['day']} but the work cells were blank.",
                }
            )

    return rows, review_flags


def parse_totals(ocr: PaddleOCR, card_image: np.ndarray, use_cls: bool) -> dict[str, str]:
    totals: dict[str, str] = {}
    for field_name, box in TOTAL_BOXES.items():
        text, _ = ocr_text(ocr, crop_box(card_image, box), use_cls)
        totals[field_name] = normalize_cell_value(field_name, text)
    return totals


def parse_footer(ocr: PaddleOCR, card_image: np.ndarray, use_cls: bool) -> dict[str, str]:
    footer: dict[str, str] = {}
    for field_name, box in FOOTER_BOXES.items():
        text, _ = ocr_text(ocr, crop_box(card_image, box), use_cls)
        footer[field_name] = clean_text(text)
    return footer


def calculate_allowance_total(row: dict[str, str]) -> str:
    total = 0.0
    for key in (
        "ot_hours",
        "rope_access_allowance",
        "transport_allowance",
        "night_shift_allowance",
        "food_allowance",
    ):
        value = row.get(key, "")
        try:
            if value:
                total += float(value)
        except ValueError:
            continue
    if total.is_integer():
        return str(int(total))
    return f"{total:.2f}".rstrip("0").rstrip(".")


def extraction_score(parsed: dict[str, Any]) -> int:
    header = parsed["header"]
    rows = parsed["daily_rows"]
    totals = parsed["totals"]

    score = 0
    for key in ("document_title", "employee_name", "employee_id", "month", "year"):
        if header.get(key):
            score += 10
    score += len(rows) * 5
    score += sum(1 for value in totals.values() if value) * 2
    return score


def table_quality(daily_rows: list[dict[str, str]], totals: dict[str, str]) -> int:
    score = len(daily_rows) * 5
    for row in daily_rows:
        score += sum(
            1
            for key, value in row.items()
            if key not in {"day", "allowance_total", "supervisor_signature"} and value
        )
    score += sum(1 for value in totals.values() if value) * 2
    return score


def process_image(
    image_path: Path,
    ocr: PaddleOCR,
    use_cls: bool,
) -> dict[str, Any]:
    image = load_image(image_path)
    best_result: dict[str, Any] | None = None
    best_score = -1
    best_header: dict[str, Any] | None = None
    best_header_score = -1

    for orientation_name, candidate in orientation_candidates(image):
        for variant_name, variant_image in image_variants(candidate):
            raw_lines = ocr_lines(ocr, variant_image, use_cls)
            width = variant_image.shape[1]
            height = variant_image.shape[0]
            header = parse_header_from_lines(raw_lines, width, height)
            daily_rows, table_flags = parse_table_fixed_grid(ocr, variant_image, use_cls, raw_lines)
            totals = parse_totals_from_lines(raw_lines, width, height)
            footer = parse_footer_from_lines(raw_lines, width, height)

            current_header_score = header_quality(header)
            if current_header_score > best_header_score:
                best_header_score = current_header_score
                best_header = header

            parsed = {
                "source_file": image_path.name,
                "selected_orientation": orientation_name,
                "selected_variant": variant_name,
                "header": header,
                "daily_rows": daily_rows,
                "totals": totals,
                "footer": footer,
                "review_flags": table_flags,
                "raw_lines": [asdict(line) for line in raw_lines],
            }
            score = table_quality(daily_rows, totals)
            if score > best_score:
                best_score = score
                best_result = parsed

    if best_result is None:
        raise ValueError(f"Could not OCR image: {image_path}")

    if best_header is not None:
        best_result["header"] = best_header

    if not best_result["header"]["employee_name"]:
        best_result["review_flags"].append(
            {
                "issue": "missing_employee_name",
                "details": "Header OCR could not confidently read the employee name.",
            }
        )
    if not best_result["header"]["employee_id"]:
        best_result["review_flags"].append(
            {
                "issue": "missing_employee_id",
                "details": "Header OCR could not confidently read the employee ID.",
            }
        )
    if not best_result["daily_rows"]:
        best_result["review_flags"].append(
            {
                "issue": "no_daily_rows_found",
                "details": "The table structure was detected, but row extraction returned no valid day values.",
            }
        )
    return best_result


def write_json_output(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_text_output(payload: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    header = payload["header"]
    lines.append(f"Title: {header.get('document_title', '')}")
    lines.append(f"Employee Name: {header.get('employee_name', '')}")
    lines.append(f"WP/FIN No: {header.get('wp_fin_no', '')}")
    lines.append(f"Employee ID: {header.get('employee_id', '')}")
    lines.append(f"Month: {header.get('month', '')}")
    lines.append(f"Year: {header.get('year', '')}")
    lines.append("")
    lines.append("Daily Rows:")
    for row in payload["daily_rows"]:
        lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_daily_rows_csv(payload: dict[str, Any], path: Path) -> None:
    header = payload.get("header", {})
    rows = payload.get("daily_rows", [])
    fieldnames = [
        "source_file",
        "employee_name",
        "wp_fin_no",
        "employee_id",
        "month",
        "year",
        "day",
        "regular_hours",
        "ot_hours",
        "rope_access_allowance",
        "transport_allowance",
        "night_shift_allowance",
        "food_allowance",
        "job_site",
        "supervisor_signature",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_file": payload.get("source_file", ""),
                    "employee_name": header.get("employee_name", ""),
                    "wp_fin_no": header.get("wp_fin_no", ""),
                    "employee_id": header.get("employee_id", ""),
                    "month": header.get("month", ""),
                    "year": header.get("year", ""),
                    "day": row.get("day", ""),
                    "regular_hours": row.get("regular_hours", ""),
                    "ot_hours": row.get("ot_hours", ""),
                    "rope_access_allowance": row.get("rope_access_allowance", ""),
                    "transport_allowance": row.get("transport_allowance", ""),
                    "night_shift_allowance": row.get("night_shift_allowance", ""),
                    "food_allowance": row.get("food_allowance", ""),
                    "job_site": row.get("job_site", ""),
                    "supervisor_signature": row.get("supervisor_signature", ""),
                }
            )


def export_summary(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "source_file",
        "employee_name",
        "employee_id",
        "month",
        "year",
        "days_detected",
        "review_flag_count",
        "json_output",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    recursive = not args.no_recursive
    use_cls = not args.disable_angle_cls

    images = iter_images(input_path, recursive=recursive)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {} if args.rescan_all else load_manifest(output_dir)

    print(f"Input path: {input_path}")
    print(f"Images found: {len(images)}")
    print(f"Output dir: {output_dir}")
    print(f"Language: {args.lang}")
    print(f"Recursive scan: {recursive}")
    print(f"Rescan all: {args.rescan_all}")
    print("")

    ocr = build_ocr_engine(args)
    summary_rows: list[dict[str, Any]] = []
    processed_count = 0
    skipped_count = 0

    for image_path in images:
        if not args.rescan_all and should_skip_image(image_path, manifest):
            skipped_count += 1
            print(f"Skipped: {image_path.name} | already processed and unchanged")
            continue

        payload = process_image(image_path=image_path, ocr=ocr, use_cls=use_cls)

        header = payload["header"]
        stem = safe_file_stem(header.get("employee_name", ""), image_path.stem)
        json_path = unique_output_path(output_dir, stem, ".json")
        txt_path = unique_output_path(output_dir, stem, ".txt")
        csv_path = unique_output_path(output_dir, f"{stem}_daily_rows", ".csv")
        write_json_output(payload, json_path)
        write_text_output(payload, txt_path)
        write_daily_rows_csv(payload, csv_path)
        update_manifest_entry(manifest, image_path, payload, json_path, txt_path)
        processed_count += 1

        summary_rows.append(
            {
                "source_file": image_path.name,
                "employee_name": header.get("employee_name", ""),
                "employee_id": header.get("employee_id", ""),
                "month": header.get("month", ""),
                "year": header.get("year", ""),
                "days_detected": len(payload["daily_rows"]),
                "review_flag_count": len(payload["review_flags"]),
                "json_output": json_path.name,
            }
        )
        print(
            f"Processed: {image_path.name} | days={len(payload['daily_rows'])} | flags={len(payload['review_flags'])}"
        )

    export_summary(summary_rows, output_dir)
    save_manifest(output_dir, manifest)

    print("")
    print(f"Images processed: {processed_count}")
    print(f"Images skipped: {skipped_count}")
    print(f"Done. Summary CSV: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
