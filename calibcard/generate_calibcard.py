"""
Bal-Setu: RBSK-CalibCard Procedural Generator
Generates exact vector SVG, high-resolution PNG, and JSON specifications
for the ₹2 laminated pocket RBSK-CalibCard (Front: Clinical Calibration, Back: Fine-Motor Tracing).
Standard ID-1 dimensions: 85.60 mm x 53.98 mm (Aspect Ratio 1.5858).
"""

import json
import math
import os
import cv2
import cv2.aruco as aruco
import numpy as np


def get_aruco_svg_path(marker_id: int, x: float, y: float, size: float, dict_type=aruco.DICT_4X4_50) -> str:
    """Generate SVG rect elements for a 4x4 ArUco marker."""
    dictionary = aruco.getPredefinedDictionary(dict_type)
    marker_img = aruco.generateImageMarker(dictionary, marker_id, 6)  # 4x4 + 1 border each side = 6x6
    rects = []
    cell_size = size / 6.0
    for r in range(6):
        for c in range(6):
            if marker_img[r, c] == 0:  # Black cell
                px = x + c * cell_size
                py = y + r * cell_size
                rects.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{cell_size:.2f}" height="{cell_size:.2f}" fill="#000000" />')
    return "\n      ".join(rects)


def generate_calibcard_front_svg(output_path: str):
    """Generate Front face: ArUco fiducials, 6-patch color strip, metric ruler, cartoon fixation target."""
    width_mm = 85.60
    height_mm = 53.98
    
    # ArUco markers at corners (IDs 0, 1, 2, 3)
    marker_size = 10.0  # 10 mm
    margin = 3.0
    
    # 6 Color Patches (Width 6.5 mm, Height 8.0 mm each)
    patches = [
        {"name": "Neutral White", "rgb": "#F5F5F5", "code": "WHT", "rgb_val": [245, 245, 245]},
        {"name": "Mid-Gray 18%", "rgb": "#767676", "code": "GRY", "rgb_val": [118, 118, 118]},
        {"name": "Dark Gray", "rgb": "#2D2D2D", "code": "BLK", "rgb_val": [45, 45, 45]},
        {"name": "Crimson Red", "rgb": "#B4141E", "code": "RED", "rgb_val": [180, 20, 30]},
        {"name": "Warm Skin (IV)", "rgb": "#C68669", "code": "SKN1", "rgb_val": [198, 134, 105]},
        {"name": "Deep Skin (V/VI)", "rgb": "#784B37", "code": "SKN2", "rgb_val": [120, 75, 55]},
    ]
    
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_mm} {height_mm}" width="{width_mm}mm" height="{height_mm}mm">
  <defs>
    <style>
      .title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: bold; font-size: 2.2px; fill: #1E293B; }}
      .sub {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 1.4px; fill: #475569; }}
      .hindi {{ font-family: "Noto Sans Devanagari", "Kohinoor Devanagari", sans-serif; font-weight: bold; font-size: 2.0px; fill: #0F172A; }}
      .micro {{ font-family: monospace; font-size: 1.1px; fill: #64748B; }}
      .scale-lbl {{ font-family: monospace; font-size: 1.2px; fill: #334155; }}
    </style>
  </defs>

  <!-- Card Body -->
  <rect x="0" y="0" width="{width_mm}" height="{height_mm}" rx="3.0" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="0.5" />

  <!-- ArUco Corner Markers -->
  <!-- ID 0: Top-Left -->
  <g id="aruco-0">
    {get_aruco_svg_path(0, margin, margin, marker_size)}
  </g>
  <!-- ID 1: Top-Right -->
  <g id="aruco-1">
    {get_aruco_svg_path(1, width_mm - margin - marker_size, margin, marker_size)}
  </g>
  <!-- ID 2: Bottom-Right -->
  <g id="aruco-2">
    {get_aruco_svg_path(2, width_mm - margin - marker_size, height_mm - margin - marker_size, marker_size)}
  </g>
  <!-- ID 3: Bottom-Left -->
  <g id="aruco-3">
    {get_aruco_svg_path(3, margin, height_mm - margin - marker_size, marker_size)}
  </g>

  <!-- Header Banner -->
  <text x="16" y="6.0" class="hindi">बाल-सेतु | Bal-Setu RBSK-CalibCard</text>
  <text x="16" y="8.2" class="sub">NHM Uttar Pradesh • Clinical Optical Reference v1.0</text>
  <text x="16" y="10.2" class="micro">Zero-Raw-Media DPDP Edge-AI Calibrator • Standard ID-1</text>

  <!-- Central High-Contrast Cartoon Fixation Target (Locks child gaze straight ahead) -->
  <g id="fixation-target" transform="translate(42.8, 25.5)">
    <!-- Outer friendly halo -->
    <circle cx="0" cy="0" r="10.5" fill="#FEF08A" stroke="#EAB308" stroke-width="0.6" />
    <circle cx="0" cy="0" r="8.0" fill="#38BDF8" stroke="#0284C7" stroke-width="0.5" />
    <!-- White Sclera -->
    <circle cx="0" cy="0" r="5.5" fill="#FFFFFF" stroke="#0F172A" stroke-width="0.4" />
    <!-- High-Contrast Pupil with Hirschberg reflection dot -->
    <circle cx="0" cy="0" r="3.2" fill="#0F172A" />
    <circle cx="-1.0" cy="-1.0" r="1.0" fill="#FFFFFF" />
    <!-- Cute friendly eye elements -->
    <circle cx="1.2" cy="1.2" r="0.5" fill="#FFFFFF" />
    <!-- Starburst smile guide -->
    <path d="M -4 4 Q 0 7 4 4" fill="none" stroke="#E11D48" stroke-width="0.6" stroke-linecap="round" />
    <text x="0" y="13.2" text-anchor="middle" class="hindi" font-size="1.5px">यहाँ देखें (Look Here)</text>
  </g>

  <!-- 6-Patch Color Constancy Strip (Left Column) -->
  <g id="color-strip" transform="translate(16, 13.5)">
    <rect x="-1" y="-1" width="13" height="27" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.3" rx="1.0" />
"""
    # Lay out 6 color patches in 2 columns of 3
    for i, p in enumerate(patches):
        col = i % 2
        row = i // 2
        px = col * 5.8
        py = row * 8.2
        svg += f"""    <rect x="{px:.2f}" y="{py:.2f}" width="5.0" height="6.5" rx="0.6" fill="{p['rgb']}" stroke="#64748B" stroke-width="0.2" />
    <text x="{px + 2.5:.2f}" y="{py + 7.7:.2f}" text-anchor="middle" class="micro">{p['code']}</text>
"""
    svg += """  </g>

  <!-- Right Information Block: Clinical Guidance -->
  <g id="clinical-notes" transform="translate(58, 13.5)">
    <rect x="0" y="0" width="14" height="25" fill="#F1F5F9" stroke="#CBD5E1" stroke-width="0.3" rx="1.0" />
    <text x="7" y="3.2" text-anchor="middle" class="title" font-size="1.4px">CLINICAL USES</text>
    <text x="1.5" y="6.5" class="micro">• Hirschberg CLR</text>
    <text x="1.5" y="9.5" class="micro">• Red-Reflex Gate</text>
    <text x="1.5" y="12.5" class="micro">• Mucosal Pallor</text>
    <text x="1.5" y="15.5" class="micro">• Skin Tone Calib</text>
    <text x="1.5" y="18.5" class="micro">• Cleft / Clubfoot</text>
    <text x="1.5" y="21.5" class="micro">• Cost: ₹2-₹5/card</text>
  </g>

  <!-- Bottom Metric Scale (50 mm in 1 mm and 5 mm ticks) -->
  <g id="metric-scale" transform="translate(17.8, 48.0)">
    <!-- Base line -->
    <line x1="0" y1="0" x2="50" y2="0" stroke="#0F172A" stroke-width="0.4" />
"""
    # 50 mm tick marks
    for mm in range(51):
        tick_h = 2.8 if (mm % 10 == 0) else (1.8 if mm % 5 == 0 else 1.0)
        svg += f'    <line x1="{mm}" y1="0" x2="{mm}" y2="-{tick_h}" stroke="#0F172A" stroke-width="0.25" />\n'
        if mm % 10 == 0:
            svg += f'    <text x="{mm}" y="2.5" text-anchor="middle" class="scale-lbl">{mm//10}</text>\n'
    svg += f"""    <text x="53.0" y="0.8" class="scale-lbl">cm</text>
  </g>
</svg>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)


def generate_calibcard_back_svg(output_path: str):
    """Generate Back face: Fine-Motor Tracing Template framing 3 age-graded patterns."""
    width_mm = 85.60
    height_mm = 53.98
    marker_size = 10.0
    margin = 3.0

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_mm} {height_mm}" width="{width_mm}mm" height="{height_mm}mm">
  <defs>
    <style>
      .title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: bold; font-size: 2.0px; fill: #1E293B; }}
      .sub {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 1.4px; fill: #475569; }}
      .hindi {{ font-family: "Noto Sans Devanagari", "Kohinoor Devanagari", sans-serif; font-weight: bold; font-size: 1.8px; fill: #0F172A; }}
      .micro {{ font-family: monospace; font-size: 1.1px; fill: #64748B; }}
      .pattern-label {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-weight: bold; font-size: 1.5px; fill: #2563EB; }}
    </style>
  </defs>

  <!-- Card Body -->
  <rect x="0" y="0" width="{width_mm}" height="{height_mm}" rx="3.0" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="0.5" />

  <!-- ArUco Corner Markers -->
  <!-- ID 0: Top-Left -->
  <g id="aruco-0">
    {get_aruco_svg_path(0, margin, margin, marker_size)}
  </g>
  <!-- ID 1: Top-Right -->
  <g id="aruco-1">
    {get_aruco_svg_path(1, width_mm - margin - marker_size, margin, marker_size)}
  </g>
  <!-- ID 2: Bottom-Right -->
  <g id="aruco-2">
    {get_aruco_svg_path(2, width_mm - margin - marker_size, height_mm - margin - marker_size, marker_size)}
  </g>
  <!-- ID 3: Bottom-Left -->
  <g id="aruco-3">
    {get_aruco_svg_path(3, margin, height_mm - margin - marker_size, marker_size)}
  </g>

  <!-- Header Banner -->
  <text x="16" y="5.8" class="hindi">बाल-सेतु | फाइन-मोटर ट्रेसिंग कार्ड (Fine-Motor Template)</text>
  <text x="16" y="8.0" class="sub">UC-5 Daily Home Activity & Frontline Assessment • Ages 3-6</text>
  <text x="16" y="10.0" class="micro">Pencil Trace → Phone Snap → On-Device Homography & Hausdorff Evaluation</text>

  <!-- Pattern 1: Circle (Ages 3-4, Diameter 18 mm) -->
  <g id="pattern-circle" transform="translate(23, 27)">
    <rect x="-8.5" y="-14" width="17" height="27" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.3" rx="1.0" />
    <text x="0" y="-10.5" text-anchor="middle" class="pattern-label">1. वृत्त (Circle)</text>
    <text x="0" y="-8.5" text-anchor="middle" class="micro">Age 3-4 yrs</text>
    <!-- Start arrow & dot -->
    <circle cx="0" cy="-6" r="0.6" fill="#16A34A" />
    <circle cx="0" cy="3" r="6.0" fill="none" stroke="#94A3B8" stroke-width="0.8" stroke-dasharray="1.2, 0.8" />
    <text x="0" y="11.5" text-anchor="middle" class="micro">Ø 12mm</text>
  </g>

  <!-- Pattern 2: Cross (Ages 4-5, 14 mm x 14 mm) -->
  <g id="pattern-cross" transform="translate(42.8, 27)">
    <rect x="-8.5" y="-14" width="17" height="27" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.3" rx="1.0" />
    <text x="0" y="-10.5" text-anchor="middle" class="pattern-label">2. धन (Cross)</text>
    <text x="0" y="-8.5" text-anchor="middle" class="micro">Age 4-5 yrs</text>
    <!-- Vertical line -->
    <line x1="0" y1="-3" x2="0" y2="9" stroke="#94A3B8" stroke-width="0.8" stroke-dasharray="1.2, 0.8" />
    <!-- Horizontal line -->
    <line x1="-6" y1="3" x2="6" y2="3" stroke="#94A3B8" stroke-width="0.8" stroke-dasharray="1.2, 0.8" />
    <!-- Green start points -->
    <circle cx="0" cy="-3" r="0.6" fill="#16A34A" />
    <circle cx="-6" cy="3" r="0.6" fill="#16A34A" />
    <text x="0" y="11.5" text-anchor="middle" class="micro">12×12mm</text>
  </g>

  <!-- Pattern 3: Zigzag Wave (Ages 5-6, Span 18 mm) -->
  <g id="pattern-zigzag" transform="translate(62.6, 27)">
    <rect x="-8.5" y="-14" width="17" height="27" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.3" rx="1.0" />
    <text x="0" y="-10.5" text-anchor="middle" class="pattern-label">3. तरंग (Zigzag)</text>
    <text x="0" y="-8.5" text-anchor="middle" class="micro">Age 5-6 yrs</text>
    <!-- Zigzag Path -->
    <path d="M -6 5 L -3 1 L 0 5 L 3 1 L 6 5" fill="none" stroke="#94A3B8" stroke-width="0.8" stroke-dasharray="1.2, 0.8" />
    <!-- Green start dot -->
    <circle cx="-6" cy="5" r="0.6" fill="#16A34A" />
    <text x="0" y="11.5" text-anchor="middle" class="micro">4 peaks / 12mm</text>
  </g>

  <!-- Footer Instructions -->
  <g id="instructions" transform="translate(16, 44.5)">
    <text x="0" y="2.0" class="hindi">निर्देश: पेंसिल से बिंदुकृत रेखाओं पर फेरें। फोन से फोटो खींचें।</text>
    <text x="0" y="4.5" class="sub">Rule: Follow dotted lines with pencil. Offline AI measures deviation &amp; tremor score.</text>
  </g>
</svg>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)


def export_specs_json(output_path: str):
    """Export complete geometric and colorimetry ground truth."""
    specs = {
        "standard": "ISO/IEC 7810 ID-1",
        "dimensions_mm": {
            "width": 85.60,
            "height": 53.98,
            "corner_radius": 3.18,
            "thickness": 0.76
        },
        "aruco_fiducials": {
            "dictionary": "DICT_4X4_50",
            "marker_size_mm": 10.0,
            "margin_mm": 3.0,
            "corners": {
                "top_left": {"id": 0, "x_mm": 3.0, "y_mm": 3.0},
                "top_right": {"id": 1, "x_mm": 72.60, "y_mm": 3.0},
                "bottom_right": {"id": 2, "x_mm": 72.60, "y_mm": 40.98},
                "bottom_left": {"id": 3, "x_mm": 3.0, "y_mm": 40.98}
            }
        },
        "color_constancy_strip": [
            {"name": "Neutral White", "code": "WHT", "rgb": [245, 245, 245], "lab": [96.0, 0.0, 0.0], "tolerance_de2000": 3.0},
            {"name": "Mid-Gray 18%", "code": "GRY", "rgb": [118, 118, 118], "lab": [49.5, 0.0, 0.0], "tolerance_de2000": 3.0},
            {"name": "Dark Gray", "code": "BLK", "rgb": [45, 45, 45], "lab": [18.5, 0.0, 0.0], "tolerance_de2000": 3.0},
            {"name": "Calibrated Crimson Red", "code": "RED", "rgb": [180, 20, 30], "purpose": "Mucosal pallor / conjunctiva reference"},
            {"name": "Warm Indian Skin", "code": "SKN1", "rgb": [198, 134, 105], "fitzpatrick": "Type IV", "purpose": "Baseline melanin calibration"},
            {"name": "Deep Indian Skin", "code": "SKN2", "rgb": [120, 75, 55], "fitzpatrick": "Type V/VI", "purpose": "Deep skin melanin calibration"}
        ],
        "metric_scale": {
            "length_mm": 50.0,
            "tick_divisions_mm": 1.0,
            "major_ticks_mm": 10.0,
            "position": {"x_start_mm": 17.8, "y_mm": 48.0}
        },
        "fixation_target": {
            "type": "High-contrast concentric cartoon eye with Hirschberg central pupil target",
            "center_mm": {"x": 42.8, "y": 25.5},
            "outer_diameter_mm": 21.0,
            "pupil_diameter_mm": 6.4,
            "hirschberg_reflection_dot_mm": 2.0
        },
        "back_tracing_patterns": {
            "pattern_1": {"name": "Circle", "age_range": "3-4 years", "target_shape": "circle", "radius_mm": 6.0, "center_mm": [23.0, 30.0]},
            "pattern_2": {"name": "Cross", "age_range": "4-5 years", "target_shape": "plus", "arm_length_mm": 12.0, "center_mm": [42.8, 30.0]},
            "pattern_3": {"name": "Zigzag", "age_range": "5-6 years", "target_shape": "zigzag", "span_mm": 12.0, "peaks": 4, "center_mm": [62.6, 30.0]}
        },
        "procurement_economics": {
            "unit_cost_inr": 2.25,
            "statewide_3_6_lakh_frontline_kits_inr": 810000.0,
            "per_child_amortized_cost_inr": 0.000095
        }
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2, ensure_ascii=False)


def render_calibcard_raster_images(calibcard_dir: str):
    """
    Renders synthetic high-res raster PNGs (front and back) representing the CalibCard
    at 300 DPI (1011 x 638 pixels) so OpenCV routines can load and test directly.
    """
    width_px = 1011
    height_px = 638
    
    # 1. Front Face
    front_img = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255
    # Rounded border simulated
    cv2.rectangle(front_img, (6, 6), (width_px - 6, height_px - 6), (200, 210, 220), 2)
    
    # Draw ArUco markers 0, 1, 2, 3
    d = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    marker_px = 118
    margin_px = 35
    
    m0 = aruco.generateImageMarker(d, 0, marker_px)
    m1 = aruco.generateImageMarker(d, 1, marker_px)
    m2 = aruco.generateImageMarker(d, 2, marker_px)
    m3 = aruco.generateImageMarker(d, 3, marker_px)
    
    front_img[margin_px:margin_px+marker_px, margin_px:margin_px+marker_px] = cv2.cvtColor(m0, cv2.COLOR_GRAY2BGR)
    front_img[margin_px:margin_px+marker_px, width_px-margin_px-marker_px:width_px-margin_px] = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
    front_img[height_px-margin_px-marker_px:height_px-margin_px, width_px-margin_px-marker_px:width_px-margin_px] = cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR)
    front_img[height_px-margin_px-marker_px:height_px-margin_px, margin_px:margin_px+marker_px] = cv2.cvtColor(m3, cv2.COLOR_GRAY2BGR)
    
    # Draw Text
    cv2.putText(front_img, "Bal-Setu RBSK-CalibCard v1.0", (180, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 30, 40), 2)
    cv2.putText(front_img, "National Health Mission - Uttar Pradesh", (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 90, 100), 1)
    
    # Draw 6 Color Patches (BGR format)
    patches_bgr = [
        ("WHT", (245, 245, 245)),
        ("GRY", (118, 118, 118)),
        ("BLK", (45, 45, 45)),
        ("RED", (30, 20, 180)),
        ("SKN1", (105, 134, 198)),
        ("SKN2", (55, 75, 120)),
    ]
    
    start_x = 180
    start_y = 150
    p_w, p_h = 60, 75
    gap_x, gap_y = 15, 20
    
    for idx, (label, color) in enumerate(patches_bgr):
        c = idx % 2
        r = idx // 2
        x1 = start_x + c * (p_w + gap_x)
        y1 = start_y + r * (p_h + gap_y)
        cv2.rectangle(front_img, (x1, y1), (x1 + p_w, y1 + p_h), color, -1)
        cv2.rectangle(front_img, (x1, y1), (x1 + p_w, y1 + p_h), (100, 100, 100), 1)
        cv2.putText(front_img, label, (x1 + 10, y1 + p_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
        
    # Draw Fixation Target (Center of Card)
    cx, cy = 505, 301
    cv2.circle(front_img, (cx, cy), 120, (140, 240, 254), -1) # yellow halo
    cv2.circle(front_img, (cx, cy), 95, (248, 189, 56), -1)  # sky blue
    cv2.circle(front_img, (cx, cy), 65, (255, 255, 255), -1) # sclera
    cv2.circle(front_img, (cx, cy), 38, (20, 20, 20), -1)    # pupil
    cv2.circle(front_img, (cx - 12, cy - 12), 12, (255, 255, 255), -1) # reflection
    cv2.putText(front_img, "LOOK HERE", (cx - 55, cy + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 30, 180), 2)
    
    # Metric scale at bottom
    scale_x0 = 210
    scale_y0 = 570
    scale_w = 590 # ~50mm
    cv2.line(front_img, (scale_x0, scale_y0), (scale_x0 + scale_w, scale_y0), (20, 20, 20), 2)
    for mm in range(51):
        x = int(scale_x0 + mm * (scale_w / 50.0))
        th = 30 if mm % 10 == 0 else (20 if mm % 5 == 0 else 10)
        cv2.line(front_img, (x, scale_y0), (x, scale_y0 - th), (20, 20, 20), 1)
        if mm % 10 == 0:
            cv2.putText(front_img, str(mm // 10), (x - 5, scale_y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
            
    cv2.imwrite(os.path.join(calibcard_dir, "calibcard_front.png"), front_img)

    # 2. Back Face (Tracing Patterns)
    back_img = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255
    cv2.rectangle(back_img, (6, 6), (width_px - 6, height_px - 6), (200, 210, 220), 2)
    
    # ArUco markers on back too
    back_img[margin_px:margin_px+marker_px, margin_px:margin_px+marker_px] = cv2.cvtColor(m0, cv2.COLOR_GRAY2BGR)
    back_img[margin_px:margin_px+marker_px, width_px-margin_px-marker_px:width_px-margin_px] = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
    back_img[height_px-margin_px-marker_px:height_px-margin_px, width_px-margin_px-marker_px:width_px-margin_px] = cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR)
    back_img[height_px-margin_px-marker_px:height_px-margin_px, margin_px:margin_px+marker_px] = cv2.cvtColor(m3, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(back_img, "Fine-Motor Tracing Template (UC-5 Home Activity)", (180, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 30, 40), 2)
    cv2.putText(back_img, "Bal-Setu AI Motor Kinematics Evaluator", (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 90, 100), 1)

    # 3 Patterns: Circle, Cross, Zigzag
    # Pattern 1: Circle
    p1_cx, p1_cy, p1_r = 270, 320, 70
    cv2.circle(back_img, (p1_cx, p1_cy), p1_r, (160, 160, 160), 2, lineType=cv2.LINE_AA)
    cv2.circle(back_img, (p1_cx, p1_cy - p1_r), 6, (0, 180, 0), -1) # green start
    cv2.putText(back_img, "1. Circle (Age 3-4)", (p1_cx - 70, p1_cy - p1_r - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 50, 20), 1)

    # Pattern 2: Cross
    p2_cx, p2_cy, p2_len = 505, 320, 70
    cv2.line(back_img, (p2_cx, p2_cy - p2_len), (p2_cx, p2_cy + p2_len), (160, 160, 160), 2, lineType=cv2.LINE_AA)
    cv2.line(back_img, (p2_cx - p2_len, p2_cy), (p2_cx + p2_len, p2_cy), (160, 160, 160), 2, lineType=cv2.LINE_AA)
    cv2.circle(back_img, (p2_cx, p2_cy - p2_len), 6, (0, 180, 0), -1)
    cv2.circle(back_img, (p2_cx - p2_len, p2_cy), 6, (0, 180, 0), -1)
    cv2.putText(back_img, "2. Cross (Age 4-5)", (p2_cx - 65, p2_cy - p2_len - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 50, 20), 1)

    # Pattern 3: Zigzag
    p3_cx, p3_cy = 740, 320
    pts = np.array([
        [p3_cx - 70, p3_cy + 40],
        [p3_cx - 35, p3_cy - 40],
        [p3_cx, p3_cy + 40],
        [p3_cx + 35, p3_cy - 40],
        [p3_cx + 70, p3_cy + 40]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(back_img, [pts], False, (160, 160, 160), 2, lineType=cv2.LINE_AA)
    cv2.circle(back_img, (p3_cx - 70, p3_cy + 40), 6, (0, 180, 0), -1)
    cv2.putText(back_img, "3. Zigzag (Age 5-6)", (p3_cx - 70, p3_cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 50, 20), 1)

    cv2.putText(back_img, "Instruction: Trace over dotted lines with pencil. Photograph card for instant AI score.", (180, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

    cv2.imwrite(os.path.join(calibcard_dir, "calibcard_back.png"), back_img)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    
    front_svg_path = os.path.join(base_dir, "calibcard_front.svg")
    back_svg_path = os.path.join(base_dir, "calibcard_back.svg")
    specs_json_path = os.path.join(base_dir, "calibcard_specs.json")
    
    print(f"Generating CalibCard Front SVG: {front_svg_path}")
    generate_calibcard_front_svg(front_svg_path)
    
    print(f"Generating CalibCard Back SVG: {back_svg_path}")
    generate_calibcard_back_svg(back_svg_path)
    
    print(f"Exporting CalibCard Specifications JSON: {specs_json_path}")
    export_specs_json(specs_json_path)

    print("Rendering high-res raster PNGs...")
    render_calibcard_raster_images(base_dir)
    print("CalibCard asset generation complete!")


if __name__ == "__main__":
    main()
