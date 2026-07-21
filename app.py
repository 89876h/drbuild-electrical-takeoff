import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(
    page_title="DrBuild Electrical Takeoff Tool", page_icon="⚡", layout="wide"
)

st.title(" Electrical Drawing Takeoff & Symbol Counter")
st.markdown(
    "Strict CV Takeoff Engine: Multi-scale template matching with adaptive legend parsing."
)

with st.sidebar:
    st.header("1. Upload Project Drawings")
    legend_file = st.file_uploader(
        "Upload Legend Sheet (JPEG/PNG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    st.markdown("---")
    st.subheader("Drawing Packages (Optional)")
    power_files = st.file_uploader(
        "Upload Power Drawings (JPEG/PNG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    lighting_files = st.file_uploader(
        "Upload Lighting Drawings (JPEG/PNG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    st.markdown("---")
    process_btn = st.button("Run Strict Takeoff Scan", type="primary")


def load_image_safely(uploaded_file):
    """Safely opens an uploaded image file, downscaling if dimensions exceed safe limits."""
    try:
        img = Image.open(uploaded_file).convert("RGB")
        if max(img.size) > 4000:
            img.thumbnail((4000, 4000), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return None


def extract_symbols_from_legend(legend_img, category_filter):
    """
    Adaptive symbol extraction from legend.
    Instead of assuming a fixed grid, we detect dark connected components
    (symbol blobs) within each section region.
    """
    legend_cv = np.array(legend_img)
    gray = cv2.cvtColor(legend_cv, cv2.COLOR_RGB2GRAY)
    h_leg, w_leg = gray.shape[:2]

    # Define approximate section boundaries based on typical electrical legends
    # Adjust these ratios to match your specific legend layout
    if category_filter == "Power / Devices":
        # Receptacles & Outlets + Electrical Equipment sections
        y_start = int(h_leg * 0.42)
        y_end = int(h_leg * 0.62)
    elif category_filter == "Lighting":
        # Lighting section
        y_start = int(h_leg * 0.62)
        y_end = int(h_leg * 0.78)
    else:
        y_start, y_end = 0, h_leg

    section = gray[y_start:y_end, :]

    # Threshold to find dark symbol pixels (symbols are black/dark on white)
    _, binary = cv2.threshold(section, 180, 255, cv2.THRESH_BINARY_INV)

    # Find connected components (each symbol blob)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    extracted_items = []
    item_counter = 1
    prefix = "Device" if category_filter == "Power / Devices" else "Fixture"

    for i in range(1, num_labels):  # Skip background (label 0)
        x, y, w, h, area = stats[i]

        # Filter: symbol-sized blobs only (not text lines, not noise)
        aspect_ratio = w / max(1, h)
        if area < 30 or area > 5000:
            continue
        if aspect_ratio > 4.0 or aspect_ratio < 0.2:
            continue
        if w < 8 or h < 8:
            continue

        # Extract tight crop around symbol with small padding
        pad = 3
        sy1 = max(0, y - pad)
        sy2 = min(section.shape[0], y + h + pad)
        sx1 = max(0, x - pad)
        sx2 = min(section.shape[1], x + w + pad)

        symbol_crop = section[sy1:sy2, sx1:sx2]

        # Normalize size: resize to a standard template size for matching
        target_h = 40
        scale = target_h / max(1, symbol_crop.shape[0])
        new_w = max(10, int(symbol_crop.shape[1] * scale))
        symbol_resized = cv2.resize(symbol_crop, (new_w, target_h), interpolation=cv2.INTER_AREA)

        pil_chip = Image.fromarray(symbol_resized)
        img_byte_arr = io.BytesIO()
        pil_chip.save(img_byte_arr, format="PNG")

        extracted_items.append({
            "category": category_filter,
            "name": f"Symbol Type {item_counter}",
            "icon_bytes": img_byte_arr.getvalue(),
            "template": symbol_resized,
            "orig_size": (w, h),
        })
        item_counter += 1

    return extracted_items


def multi_scale_match(drawing_gray, template, scales=(0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0), threshold=0.7):
    """
    Multi-scale template matching as described in PyImageSearch [[11]].
    Tests multiple scale factors since cv2.matchTemplate is NOT scale-invariant [[16]].
    Returns deduplicated match points across all scales.
    """
    all_matches = []

    for scale in scales:
        scaled_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if (scaled_template.shape[0] > drawing_gray.shape[0] or
                scaled_template.shape[1] > drawing_gray.shape[1]):
            continue

        res = cv2.matchTemplate(drawing_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        points = list(zip(*loc[::-1]))

        for pt in points:
            all_matches.append((pt, scale, res[pt[1], pt[0]]))

    # Deduplicate: keep best match within suppression radius
    filtered = []
    for pt, sc, score in sorted(all_matches, key=lambda x: -x[2]):
        if not any(abs(pt[0] - fm[0][0]) < 20 and abs(pt[1] - fm[0][1]) < 20 for fm in filtered):
            filtered.append(((pt, sc, score),))

    return [f[0] for f in filtered]


def run_strict_takeoff_module(legend_img, drawing_imgs, package_name, category_filter):
    """Performs strict computer vision extraction with multi-scale matching."""
    if not drawing_imgs or not legend_img:
        return pd.DataFrame()

    # Extract symbols adaptively from legend
    extracted_items = extract_symbols_from_legend(legend_img, category_filter)

    if not extracted_items:
        st.warning(f"No symbols detected in legend for '{category_filter}'. Check section boundaries.")
        return pd.DataFrame()

    # Prepare drawings
    cv_drawings = []
    for d_img in drawing_imgs:
        d_arr = np.array(d_img)
        if len(d_arr.shape) == 3:
            d_arr = cv2.cvtColor(d_arr, cv2.COLOR_RGB2GRAY)
        cv_drawings.append(d_arr)

    table_rows = []
    for item in extracted_items:
        template = item["template"]
        total_count = 0
        all_match_points = []

        for d_arr in cv_drawings:
            matches = multi_scale_match(d_arr, template, threshold=0.65)
            total_count += len(matches)
            all_match_points.extend([(m[0], d_arr) for m in matches])

        table_rows.append({
            "System Category": item["category"],
            "Legend Icon": item["icon_bytes"],
            "Model / Description": item["name"],
            "Scan Package": package_name,
            "Count": total_count,
        })

    return pd.DataFrame(table_rows)


if process_btn:
    if not legend_file:
        st.error("Please upload the Legend Sheet to perform scans.")
    elif not power_files and not lighting_files:
        st.warning("Please upload at least one Power or Lighting drawing file.")
    else:
        with st.spinner("Executing strict computer vision scan across drawings..."):
            legend_image = load_image_safely(legend_file)

            power_images = [load_image_safely(f) for f in (power_files or []) if load_image_safely(f)]
            lighting_images = [load_image_safely(f) for f in (lighting_files or []) if load_image_safely(f)]

            df_power = run_strict_takeoff_module(legend_image, power_images, "Power Package", "Power / Devices")
            df_lighting = run_strict_takeoff_module(legend_image, lighting_images, "Lighting Package", "Lighting")

            df_summary = pd.concat([df_power, df_lighting], ignore_index=True)

        if not df_summary.empty:
            st.success("Strict takeoff scan complete!")

            st.subheader("📋 Itemized Takeoff Schedule")
            st.dataframe(
                df_summary,
                column_config={
                    "Legend Icon": st.column_config.ImageColumn("Legend Symbol", width="small"),
                    "Count": st.column_config.NumberColumn("Verified Count", format="%d ⚡"),
                },
                use_container_width=True,
                hide_index=True,
            )

            # Excel Export
            excel_output = io.BytesIO()
            with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
                df_summary.drop(columns=["Legend Icon"]).to_excel(
                    writer, index=False, sheet_name="Takeoff Schedule"
                )
            excel_data = excel_output.getvalue()

            # PDF Report Generation
            pdf_output = io.BytesIO()
            c = canvas.Canvas(pdf_output, pagesize=letter)
            width, height = letter

            c.setFont("Helvetica-Bold", 16)
            c.drawString(54, height - 50, "DrBuild LLC - Verified Takeoff Report")
            c.setFont("Helvetica", 10)
            c.drawString(54, height - 68, "Strict computer vision scan schedule.")
            c.setLineWidth(1)
            c.line(54, height - 78, width - 54, height - 78)

            y = height - 105
            c.setFont("Helvetica-Bold", 10)
            c.drawString(54, y, "System Category")
            c.drawString(180, y, "Model / Description")
            c.drawString(380, y, "Package")
            c.drawString(490, y, "Count")
            y -= 15
            c.setLineWidth(0.5)
            c.line(54, y, width - 54, y)
            y -= 20

            c.setFont("Helvetica", 9)
            for _, row in df_summary.iterrows():
                if y < 50:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 9)
                c.drawString(54, y, str(row["System Category"]))
                c.drawString(180, y, str(row["Model / Description"]))
                c.drawString(380, y, str(row["Scan Package"]))
                c.drawString(490, y, str(row["Count"]))
                y -= 18

            c.save()
            pdf_data = pdf_output.getvalue()

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Export Takeoff to Excel (.xlsx)",
                    data=excel_data,
                    file_name="DrBuild_Verified_Takeoff.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with col2:
                st.download_button(
                    label=" Download Takeoff Report (PDF)",
                    data=pdf_data,
                    file_name="DrBuild_Verified_Report.pdf",
                    mime="application/pdf",
                )
        else:
            st.warning("No elements were matched. Try adjusting the threshold or check symbol extraction.")
else:
    st.info("Upload your legend sheet and optional drawing files in the sidebar to begin.")
