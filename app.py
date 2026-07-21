import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import traceback

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & LAYOUT (Always runs first)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DrBuild Electrical Takeoff Tool", 
    page_icon="⚡", 
    layout="wide"
)

st.title("⚡ Electrical Drawing Takeoff & Symbol Counter")
st.markdown(
    "Strict CV Takeoff Engine: Multi-scale template matching with adaptive legend parsing."
)

# -----------------------------------------------------------------------------
# 2. SIDEBAR INPUTS (Defined globally so they persist across reruns)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload Project Drawings")
    legend_file = st.file_uploader(
        "Upload Legend Sheet (JPEG/PNG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="legend_uploader"
    )

    st.markdown("---")
    st.subheader("Drawing Packages (Optional)")
    power_files = st.file_uploader(
        "Upload Power Drawings (JPEG/PNG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="power_uploader"
    )
    lighting_files = st.file_uploader(
        "Upload Lighting Drawings (JPEG/PNG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="lighting_uploader"
    )

    st.markdown("---")
    # Button is defined here, OUTSIDE of any if-block
    process_btn = st.button("Run Strict Takeoff Scan", type="primary")

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def load_image_safely(uploaded_file):
    """Safely opens an uploaded image file."""
    try:
        img = Image.open(uploaded_file).convert("RGB")
        if max(img.size) > 5000:
            img.thumbnail((5000, 5000), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return None

def extract_symbols_from_legend(legend_img, category_filter):
    """Adaptive symbol extraction using connected components."""
    try:
        legend_cv = np.array(legend_img)
        gray = cv2.cvtColor(legend_cv, cv2.COLOR_RGB2GRAY)
        h_leg, w_leg = gray.shape[:2]

        # Define section boundaries based on typical electrical legend layouts
        if category_filter == "Power / Devices":
            y_start = int(h_leg * 0.42)
            y_end = int(h_leg * 0.62)
        elif category_filter == "Lighting":
            y_start = int(h_leg * 0.62)
            y_end = int(h_leg * 0.78)
        else:
            y_start, y_end = 0, h_leg

        section = gray[y_start:y_end, :]
        _, binary = cv2.threshold(section, 180, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        extracted_items = []
        item_counter = 1
        prefix = "Device" if category_filter == "Power / Devices" else "Fixture"

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            aspect_ratio = w / max(1, h)
            if area < 30 or area > 5000: continue
            if aspect_ratio > 4.0 or aspect_ratio < 0.2: continue
            if w < 8 or h < 8: continue

            pad = 3
            sy1 = max(0, y - pad); sy2 = min(section.shape[0], y + h + pad)
            sx1 = max(0, x - pad); sx2 = min(section.shape[1], x + w + pad)

            symbol_crop = section[sy1:sy2, sx1:sx2]
            target_h = 40
            scale = target_h / max(1, symbol_crop.shape[0])
            new_w = max(10, int(symbol_crop.shape[1] * scale))
            symbol_resized = cv2.resize(symbol_crop, (new_w, target_h), interpolation=cv2.INTER_AREA)

            pil_chip = Image.fromarray(symbol_resized)
            img_byte_arr = io.BytesIO()
            pil_chip.save(img_byte_arr, format="PNG")

            extracted_items.append({
                "category": category_filter,
                "name": f"{prefix} Type {item_counter}",
                "icon_bytes": img_byte_arr.getvalue(),
                "template": symbol_resized,
            })
            item_counter += 1
        return extracted_items
    except Exception as e:
        st.error(f"Legend extraction failed: {str(e)}")
        return []

def multi_scale_match(drawing_gray, template, scales=(0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0), threshold=0.65):
    """Multi-scale template matching with deduplication."""
    all_matches = []
    if template is None or template.size == 0: return []

    for scale in scales:
        try:
            scaled_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scaled_template.shape[0] > drawing_gray.shape[0] or scaled_template.shape[1] > drawing_gray.shape[1]:
                continue
            
            res = cv2.matchTemplate(drawing_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            points = list(zip(*loc[::-1]))
            
            for pt in points:
                all_matches.append((pt, scale, float(res[pt[1], pt[0]])))
        except Exception:
            continue

    all_matches.sort(key=lambda x: -x[2])
    filtered = []
    for pt, sc, score in all_matches:
        too_close = False
        for fm_pt, fm_sc, fm_score in filtered:
            if abs(pt[0] - fm_pt[0]) < 20 and abs(pt[1] - fm_pt[1]) < 20:
                too_close = True; break
        if not too_close:
            filtered.append((pt, sc, score))
    return filtered

def run_strict_takeoff_module_live(legend_img, drawing_imgs, package_name, category_filter, status_ph, metrics_ph, table_ph, accumulator):
    """Live processing function."""
    if not drawing_imgs or not legend_img: return

    valid_drawings = []
    for d_img in drawing_imgs:
        if d_img is None: continue
        try:
            d_arr = np.array(d_img)
            if len(d_arr.shape) == 3: d_arr = cv2.cvtColor(d_arr, cv2.COLOR_RGB2GRAY)
            valid_drawings.append(d_arr)
        except Exception: continue

    if not valid_drawings:
        status_ph.warning(f"No valid drawings found for {package_name}")
        return

    extracted_items = extract_symbols_from_legend(legend_img, category_filter)
    if not extracted_items:
        status_ph.warning(f"No symbols detected for '{category_filter}'.")
        return

    total_drawings = len(valid_drawings)
    cumulative_scanned = 0
    cumulative_matches = sum(item['Count'] for item in accumulator.values())
    
    for idx, d_arr in enumerate(valid_drawings, 1):
        drawing_matches = 0
        
        for item in extracted_items:
            try:
                matches = multi_scale_match(d_arr, item["template"], threshold=0.65)
                symbol_key = item["name"]
                
                if symbol_key not in accumulator:
                    accumulator[symbol_key] = {
                        "System Category": item["category"],
                        "Legend Icon": item["icon_bytes"],
                        "Model / Description": symbol_key,
                        "Scan Package": package_name,
                        "Count": 0,
                    }
                
                count_increment = len(matches)
                accumulator[symbol_key]["Count"] += count_increment
                drawing_matches += count_increment
            except Exception as e:
                print(f"Match error: {e}")
                continue

        cumulative_scanned += 1
        cumulative_matches += drawing_matches

        # Update Metrics Dashboard
        metrics_ph.markdown(
            f"""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; text-align:center;">
                    <div>
                        <div style="font-size:12px; color:#666;">📄 DRAWINGS SCANNED</div>
                        <div style="font-size:24px; font-weight:bold; color:#1f77b4;">{cumulative_scanned}/{total_drawings}</div>
                    </div>
                    <div>
                        <div style="font-size:12px; color:#666;"> TOTAL MATCHES FOUND</div>
                        <div style="font-size:24px; font-weight:bold; color:#2ca02c;">{cumulative_matches}</div>
                    </div>
                    <div>
                        <div style="font-size:12px; color:#666;">🔍 ACTIVE TEMPLATES</div>
                        <div style="font-size:24px; font-weight:bold; color:#ff7f0e;">{len(extracted_items)}</div>
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Update Status Text
        status_text = (
            f"**Scanning:** {package_name}\n"
            f"**Progress:** Drawing {idx}/{total_drawings} | +{drawing_matches} new matches\n"
            f"**Cumulative:** {cumulative_scanned} scanned → {cumulative_matches} total matches\n"
            f"**Status:** Matching scale variants..."
        )
        status_ph.info(status_text)

        # Update Table
        if accumulator:
            live_df = pd.DataFrame(list(accumulator.values()))
            try:
                table_ph.dataframe(
                    live_df,
                    column_config={
                        "Legend Icon": st.column_config.ImageColumn("Legend Symbol", width="small"),
                        "Count": st.column_config.NumberColumn("Verified Count", format="%d ⚡"),
                    },
                    use_container_width=True, hide_index=True,
                )
            except Exception:
                table_ph.dataframe(live_df.drop(columns=["Legend Icon"], errors="ignore"))

# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION LOGIC
# -----------------------------------------------------------------------------
# Initialize session state for results if not present
if 'results_data' not in st.session_state:
    st.session_state.results_data = {}
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False

# The button variable 'process_btn' is now GUARANTEED to exist because it's in the sidebar block above
if process_btn:
    # Reset state for new scan
    st.session_state.results_data = {}
    st.session_state.scan_complete = False
    
    if not legend_file:
        st.error("Please upload the Legend Sheet to perform scans.")
    elif not power_files and not lighting_files:
        st.warning("Please upload at least one Power or Lighting drawing file.")
    else:
        # Create placeholders
        status_box = st.empty()
        metrics_box = st.empty()
        table_box = st.empty()
        
        try:
            legend_image = load_image_safely(legend_file)
            if legend_image is None:
                st.error("Failed to load legend image.")
                st.stop()

            power_images = [load_image_safely(f) for f in (power_files or []) if load_image_safely(f)]
            lighting_images = [load_image_safely(f) for f in (lighting_files or []) if load_image_safely(f)]

            # Run Scans
            run_strict_takeoff_module_live(
                legend_image, power_images, "Power Package", "Power / Devices", 
                status_box, metrics_box, table_box, st.session_state.results_data
            )
            
            run_strict_takeoff_module_live(
                legend_image, lighting_images, "Lighting Package", "Lighting", 
                status_box, metrics_box, table_box, st.session_state.results_data
            )

            st.session_state.scan_complete = True
            st.rerun() # Rerun to show final export buttons cleanly

        except Exception as e:
            st.error(f"Critical Application Error: {str(e)}")
            st.code(traceback.format_exc())

# Display Results if scan is complete
elif st.session_state.scan_complete and st.session_state.results_data:
    df_summary = pd.DataFrame(list(st.session_state.results_data.values()))
    
    total_matches = sum(row['Count'] for row in st.session_state.results_data.values())
    st.success(f"✅ SCAN COMPLETE! Total: {total_matches} verified matches.")
    
    st.subheader("📋 Itemized Takeoff Schedule")
    st.dataframe(
        df_summary,
        column_config={
            "Legend Icon": st.column_config.ImageColumn("Legend Symbol", width="small"),
            "Count": st.column_config.NumberColumn("Verified Count", format="%d ⚡"),
        },
        use_container_width=True, hide_index=True,
    )

    # Export Buttons
    excel_output = io.BytesIO()
    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        df_summary.drop(columns=["Legend Icon"]).to_excel(writer, index=False, sheet_name="Takeoff Schedule")
    
    pdf_output = io.BytesIO()
    c = canvas.Canvas(pdf_output, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(54, height - 50, "DrBuild LLC - Verified Takeoff Report")
    c.setFont("Helvetica", 10)
    c.drawString(54, height - 68, "Strict computer vision scan schedule.")
    c.line(54, height - 78, width - 54, height - 78)
    
    y = height - 105
    c.setFont("Helvetica-Bold", 10)
    c.drawString(54, y, "Category"); c.drawString(180, y, "Description"); c.drawString(380, y, "Package"); c.drawString(490, y, "Count")
    y -= 20; c.setFont("Helvetica", 9)
    
    for _, row in df_summary.iterrows():
        if y < 50: c.showPage(); y = height - 50; c.setFont("Helvetica", 9)
        c.drawString(54, y, str(row["System Category"]))
        c.drawString(180, y, str(row["Model / Description"]))
        c.drawString(380, y, str(row["Scan Package"]))
        c.drawString(490, y, str(row["Count"]))
        y -= 18
    c.save()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Export Excel (.xlsx)", excel_output.getvalue(), "DrBuild_Takeoff.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col2:
        st.download_button(" Download PDF Report", pdf_output.getvalue(), "DrBuild_Report.pdf", "application/pdf")

else:
    st.info("Upload your legend sheet and optional drawing files in the sidebar to begin.")
