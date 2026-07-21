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
# 1. PAGE CONFIG & LAYOUT
# -----------------------------------------------------------------------------
st.set_page_config(page_title="DrBuild Electrical Takeoff Tool", page_icon="⚡", layout="wide")

st.title("⚡ Electrical Drawing Takeoff & Symbol Counter")
st.markdown("Strict CV Takeoff Engine: Multi-scale template matching with adaptive legend parsing.")

with st.sidebar:
    st.header("1. Upload Project Drawings")
    legend_file = st.file_uploader("Upload Legend Sheet (JPEG/PNG)", type=["png", "jpg", "jpeg"], key="legend_uploader")
    
    st.markdown("---")
    st.subheader("Drawing Packages")
    power_files = st.file_uploader("Upload Power Drawings", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="power_uploader")
    lighting_files = st.file_uploader("Upload Lighting Drawings", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="lighting_uploader")
    
    st.markdown("---")
    st.subheader("Scan Settings")
    match_threshold = st.slider("Match Sensitivity", 0.50, 0.90, 0.60, 0.05)
    max_scales = st.selectbox("Max Scale Factors", [5, 10, 15], index=1, help="Fewer = faster scan")
    debug_mode = st.checkbox("Show Debug Matches", value=False)
    
    st.markdown("---")
    preview_btn = st.button("🔍 Preview Legend Templates", type="secondary")
    process_btn = st.button(" Start Strict Takeoff Scan", type="primary")

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS (PURE OPENCV - NO TESSERACT)
# -----------------------------------------------------------------------------
def load_image_safely(uploaded_file):
    try:
        img = Image.open(uploaded_file).convert("RGB")
        if max(img.size) > 5000:
            img.thumbnail((5000, 5000), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None

def is_likely_text(symbol_crop):
    """
    Pure OpenCV heuristic to reject alphabets/numbers.
    Returns True if the blob looks like a letter (P, E, F, etc.)
    Returns False if it looks like a symbol (Circle, Triangle, Cross).
    """
    h, w = symbol_crop.shape
    if h == 0 or w == 0: return True
    
    # 1. Aspect Ratio Check
    # Letters are typically tall (I, l, 1) or standard portrait (P, E). 
    # Symbols are often square or landscape.
    aspect_ratio = w / h
    if aspect_ratio < 0.4: return True  # Too thin (likely I, l, 1)
    if aspect_ratio > 2.5: return True  # Too wide (likely -, _, or text line)

    # 2. Pixel Density (Solidity) Check
    # Letters have lots of white space inside their bounding box.
    # Symbols (like filled circles or thick crosses) are denser.
    total_pixels = h * w
    dark_pixels = np.sum(symbol_crop < 128)
    density = dark_pixels / total_pixels
    
    # Most letters have density < 0.25. Symbols are usually > 0.25
    if density < 0.20: return True 

    # 3. Horizontal Stroke Check (The "E", "F", "H" filter)
    # Letters often have long continuous horizontal lines.
    row_sums = np.sum(symbol_crop < 128, axis=1)
    max_row_fill_ratio = np.max(row_sums) / w
    
    # If a single row is >80% filled AND the object is smallish, it's likely a hyphen or part of a letter
    if max_row_fill_ratio > 0.85 and h < 30: 
        # Additional check: is it JUST a line? 
        if density < 0.3: return True

    # 4. Vertical Stroke Check (The "I", "l", "T" filter)
    col_sums = np.sum(symbol_crop < 128, axis=0)
    max_col_fill_ratio = np.max(col_sums) / h
    if max_col_fill_ratio > 0.85 and w < 20:
        if density < 0.3: return True

    return False

def extract_symbols_from_legend(legend_img, category_filter="All"):
    """Extracts symbols using Connected Components + Geometric Text Filtering."""
    try:
        legend_cv = np.array(legend_img)
        gray = cv2.cvtColor(legend_cv, cv2.COLOR_RGB2GRAY)
        h_leg, w_leg = gray.shape[:2]

        # Define Section Boundaries
        if category_filter == "Power / Devices":
            y_start, y_end = int(h_leg * 0.42), int(h_leg * 0.62)
        elif category_filter == "Lighting":
            y_start, y_end = int(h_leg * 0.62), int(h_leg * 0.78)
        else:
            y_start, y_end = 0, h_leg

        section = gray[y_start:y_end, :]
        _, binary = cv2.threshold(section, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Use morphology to connect nearby strokes (e.g., circle + line)
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        extracted_items = []
        item_counter = 1
        prefix = "Device" if category_filter == "Power / Devices" else "Fixture"
        filtered_count = 0

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Basic size filters
            if area < 50 or area > 15000: continue
            if w < 10 or h < 10: continue
            
            # Extract crop with padding
            pad = 5
            sy1, sy2 = max(0, y-pad), min(section.shape[0], y+h+pad)
            sx1, sx2 = max(0, x-pad), min(section.shape[1], x+w+pad)
            symbol_crop = section[sy1:sy2, sx1:sx2]
            
            # CRITICAL: Apply Geometric Text Filter
            if is_likely_text(symbol_crop):
                filtered_count += 1
                continue
            
            # Normalize size
            target_h = 60
            scale = target_h / max(1, symbol_crop.shape[0])
            new_w = max(20, int(symbol_crop.shape[1] * scale))
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
            
        print(f"Extraction: {len(extracted_items)} symbols kept, {filtered_count} text items rejected.")
        return extracted_items
    except Exception as e:
        st.error(f"Legend extraction failed: {str(e)}")
        traceback.print_exc()
        return []

def multi_scale_match(drawing_gray, template, threshold, scales=None):
    """Optimized multi-scale matching."""
    if scales is None:
        scales = [0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]
    
    all_matches = []
    if template is None or template.size == 0: return [], []

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
            suppression = max(15, int(25 * sc)) 
            if abs(pt[0] - fm_pt[0]) < suppression and abs(pt[1] - fm_pt[1]) < suppression:
                too_close = True; break
        if not too_close:
            filtered.append((pt, sc, score))
            
    return filtered, [m[0] for m in filtered[:10]]

# -----------------------------------------------------------------------------
# 3. PREVIEW TEMPLATES SECTION
# -----------------------------------------------------------------------------
if legend_file and preview_btn:
    with st.spinner("Extracting symbols from legend..."):
        legend_img = load_image_safely(legend_file)
        if legend_img:
            all_templates = extract_symbols_from_legend(legend_img, category_filter="All")
            
            if all_templates:
                st.success(f"✅ Extracted {len(all_templates)} TRUE SYMBOLS (text filtered out).")
                
                st.subheader("️ Verified Symbol Templates")
                cols = st.columns(6)
                for idx, item in enumerate(all_templates):
                    col = cols[idx % 6]
                    with col:
                        st.image(item["icon_bytes"], caption=item['name'], use_column_width=True)
                
                st.session_state.preview_templates = all_templates
                st.info("Templates saved. Upload drawings and click 'Start Scan'.")
            else:
                st.warning("No valid symbols detected. Check legend image quality.")
        else:
            st.error("Failed to load legend.")

# -----------------------------------------------------------------------------
# 4. MAIN SCAN WITH REAL-TIME PROGRESS
# -----------------------------------------------------------------------------
if 'results_data' not in st.session_state: st.session_state.results_data = {}
if 'scan_complete' not in st.session_state: st.session_state.scan_complete = False
if 'progress_pct' not in st.session_state: st.session_state.progress_pct = 0
if 'current_status' not in st.session_state: st.session_state.current_status = ""
if 'match_count' not in st.session_state: st.session_state.match_count = 0

# Persistent dashboard
dashboard = st.container()
status_box = dashboard.empty()
metrics_box = dashboard.empty()
progress_bar = dashboard.progress(0)
status_log = dashboard.empty()
table_box = dashboard.empty()

if process_btn:
    st.session_state.results_data = {}
    st.session_state.scan_complete = False
    st.session_state.progress_pct = 0
    st.session_state.match_count = 0
    
    if not legend_file:
        st.error("Please upload Legend Sheet first.")
    elif not power_files and not lighting_files:
        st.warning("Please upload at least one drawing file.")
    else:
        try:
            legend_image = load_image_safely(legend_file)
            if not legend_image: st.stop()

            power_images = [load_image_safely(f) for f in (power_files or []) if load_image_safely(f)]
            lighting_images = [load_image_safely(f) for f in (lighting_files or []) if load_image_safely(f)]
            
            total_drawings = len(power_images) + len(lighting_images)
            
            # Get templates
            if hasattr(st.session_state, 'preview_templates') and st.session_state.preview_templates:
                all_templates = st.session_state.preview_templates
            else:
                with st.spinner("Extracting templates from legend..."):
                    all_templates = extract_symbols_from_legend(legend_image, "All")
                if not all_templates:
                    st.warning("No templates found."); st.stop()

            # Build accumulator
            accumulator = {
                item["name"]: {
                    "System Category": item["category"],
                    "Legend Icon": item["icon_bytes"],
                    "Model / Description": item["name"],
                    "Scan Package": "Unknown",
                    "Count": 0,
                    "template": item["template"]
                }
                for item in all_templates
            }

            # Initial dashboard state
            metrics_box.markdown(
                f"""<div style="background:#f0f2f6;padding:15px;border-radius:8px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;text-align:center;">
                <div><div style="font-size:12px;color:#666;">DRAWINGS SCANNED</div>
                <div style="font-size:24px;font-weight:bold;color:#1f77b4;">0/{total_drawings}</div></div>
                <div><div style="font-size:12px;color:#666;">⚡ MATCHES FOUND</div>
                <div style="font-size:24px;font-weight:bold;color:#2ca02c;">0</div></div>
                <div><div style="font-size:12px;color:#666;">TEMPLATES</div>
                <div style="font-size:24px;font-weight:bold;color:#ff7f0e;">{len(all_templates)}</div></div>
                </div></div>""", unsafe_allow_html=True
            )
            status_box.info(f"Starting scan...\nTemplates: {len(all_templates)}\nThreshold: {match_threshold}")

            # Generate scale list based on user selection
            if max_scales == 5:
                scales = [0.5, 0.8, 1.0, 1.5, 2.0]
            elif max_scales == 10:
                scales = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]
            else:
                scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]

            drawing_idx = 0
            
            # Process Power Drawings
            for idx, d_img in enumerate(power_images):
                d_arr = np.array(d_img)
                if len(d_arr.shape) == 3: d_arr = cv2.cvtColor(d_arr, cv2.COLOR_RGB2GRAY)
                
                drawing_idx += 1
                pct = int((drawing_idx / total_drawings) * 100)
                
                st.session_state.progress_pct = pct
                st.session_state.current_status = f"[Power] Drawing {idx+1}/{len(power_images)}..."
                status_log.text(st.session_state.current_status)
                progress_bar.progress(pct / 100)
                
                drawing_matches = 0
                for name, row in accumulator.items():
                    matches, _ = multi_scale_match(d_arr, row["template"], match_threshold, scales)
                    row["Count"] += len(matches)
                    drawing_matches += len(matches)
                
                st.session_state.match_count += drawing_matches
                
                metrics_box.markdown(
                    f"""<div style="background:#f0f2f6;padding:15px;border-radius:8px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;text-align:center;">
                    <div><div style="font-size:12px;color:#666;">DRAWINGS SCANNED</div>
                    <div style="font-size:24px;font-weight:bold;color:#1f77b4;">{drawing_idx}/{total_drawings}</div></div>
                    <div><div style="font-size:12px;color:#666;">⚡ MATCHES FOUND</div>
                    <div style="font-size:24px;font-weight:bold;color:#2ca02c;">{st.session_state.match_count}</div></div>
                    <div><div style="font-size:12px;color:#666;">🔍 TEMPLATES</div>
                    <div style="font-size:24px;font-weight:bold;color:#ff7f0e;">{len(all_templates)}</div></div>
                    </div></div>""", unsafe_allow_html=True
                )
                
                if idx % 2 == 0 or idx == len(power_images)-1:
                    live_df = pd.DataFrame([{k:v for k,v in row.items() if k != "template"} for row in accumulator.values()])
                    table_box.dataframe(live_df, column_config={
                        "Legend Icon": st.column_config.ImageColumn("Symbol", width="small"),
                        "Count": st.column_config.NumberColumn("Count", format="%d ⚡"),
                    }, use_container_width=True, hide_index=True)

            # Process Lighting Drawings
            for idx, d_img in enumerate(lighting_images):
                d_arr = np.array(d_img)
                if len(d_arr.shape) == 3: d_arr = cv2.cvtColor(d_arr, cv2.COLOR_RGB2GRAY)
                
                drawing_idx += 1
                pct = int((drawing_idx / total_drawings) * 100)
                
                st.session_state.progress_pct = pct
                st.session_state.current_status = f"[Lighting] Drawing {idx+1}/{len(lighting_images)}..."
                status_log.text(st.session_state.current_status)
                progress_bar.progress(pct / 100)
                
                drawing_matches = 0
                for name, row in accumulator.items():
                    matches, _ = multi_scale_match(d_arr, row["template"], match_threshold, scales)
                    row["Count"] += len(matches)
                    drawing_matches += len(matches)
                
                st.session_state.match_count += drawing_matches
                
                metrics_box.markdown(
                    f"""<div style="background:#f0f2f6;padding:15px;border-radius:8px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;text-align:center;">
                    <div><div style="font-size:12px;color:#666;">DRAWINGS SCANNED</div>
                    <div style="font-size:24px;font-weight:bold;color:#1f77b4;">{drawing_idx}/{total_drawings}</div></div>
                    <div><div style="font-size:12px;color:#666;">⚡ MATCHES FOUND</div>
                    <div style="font-size:24px;font-weight:bold;color:#2ca02c;">{st.session_state.match_count}</div></div>
                    <div><div style="font-size:12px;color:#666;">🔍 TEMPLATES</div>
                    <div style="font-size:24px;font-weight:bold;color:#ff7f0e;">{len(all_templates)}</div></div>
                    </div></div>""", unsafe_allow_html=True
                )
                
                if idx % 2 == 0 or idx == len(lighting_images)-1:
                    live_df = pd.DataFrame([{k:v for k,v in row.items() if k != "template"} for row in accumulator.values()])
                    table_box.dataframe(live_df, column_config={
                        "Legend Icon": st.column_config.ImageColumn("Symbol", width="small"),
                        "Count": st.column_config.NumberColumn("Count", format="%d ⚡"),
                    }, use_container_width=True, hide_index=True)

            # Complete
            clean_results = {k: {kk:vv for kk,vv in v.items() if kk != "template"} for k,v in accumulator.items()}
            st.session_state.results_data = clean_results
            st.session_state.scan_complete = True
            st.rerun()

        except Exception as e:
            st.error(f"Critical Error: {str(e)}")
            st.code(traceback.format_exc())

elif st.session_state.scan_complete:
    df_summary = pd.DataFrame(list(st.session_state.results_data.values()))
    total_matches = sum(row['Count'] for row in st.session_state.results_data.values())
    
    status_box.success(f"✅ COMPLETE! {total_matches} matches found.")
    metrics_box.empty(); progress_bar.empty(); status_log.empty()

    st.subheader("📋 Takeoff Schedule")
    st.dataframe(df_summary, column_config={
        "Legend Icon": st.column_config.ImageColumn("Symbol", width="small"),
        "Count": st.column_config.NumberColumn("Count", format="%d ⚡"),
    }, use_container_width=True, hide_index=True)

    # Export
    excel_output = io.BytesIO()
    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        df_summary.drop(columns=["Legend Icon"]).to_excel(writer, index=False, sheet_name="Takeoff")
    
    pdf_output = io.BytesIO()
    c = canvas.Canvas(pdf_output, pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 16); c.drawString(54, h-50, "DrBuild LLC - Verified Takeoff Report")
    c.setFont("Helvetica", 10); c.drawString(54, h-68, "Computer vision scan schedule.")
    c.line(54, h-78, w-54, h-78)
    y = h-105; c.setFont("Helvetica-Bold", 10)
    c.drawString(54, y, "Category"); c.drawString(180, y, "Description"); c.drawString(380, y, "Package"); c.drawString(490, y, "Count")
    y -= 20; c.setFont("Helvetica", 9)
    for _, row in df_summary.iterrows():
        if y < 50: c.showPage(); y = h-50; c.setFont("Helvetica", 9)
        c.drawString(54, y, str(row["System Category"])); c.drawString(180, y, str(row["Model / Description"]))
        c.drawString(380, y, str(row["Scan Package"])); c.drawString(490, y, str(row["Count"]))
        y -= 18
    c.save()

    col1, col2 = st.columns(2)
    with col1: st.download_button("📥 Excel (.xlsx)", excel_output.getvalue(), "DrBuild_Takeoff.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col2: st.download_button("📄 PDF Report", pdf_output.getvalue(), "DrBuild_Report.pdf", "application/pdf")

else:
    status_box.empty(); metrics_box.empty(); progress_bar.empty(); status_log.empty(); table_box.empty()
    st.info("Upload legend and drawings to begin.")
    st.caption("💡 Click 'Preview Legend Templates' first to verify symbols before scanning.")
