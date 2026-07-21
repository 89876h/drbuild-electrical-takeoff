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
st.markdown("Feature-based symbol detection with shape analysis.")

with st.sidebar:
    st.header("1. Upload Project Drawings")
    legend_file = st.file_uploader("Upload Legend Sheet (JPEG/PNG)", type=["png", "jpg", "jpeg"], key="legend_uploader")
    
    st.markdown("---")
    st.subheader("Drawing Packages")
    power_files = st.file_uploader("Upload Power Drawings", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="power_uploader")
    lighting_files = st.file_uploader("Upload Lighting Drawings", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="lighting_uploader")
    
    st.markdown("---")
    st.subheader("Detection Settings")
    sensitivity = st.slider("Match Sensitivity", 0.3, 0.9, 0.55, 0.05, help="Lower = more matches but more false positives")
    debug_mode = st.checkbox("Show Debug Visualization", value=False)
    
    st.markdown("---")
    preview_btn = st.button("🔍 Preview Extracted Symbols", type="secondary")
    process_btn = st.button("▶️ Start Symbol Detection", type="primary")

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
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

def extract_electrical_symbols(legend_img, category_filter="All"):
    """
    Extracts electrical symbols using contour analysis.
    OPTIMIZED: Relaxed filters to preserve complex symbols (circles with lines, arrows).
    """
    try:
        legend_cv = np.array(legend_img)
        gray = cv2.cvtColor(legend_cv, cv2.COLOR_RGB2GRAY)
        h_leg, w_leg = gray.shape[:2]

        # Define section boundaries
        if category_filter == "Power / Devices":
            y_start, y_end = int(h_leg * 0.42), int(h_leg * 0.62)
        elif category_filter == "Lighting":
            y_start, y_end = int(h_leg * 0.62), int(h_leg * 0.78)
        else:
            y_start, y_end = 0, h_leg

        section = gray[y_start:y_end, :]
        
        # Threshold to binary (symbols are dark on white)
        _, binary = cv2.threshold(section, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Find external contours only
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        extracted_symbols = []
        item_counter = 1
        prefix = "Device" if category_filter == "Power / Devices" else "Fixture"
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # --- RELAXED SIZE FILTERS FOR SMALL LEGEND SYMBOLS ---
            if area < 20: continue      # Very low threshold to catch small symbols
            if w < 5 or h < 5: continue 
            if w > 150 or h > 150: continue # Max size to avoid text blocks
            
            # Extract symbol region with padding
            pad = 5
            sy1 = max(0, y - pad); sy2 = min(section.shape[0], y + h + pad)
            sx1 = max(0, x - pad); sx2 = min(section.shape[1], x + w + pad)
            symbol_crop = section[sy1:sy2, sx1:sx2]
            
            # --- MINIMAL TEXT REJECTION (Preserves Complex Symbols) ---
            aspect_ratio = w / h
            density = area / max(1, w * h)
            is_text = False
            
            # 1. Extreme aspect ratios (thin letters like I, l or wide like -)
            # Complex symbols are usually roughly square or rectangular
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                is_text = True
            
            # 2. Very low density + small area (likely punctuation or noise)
            # Complex symbols have wires/lines so they have decent density
            if density < 0.08 and area < 50:
                is_text = True
            
            # REMOVED: Aggressive hole detection. 
            # Complex symbols (circle with line) have holes but ARE NOT text.
            # We rely on shape matching later to distinguish them from letters.
            
            if is_text:
                continue
            
            # --- NORMALIZE SYMBOL FOR DISPLAY & MATCHING ---
            target_size = 60
            scale = target_size / max(w, h)
            new_w = max(15, int(w * scale))
            new_h = max(15, int(h * scale))
            
            symbol_resized = cv2.resize(symbol_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            pil_chip = Image.fromarray(symbol_resized)
            img_byte_arr = io.BytesIO()
            pil_chip.save(img_byte_arr, format="PNG")
            
            extracted_symbols.append({
                "category": category_filter,
                "name": f"{prefix} Type {item_counter}",
                "icon_bytes": img_byte_arr.getvalue(),
                "template": symbol_resized,
                "contour": contour, # Keep original contour for shape matching
                "bbox": (x, y, w, h),
            })
            item_counter += 1
        
        print(f"✅ Extracted {len(extracted_symbols)} symbols from '{category_filter}' section")
        return extracted_symbols
        
    except Exception as e:
        st.error(f"Symbol extraction failed: {str(e)}")
        traceback.print_exc()
        return []

def match_symbols_by_shape(drawing_gray, symbol_templates, sensitivity=0.55):
    """
    Matches symbols in drawing using contour shape comparison (Hu Moments).
    This distinguishes complex symbols from letters based on geometry.
    """
    results = {tmpl["name"]: 0 for tmpl in symbol_templates}
    debug_matches = []
    
    # Threshold drawing
    _, drawing_binary = cv2.threshold(drawing_gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in drawing
    drawing_contours, _ = cv2.findContours(drawing_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for d_contour in drawing_contours:
        dx, dy, dw, dh = cv2.boundingRect(d_contour)
        d_area = cv2.contourArea(d_contour)
        
        # Filter drawing contours by reasonable symbol size
        if d_area < 20 or dw < 5 or dh < 5: continue
        if dw > 300 or dh > 300: continue
        
        d_aspect = dw / max(1, dh)
        
        best_match_name = None
        best_score = 0 # Higher is better (similarity)
        
        for tmpl in symbol_templates:
            t_contour = tmpl["contour"]
            tx, ty, tw, th = tmpl["bbox"]
            t_aspect = tw / max(1, th)
            
            # Quick aspect ratio filter (must be within 60% similarity)
            if abs(d_aspect - t_aspect) / max(d_aspect, t_aspect, 0.1) > 0.6:
                continue
            
            # Compare shapes using Hu Moments (scale/rotation invariant)
            # CONTOURS_MATCH_I1 is best for complex shapes with holes
            score = cv2.matchShapes(d_contour, t_contour, cv2.CONTOURS_MATCH_I1, 0)
            similarity = 1.0 / (1.0 + score)
            
            if similarity > best_score:
                best_score = similarity
                best_match_name = tmpl["name"]
        
        # If we found a good match above sensitivity threshold
        if best_score > sensitivity and best_match_name:
            results[best_match_name] += 1
            
            if debug_mode:
                debug_matches.append((dx, dy, dw, dh, best_match_name, best_score))
    
    return results, debug_matches

# -----------------------------------------------------------------------------
# 3. PREVIEW SECTION
# -----------------------------------------------------------------------------
if legend_file and preview_btn:
    with st.spinner("Extracting symbols from legend..."):
        legend_img = load_image_safely(legend_file)
        if legend_img:
            symbols = extract_electrical_symbols(legend_img, "All")
            
            if symbols:
                st.success(f"✅ Found {len(symbols)} electrical symbols (text filtered out)")
                
                st.subheader("️ Extracted Symbols Preview")
                cols = st.columns(6)
                for idx, sym in enumerate(symbols):
                    col = cols[idx % 6]
                    with col:
                        st.image(sym["icon_bytes"], caption=sym['name'], use_column_width=True)
                
                st.session_state.extracted_symbols = symbols
                st.info("Symbols saved! Upload drawings and click 'Start Symbol Detection'.")
            else:
                st.warning("No symbols detected. Try adjusting image quality or thresholds.")
        else:
            st.error("Failed to load legend image.")

# -----------------------------------------------------------------------------
# 4. MAIN PROCESSING WITH LIVE PROGRESS
# -----------------------------------------------------------------------------
if 'results_data' not in st.session_state: st.session_state.results_data = {}
if 'scan_complete' not in st.session_state: st.session_state.scan_complete = False

dashboard = st.container()
status_box = dashboard.empty()
metrics_box = dashboard.empty()
progress_bar = dashboard.progress(0)
status_log = dashboard.empty()
table_box = dashboard.empty()

if process_btn:
    st.session_state.results_data = {}
    st.session_state.scan_complete = False
    
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
            
            # Get symbols (reuse preview if available)
            if hasattr(st.session_state, 'extracted_symbols') and st.session_state.extracted_symbols:
                symbols = st.session_state.extracted_symbols
            else:
                with st.spinner("Extracting symbols from legend..."):
                    symbols = extract_electrical_symbols(legend_image, "All")
                if not symbols:
                    st.warning("No symbols found in legend."); st.stop()

            # Initialize accumulator
            accumulator = {
                sym["name"]: {
                    "System Category": sym["category"],
                    "Legend Icon": sym["icon_bytes"],
                    "Model / Description": sym["name"],
                    "Scan Package": "Unknown",
                    "Count": 0,
                }
                for sym in symbols
            }

            # Initial dashboard state
            metrics_box.markdown(
                f"""<div style="background:#f0f2f6;padding:15px;border-radius:8px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;text-align:center;">
                <div><div style="font-size:12px;color:#666;">DRAWINGS SCANNED</div>
                <div style="font-size:24px;font-weight:bold;color:#1f77b4;">0/{total_drawings}</div></div>
                <div><div style="font-size:12px;color:#666;">⚡ SYMBOLS FOUND</div>
                <div style="font-size:24px;font-weight:bold;color:#2ca02c;">0</div></div>
                <div><div style="font-size:12px;color:#666;">TEMPLATES</div>
                <div style="font-size:24px;font-weight:bold;color:#ff7f0e;">{len(symbols)}</div></div>
                </div></div>""", unsafe_allow_html=True
            )
            status_box.info(f"Starting detection...\nTemplates: {len(symbols)}\nSensitivity: {sensitivity}")

            drawing_idx = 0
            total_matches = 0
            
            # Process Power Drawings
            for idx, d_img in enumerate(power_images):
                d_arr = np.array(d_img)
                if len(d_arr.shape) == 3: d_arr = cv2.cvtColor(d_arr, cv2.COLOR_RGB2GRAY)
                
                drawing_idx += 1
                pct = int((drawing_idx / total_drawings) * 100)
                
                status_log.text(f"[Power] Processing drawing {idx+1}/{len(power_images)}...")
                progress_bar.progress(pct / 100)
                
                matches, _ = match_symbols_by_shape(d_arr, symbols, sensitivity)
                
                drawing_count = 0
                for name, count in matches.items():
                    if name in accumulator:
                        accumulator[name]["Count"] += count
                        drawing_count += count
                
                total_matches += drawing_count
                
                metrics_box.markdown(
                    f"""<div style="background:#f0f2f6;padding:15px;border-radius:8px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;text-align:center;">
                    <div><div style="font-size:12px;color:#666;">DRAWINGS SCANNED</div>
                    <div style="font-size:24px;font-weight:bold;color:#1f77b4;">{drawing_idx}/{total_drawings}</div></div>
                    <div><div style="font-size:12px;color:#666;">⚡ SYMBOLS FOUND</div>
                    <div style="font-size:24px;font-weight:bold;color:#2ca02c;">{total_matches}</div></div>
                    <div><div style="font-size:12px;color:#666;">TEMPLATES</div>
                    <div style="font-size:24px;font-weight:bold;color:#ff7f0e;">{len(symbols)}</div></div>
                    </div></div>""", unsafe_allow_html=True
                )
                
                if idx % 2 == 0 or idx == len(power_images)-1:
                    live_df = pd.DataFrame(list(accumulator.values()))
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
                
                status_log.text(f"[Lighting] Processing drawing {idx+1}/{len(lighting_images)}...")
                progress_bar.progress(pct / 100)
                
                matches, _ = match_symbols_by_shape(d_arr, symbols, sensitivity)
                
                drawing_count = 0
                for name, count in matches.items():
                    if name in accumulator:
                        accumulator[name]["Count"] += count
                        drawing_count += count
                
                total_matches += drawing_count
                
                metrics_box.markdown(
                    f"""<div style="background:#f0f2f6;padding:15px;border-radius:8px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;text-align:center;">
                    <div><div style="font-size:12px;color:#666;">DRAWINGS SCANNED</div>
                    <div style="font-size:24px;font-weight:bold;color:#1f77b4;">{drawing_idx}/{total_drawings}</div></div>
                    <div><div style="font-size:12px;color:#666;"> SYMBOLS FOUND</div>
                    <div style="font-size:24px;font-weight:bold;color:#2ca02c;">{total_matches}</div></div>
                    <div><div style="font-size:12px;color:#666;">TEMPLATES</div>
                    <div style="font-size:24px;font-weight:bold;color:#ff7f0e;">{len(symbols)}</div></div>
                    </div></div>""", unsafe_allow_html=True
                )
                
                if idx % 2 == 0 or idx == len(lighting_images)-1:
                    live_df = pd.DataFrame(list(accumulator.values()))
                    table_box.dataframe(live_df, column_config={
                        "Legend Icon": st.column_config.ImageColumn("Symbol", width="small"),
                        "Count": st.column_config.NumberColumn("Count", format="%d ⚡"),
                    }, use_container_width=True, hide_index=True)

            # Complete
            st.session_state.results_data = accumulator
            st.session_state.scan_complete = True
            st.rerun()

        except Exception as e:
            st.error(f"Critical Error: {str(e)}")
            st.code(traceback.format_exc())

elif st.session_state.scan_complete:
    df_summary = pd.DataFrame(list(st.session_state.results_data.values()))
    total_matches = sum(row['Count'] for row in st.session_state.results_data.values())
    
    status_box.success(f"✅ COMPLETE! {total_matches} symbols detected across all drawings.")
    metrics_box.empty(); progress_bar.empty(); status_log.empty()

    st.subheader("📋 Symbol Takeoff Schedule")
    st.dataframe(df_summary, column_config={
        "Legend Icon": st.column_config.ImageColumn("Symbol", width="small"),
        "Count": st.column_config.NumberColumn("Count", format="%d "),
    }, use_container_width=True, hide_index=True)

    # Export buttons
    excel_output = io.BytesIO()
    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        df_summary.drop(columns=["Legend Icon"]).to_excel(writer, index=False, sheet_name="Takeoff")
    
    pdf_output = io.BytesIO()
    c = canvas.Canvas(pdf_output, pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 16); c.drawString(54, h-50, "DrBuild LLC - Symbol Takeoff Report")
    c.setFont("Helvetica", 10); c.drawString(54, h-68, "Computer vision detection results.")
    c.line(54, h-78, w-54, h-78)
    y = h-105; c.setFont("Helvetica-Bold", 10)
    c.drawString(54, y, "Category"); c.drawString(180, y, "Description"); c.drawString(380, y, "Package"); c.drawString(490, y, "Count")
    y -= 20; c.setFont("Helvetica", 9)
    for _, row in df_summary.iterrows():
        if y < 50: c.showPage(); y = h-50; c.setFont("Helvetica", 9)
        c.drawString(54, y, str(row["System Category"]))
        c.drawString(180, y, str(row["Model / Description"]))
        c.drawString(380, y, str(row["Scan Package"]))
        c.drawString(490, y, str(row["Count"]))
        y -= 18
    c.save()

    col1, col2 = st.columns(2)
    with col1: st.download_button(" Excel (.xlsx)", excel_output.getvalue(), "DrBuild_Takeoff.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col2: st.download_button("📄 PDF Report", pdf_output.getvalue(), "DrBuild_Report.pdf", "application/pdf")

else:
    status_box.empty(); metrics_box.empty(); progress_bar.empty(); status_log.empty(); table_box.empty()
    st.info("Upload legend and drawings to begin.")
    st.caption("💡 Click 'Preview Extracted Symbols' first to verify detection before scanning.")
