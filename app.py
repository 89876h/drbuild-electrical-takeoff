import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import traceback
import fitz  # PyMuPDF
import pytesseract

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & LAYOUT
# -----------------------------------------------------------------------------
st.set_page_config(page_title="DrBuild Electrical Takeoff Tool", page_icon="⚡", layout="wide")

st.title("⚡ Electrical Drawing Takeoff & Symbol Counter")
st.markdown("Spatial-aware extraction: Symbols only, zero alphabets.")

with st.sidebar:
    st.header("1. Upload Project Drawings")
    legend_file = st.file_uploader(
        "Upload Legend Sheet (JPEG/PNG/PDF)", 
        type=["png", "jpg", "jpeg", "pdf"], 
        key="legend_uploader"
    )
    
    st.markdown("---")
    st.subheader("Drawing Packages")
    power_files = st.file_uploader(
        "Upload Power Drawings (JPEG/PNG/PDF)", 
        type=["png", "jpg", "jpeg", "pdf"], 
        accept_multiple_files=True, 
        key="power_uploader"
    )
    lighting_files = st.file_uploader(
        "Upload Lighting Drawings (JPEG/PNG/PDF)", 
        type=["png", "jpg", "jpeg", "pdf"], 
        accept_multiple_files=True, 
        key="lighting_uploader"
    )
    
    st.markdown("---")
    st.subheader("Detection Settings")
    match_threshold = st.slider("Match Threshold", 0.50, 0.95, 0.65, 0.05)
    debug_mode = st.checkbox("Show Debug Matches", value=False)
    
    st.markdown("---")
    preview_btn = st.button("🔍 Preview Receptacles Only", type="secondary")
    process_btn = st.button("▶️ Run Takeoff Scan", type="primary")

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def load_image_safely(uploaded_file, max_dim=4000):
    """Safely loads images and PDFs with dynamic DPI scaling."""
    try:
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.pdf'):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            if len(doc) == 0: return None
            
            page = doc[0]
            rect = page.rect
            scale = min(max_dim / rect.width, max_dim / rect.height)
            dpi = int(scale * 72)
            
            pix = page.get_pixmap(dpi=max(72, dpi)) 
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
        else:
            img = Image.open(uploaded_file).convert("RGB")
            
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
        return img
        
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {str(e)}")
        return None

def extract_receptacles_spatial(legend_img, category_filter="All"):
    """
    SPATIAL EXTRACTION: Finds text via OCR, then extracts ONLY the region 
    immediately to the LEFT of each text label. Guarantees zero alphabets.
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
        
        # Use OCR to find TEXT bounding boxes
        # This is the KEY to eliminating alphabets: we find text FIRST
        data = pytesseract.image_to_data(section, output_type=pytesseract.Output.DICT)
        
        text_boxes = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:  # Confidence threshold
                x = int(data['left'][i])
                y = int(data['top'][i])
                w = int(data['width'][i])
                h = int(data['height'][i])
                text = data['text'][i].strip()
                
                # Skip empty or single-character noise
                if len(text) < 1: continue
                
                text_boxes.append((x, y, w, h, text))

        # Sort by Y position to maintain logical order
        text_boxes.sort(key=lambda b: (b[1], b[0]))
        
        symbols = []
        item_counter = 1
        prefix = "Device" if category_filter == "Power / Devices" else "Fixture"
        used_regions = []  # Prevent overlapping extractions

        for tx, ty, tw, th, text_content in text_boxes:
            # Define the "Symbol Zone": Left of the text, same height
            # Assume symbol is roughly square or slightly wider than tall
            sym_w = max(th, int(tw * 0.8)) 
            sym_h = th
            
            sx1 = max(0, tx - sym_w - 5)  # 5px gap between symbol and text
            sx2 = max(0, tx - 5)
            sy1 = max(0, ty - 2)
            sy2 = min(section.shape[0], ty + sym_h + 2)
            
            # Check if this region overlaps with an already extracted symbol
            overlap = False
            for ux1, uy1, ux2, uy2 in used_regions:
                if not (sx2 < ux1 or sx1 > ux2 or sy2 < uy1 or sy1 > uy2):
                    overlap = True
                    break
            
            if overlap: continue
            
            symbol_crop = section[sy1:sy2, sx1:sx2]
            
            # CRITICAL: Verify the crop actually contains dark pixels (a real symbol)
            # If it's just whitespace, this text had no symbol next to it
            if np.mean(symbol_crop) > 240: 
                continue 
            
            # Additional check: ensure it's not just another piece of text
            # by checking density. Symbols are usually denser than thin letters.
            _, bin_crop = cv2.threshold(symbol_crop, 128, 255, cv2.THRESH_BINARY_INV)
            density = np.sum(bin_crop > 0) / max(1, symbol_crop.shape[0] * symbol_crop.shape[1])
            if density < 0.10: continue  # Too sparse, likely stray mark

            # Normalize
            target_h = 50
            scale = target_h / max(1, symbol_crop.shape[0])
            new_w = max(15, int(symbol_crop.shape[1] * scale))
            symbol_resized = cv2.resize(symbol_crop, (new_w, target_h), interpolation=cv2.INTER_AREA)
            
            pil_chip = Image.fromarray(symbol_resized)
            img_byte_arr = io.BytesIO()
            pil_chip.save(img_byte_arr, format="PNG")
            
            symbols.append({
                "category": category_filter,
                "name": f"{prefix} Type {item_counter}",
                "icon_bytes": img_byte_arr.getvalue(),
                "template": symbol_resized,
                "orig_size": (symbol_crop.shape[1], symbol_crop.shape[0]),
                "label_text": text_content  # Store associated text for reference
            })
            used_regions.append((sx1, sy1, sx2, sy2))
            item_counter += 1
        
        print(f"✅ Spatial extraction found {len(symbols)} receptacle symbols")
        return symbols
        
    except Exception as e:
        st.error(f"Spatial extraction failed: {str(e)}")
        traceback.print_exc()
        return []

def multi_scale_rotate_match(drawing_gray, template, threshold, 
                             scales=None, rotations=None):
    """Multi-scale, multi-rotation template matching with NMS."""
    if scales is None: scales = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    if rotations is None: rotations = [0, 90, 180, 270]
    
    all_matches = []
    
    for scale in scales:
        scaled_template = cv2.resize(
            template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        
        for angle in rotations:
            if angle == 0:
                rotated = scaled_template
            else:
                h, w = scaled_template.shape[:2]
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(scaled_template, M, (w, h), borderValue=0)
            
            if (rotated.shape[0] > drawing_gray.shape[0] or 
                rotated.shape[1] > drawing_gray.shape[1]):
                continue
            
            res = cv2.matchTemplate(drawing_gray, rotated, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            points = list(zip(*loc[::-1]))
            
            for pt in points:
                score = float(res[pt[1], pt[0]])
                all_matches.append((pt, score, scale, angle))
    
    # Non-maximum suppression
    all_matches.sort(key=lambda x: -x[1])
    filtered = []
    for pt, score, sc, ang in all_matches:
        too_close = False
        for fm_pt, fm_score, fm_sc, fm_ang in filtered:
            suppression = max(10, int(15 * sc))
            if (abs(pt[0] - fm_pt[0]) < suppression and 
                abs(pt[1] - fm_pt[1]) < suppression):
                too_close = True
                break
        if not too_close:
            filtered.append((pt, score, sc, ang))
    
    return filtered

# -----------------------------------------------------------------------------
# 3. PREVIEW SECTION
# -----------------------------------------------------------------------------
if legend_file and preview_btn:
    with st.spinner("Extracting receptacles using spatial analysis..."):
        legend_img = load_image_safely(legend_file)
        if legend_img:
            symbols = extract_receptacles_spatial(legend_img, "All")
            
            if symbols:
                st.success(f"✅ Found {len(symbols)} RECEPTACLE SYMBOLS (zero alphabets)")
                
                st.subheader("️ Verified Receptacle Templates")
                cols = st.columns(6)
                for idx, sym in enumerate(symbols):
                    col = cols[idx % 6]
                    with col:
                        st.image(sym["icon_bytes"], 
                                caption=f"{sym['name']}\n(Label: {sym.get('label_text','?')})", 
                                use_column_width=True)
                
                st.session_state.extracted_symbols = symbols
                st.info("Receptacles saved! Upload drawings and click 'Run Takeoff Scan'.")
            else:
                st.warning("No receptacles found. Ensure legend has symbols LEFT of text labels.")
        else:
            st.error("Failed to load legend.")

# -----------------------------------------------------------------------------
# 4. MAIN PROCESSING
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
            
            # Get symbols
            if hasattr(st.session_state, 'extracted_symbols') and st.session_state.extracted_symbols:
                symbols = st.session_state.extracted_symbols
            else:
                with st.spinner("Processing legend..."):
                    symbols = extract_receptacles_spatial(legend_image, "All")
                if not symbols:
                    st.warning("No receptacles found."); st.stop()

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

            # Initial dashboard
            metrics_box.markdown(
                f"""<div style="background:#f0f2f6;padding:15px;border-radius:8px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;text-align:center;">
                <div><div style="font-size:12px;color:#666;">DRAWINGS SCANNED</div>
                <div style="font-size:24px;font-weight:bold;color:#1f77b4;">0/{total_drawings}</div></div>
                <div><div style="font-size:12px;color:#666;">⚡ RECEPTACLES FOUND</div>
                <div style="font-size:24px;font-weight:bold;color:#2ca02c;">0</div></div>
                <div><div style="font-size:12px;color:#666;">TEMPLATES</div>
                <div style="font-size:24px;font-weight:bold;color:#ff7f0e;">{len(symbols)}</div></div>
                </div></div>""", unsafe_allow_html=True
            )
            status_box.info(f"Starting scan...\nTemplates: {len(symbols)}\nThreshold: {match_threshold}")

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
                
                drawing_count = 0
                for sym in symbols:
                    matches = multi_scale_rotate_match(d_arr, sym["template"], match_threshold)
                    count = len(matches)
                    accumulator[sym["name"]]["Count"] += count
                    drawing_count += count
                    
                    if debug_mode and matches:
                        vis = cv2.cvtColor(d_arr, cv2.COLOR_GRAY2RGB)
                        for pt, score, sc, ang in matches[:5]:
                            th, tw = sym["template"].shape
                            scaled_tw = int(tw * sc)
                            scaled_th = int(th * sc)
                            cv2.rectangle(vis, pt, (pt[0]+scaled_tw, pt[1]+scaled_th), (0,255,0), 2)
                        st.image(vis, caption=f"{sym['name']} - {count} matches", use_column_width=True)
                
                total_matches += drawing_count
                
                metrics_box.markdown(
                    f"""<div style="background:#f0f2f6;padding:15px;border-radius:8px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;text-align:center;">
                    <div><div style="font-size:12px;color:#666;">DRAWINGS SCANNED</div>
                    <div style="font-size:24px;font-weight:bold;color:#1f77b4;">{drawing_idx}/{total_drawings}</div></div>
                    <div><div style="font-size:12px;color:#666;">⚡ RECEPTACLES FOUND</div>
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
                
                drawing_count = 0
                for sym in symbols:
                    matches = multi_scale_rotate_match(d_arr, sym["template"], match_threshold)
                    count = len(matches)
                    accumulator[sym["name"]]["Count"] += count
                    drawing_count += count
                
                total_matches += drawing_count
                
                metrics_box.markdown(
                    f"""<div style="background:#f0f2f6;padding:15px;border-radius:8px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;text-align:center;">
                    <div><div style="font-size:12px;color:#666;">DRAWINGS SCANNED</div>
                    <div style="font-size:24px;font-weight:bold;color:#1f77b4;">{drawing_idx}/{total_drawings}</div></div>
                    <div><div style="font-size:12px;color:#666;"> RECEPTACLES FOUND</div>
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
    
    status_box.success(f"✅ COMPLETE! {total_matches} receptacles detected.")
    metrics_box.empty(); progress_bar.empty(); status_log.empty()

    st.subheader("📋 Receptacle Takeoff Schedule")
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
    c.setFont("Helvetica-Bold", 16); c.drawString(54, h-50, "DrBuild LLC - Receptacle Takeoff Report")
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
    with col1: st.download_button("📥 Excel (.xlsx)", excel_output.getvalue(), "DrBuild_Takeoff.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col2: st.download_button("📄 PDF Report", pdf_output.getvalue(), "DrBuild_Report.pdf", "application/pdf")

else:
    status_box.empty(); metrics_box.empty(); progress_bar.empty(); status_log.empty(); table_box.empty()
    st.info("Upload legend and drawings to begin.")
    st.caption("💡 Click 'Preview Receptacles Only' to verify spatial extraction before scanning.")
