import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import traceback
import fitz  # PyMuPDF for PDF support

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & LAYOUT
# -----------------------------------------------------------------------------
st.set_page_config(page_title="DrBuild Electrical Takeoff Tool", page_icon="⚡", layout="wide")

st.title("⚡ Electrical Drawing Takeoff & Symbol Counter")
st.markdown("Multi-scale template matching with text removal and rotation support.")

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
    preview_btn = st.button("🔍 Preview Symbols", type="secondary")
    process_btn = st.button("▶️ Run Takeoff Scan", type="primary")

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def load_image_safely(uploaded_file):
    """Safely opens uploaded files, converting PDFs to images automatically."""
    try:
        file_name = uploaded_file.name.lower()
        
        # Handle PDF files using PyMuPDF
        if file_name.endswith('.pdf'):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            if len(doc) == 0:
                st.error(f"PDF {uploaded_file.name} is empty.")
                return None
            page = doc[0]  # Process first page only
            pix = page.get_pixmap(dpi=300)  # 300 DPI for crisp symbol detection
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
        else:
            # Handle standard images
            img = Image.open(uploaded_file).convert("RGB")
        
        # Downscale if too large to prevent memory issues
        if max(img.size) > 5000:
            img.thumbnail((5000, 5000), Image.Resampling.LANCZOS)
            
        return img
        
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {str(e)}")
        return None

def remove_text_from_legend(gray_img):
    """
    Removes text while preserving electrical symbols.
    Text is typically thinner and more linear than symbols.
    """
    # Adaptive threshold handles faded scans better than fixed threshold
    binary = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Detect thin horizontal/vertical structures (text lines)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    text_mask = cv2.bitwise_or(h_lines, v_lines)
    
    # Dilate mask slightly to catch serifs and adjacent pixels
    text_mask = cv2.dilate(text_mask, np.ones((2,2), np.uint8), iterations=1)
    
    # Remove text from binary image
    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(text_mask))
    
    # Clean up remaining noise
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def extract_symbols_from_cleaned_legend(cleaned_binary, category_filter="All"):
    """Extracts symbols from text-removed legend using connected components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned_binary, connectivity=8
    )
    
    symbols = []
    item_counter = 1
    prefix = "Device" if category_filter == "Power / Devices" else "Fixture"
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # Size filters (text already removed, so these catch symbols)
        if area < 30 or w < 8 or h < 8: 
            continue
        if w > 150 or h > 150: 
            continue  # Skip large artifacts
            
        # Extract with padding
        pad = 5
        sy1 = max(0, y-pad); sy2 = min(cleaned_binary.shape[0], y+h+pad)
        sx1 = max(0, x-pad); sx2 = min(cleaned_binary.shape[1], x+w+pad)
        symbol_crop = cleaned_binary[sy1:sy2, sx1:sx2]
        
        # Normalize to standard size for template matching
        target_h = 50
        scale = target_h / max(1, h)
        new_w = max(15, int(w * scale))
        symbol_resized = cv2.resize(
            symbol_crop, (new_w, target_h), interpolation=cv2.INTER_AREA
        )
        
        pil_chip = Image.fromarray(symbol_resized)
        img_byte_arr = io.BytesIO()
        pil_chip.save(img_byte_arr, format="PNG")
        
        symbols.append({
            "category": category_filter,
            "name": f"{prefix} Type {item_counter}",
            "icon_bytes": img_byte_arr.getvalue(),
            "template": symbol_resized,
            "orig_size": (w, h)
        })
        item_counter += 1
    
    return symbols

def multi_scale_rotate_match(drawing_gray, template, threshold, 
                             scales=None, rotations=None):
    """
    Multi-scale, multi-rotation template matching with non-maximum suppression.
    """
    if scales is None: 
        scales = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    if rotations is None: 
        rotations = [0, 90, 180, 270]
    
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
                rotated = cv2.warpAffine(
                    scaled_template, M, (w, h), borderValue=0
                )
            
            # Skip if template larger than drawing
            if (rotated.shape[0] > drawing_gray.shape[0] or 
                rotated.shape[1] > drawing_gray.shape[1]):
                continue
            
            res = cv2.matchTemplate(
                drawing_gray, rotated, cv2.TM_CCOEFF_NORMED
            )
            loc = np.where(res >= threshold)
            points = list(zip(*loc[::-1]))
            
            for pt in points:
                score = float(res[pt[1], pt[0]])
                all_matches.append((pt, score, scale, angle))
    
    # Non-maximum suppression: keep best match in local area
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
    with st.spinner("Processing legend (removing text + extracting symbols)..."):
        legend_img = load_image_safely(legend_file)
        if legend_img:
            gray = cv2.cvtColor(np.array(legend_img), cv2.COLOR_RGB2GRAY)
            cleaned = remove_text_from_legend(gray)
            
            # Show before/after
            col1, col2 = st.columns(2)
            with col1: 
                st.image(gray, caption="Original Legend", use_column_width=True)
            with col2: 
                st.image(cleaned, caption="After Text Removal", use_column_width=True)
            
            symbols = extract_symbols_from_cleaned_legend(cleaned, "All")
            
            if symbols:
                st.success(f"✅ Extracted {len(symbols)} symbols (text removed)")
                
                st.subheader("🖼️ Extracted Symbols")
                cols = st.columns(6)
                for idx, sym in enumerate(symbols):
                    col = cols[idx % 6]
                    with col:
                        st.image(
                            sym["icon_bytes"], 
                            caption=sym['name'], 
                            use_column_width=True
                        )
                
                st.session_state.extracted_symbols = symbols
                st.session_state.cleaned_legend = cleaned
                st.info("Symbols saved! Upload drawings and click 'Run Takeoff Scan'.")
            else:
                st.warning("No symbols found after text removal. Check legend quality.")
        else:
            st.error("Failed to load legend.")

# -----------------------------------------------------------------------------
# 4. MAIN PROCESSING WITH LIVE PROGRESS
# -----------------------------------------------------------------------------
if 'results_data' not in st.session_state: 
    st.session_state.results_data = {}
if 'scan_complete' not in st.session_state: 
    st.session_state.scan_complete = False

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
            if not legend_image: 
                st.stop()

            power_images = [
                load_image_safely(f) 
                for f in (power_files or []) 
                if load_image_safely(f)
            ]
            lighting_images = [
                load_image_safely(f) 
                for f in (lighting_files or []) 
                if load_image_safely(f)
            ]
            
            total_drawings = len(power_images) + len(lighting_images)
            
            # Get symbols (reuse preview if available)
            if (hasattr(st.session_state, 'extracted_symbols') and 
                st.session_state.extracted_symbols):
                symbols = st.session_state.extracted_symbols
            else:
                with st.spinner("Processing legend..."):
                    gray = cv2.cvtColor(
                        np.array(legend_image), cv2.COLOR_RGB2GRAY
                    )
                    cleaned = remove_text_from_legend(gray)
                    symbols = extract_symbols_from_cleaned_legend(cleaned, "All")
                if not symbols:
                    st.warning("No symbols found."); 
                    st.stop()

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
                </div></div>""", 
                unsafe_allow_html=True
            )
            status_box.info(
                f"Starting scan...\nTemplates: {len(symbols)}\n"
                f"Threshold: {match_threshold}"
            )

            drawing_idx = 0
            total_matches = 0
            
            # Process Power Drawings
            for idx, d_img in enumerate(power_images):
                d_arr = np.array(d_img)
                if len(d_arr.shape) == 3: 
                    d_arr = cv2.cvtColor(d_arr, cv2.COLOR_RGB2GRAY)
                
                drawing_idx += 1
                pct = int((drawing_idx / total_drawings) * 100)
                
                status_log.text(
                    f"[Power] Processing drawing {idx+1}/{len(power_images)}..."
                )
                progress_bar.progress(pct / 100)
                
                drawing_count = 0
                for sym in symbols:
                    matches = multi_scale_rotate_match(
                        d_arr, sym["template"], match_threshold
                    )
                    count = len(matches)
                    accumulator[sym["name"]]["Count"] += count
                    drawing_count += count
                    
                    # Debug visualization
                    if debug_mode and matches:
                        vis = cv2.cvtColor(d_arr, cv2.COLOR_GRAY2RGB)
                        for pt, score, sc, ang in matches[:5]:
                            th, tw = sym["template"].shape
                            scaled_tw = int(tw * sc)
                            scaled_th = int(th * sc)
                            cv2.rectangle(
                                vis, pt, 
                                (pt[0]+scaled_tw, pt[1]+scaled_th), 
                                (0,255,0), 2
                            )
                        st.image(
                            vis, 
                            caption=f"{sym['name']} - {count} matches", 
                            use_column_width=True
                        )
                
                total_matches += drawing_count
                
                # Update metrics
                metrics_box.markdown(
                    f"""<div style="background:#f0f2f6;padding:15px;border-radius:8px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;text-align:center;">
                    <div><div style="font-size:12px;color:#666;">DRAWINGS SCANNED</div>
                    <div style="font-size:24px;font-weight:bold;color:#1f77b4;">{drawing_idx}/{total_drawings}</div></div>
                    <div><div style="font-size:12px;color:#666;">⚡ SYMBOLS FOUND</div>
                    <div style="font-size:24px;font-weight:bold;color:#2ca02c;">{total_matches}</div></div>
                    <div><div style="font-size:12px;color:#666;">TEMPLATES</div>
                    <div style="font-size:24px;font-weight:bold;color:#ff7f0e;">{len(symbols)}</div></div>
                    </div></div>""", 
                    unsafe_allow_html=True
                )
                
                # Update table periodically
                if idx % 2 == 0 or idx == len(power_images)-1:
                    live_df = pd.DataFrame(list(accumulator.values()))
                    table_box.dataframe(
                        live_df, 
                        column_config={
                            "Legend Icon": st.column_config.ImageColumn(
                                "Symbol", width="small"
                            ),
                            "Count": st.column_config.NumberColumn(
                                "Count", format="%d ⚡"
                            ),
                        }, 
                        use_container_width=True, 
                        hide_index=True
                    )

            # Process Lighting Drawings
            for idx, d_img in enumerate(lighting_images):
                d_arr = np.array(d_img)
                if len(d_arr.shape) == 3: 
                    d_arr = cv2.cvtColor(d_arr, cv2.COLOR_RGB2GRAY)
                
                drawing_idx += 1
                pct = int((drawing_idx / total_drawings) * 100)
                
                status_log.text(
                    f"[Lighting] Processing drawing {idx+1}/{len(lighting_images)}..."
                )
                progress_bar.progress(pct / 100)
                
                drawing_count = 0
                for sym in symbols:
                    matches = multi_scale_rotate_match(
                        d_arr, sym["template"], match_threshold
                    )
                    count = len(matches)
                    accumulator[sym["name"]]["Count"] += count
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
                    </div></div>""", 
                    unsafe_allow_html=True
                )
                
                if idx % 2 == 0 or idx == len(lighting_images)-1:
                    live_df = pd.DataFrame(list(accumulator.values()))
                    table_box.dataframe(
                        live_df, 
                        column_config={
                            "Legend Icon": st.column_config.ImageColumn(
                                "Symbol", width="small"
                            ),
                            "Count": st.column_config.NumberColumn(
                                "Count", format="%d ⚡"
                            ),
                        }, 
                        use_container_width=True, 
                        hide_index=True
                    )

            # Complete
            st.session_state.results_data = accumulator
            st.session_state.scan_complete = True
            st.rerun()

        except Exception as e:
            st.error(f"Critical Error: {str(e)}")
            st.code(traceback.format_exc())

elif st.session_state.scan_complete:
    df_summary = pd.DataFrame(list(st.session_state.results_data.values()))
    total_matches = sum(
        row['Count'] for row in st.session_state.results_data.values()
    )
    
    status_box.success(
        f"✅ COMPLETE! {total_matches} symbols detected."
    )
    metrics_box.empty()
    progress_bar.empty()
    status_log.empty()

    st.subheader(" Symbol Takeoff Schedule")
    st.dataframe(
        df_summary, 
        column_config={
            "Legend Icon": st.column_config.ImageColumn(
                "Symbol", width="small"
            ),
            "Count": st.column_config.NumberColumn(
                "Count", format="%d ⚡"
            ),
        }, 
        use_container_width=True, 
        hide_index=True
    )

    # Export buttons
    excel_output = io.BytesIO()
    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        df_summary.drop(columns=["Legend Icon"]).to_excel(
            writer, index=False, sheet_name="Takeoff"
        )
    
    pdf_output = io.BytesIO()
    c = canvas.Canvas(pdf_output, pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(54, h-50, "DrBuild LLC - Symbol Takeoff Report")
    c.setFont("Helvetica", 10)
    c.drawString(54, h-68, "Computer vision detection results.")
    c.line(54, h-78, w-54, h-78)
    y = h-105
    c.setFont("Helvetica-Bold", 10)
    c.drawString(54, y, "Category")
    c.drawString(180, y, "Description")
    c.drawString(380, y, "Package")
    c.drawString(490, y, "Count")
    y -= 20
    c.setFont("Helvetica", 9)
    for _, row in df_summary.iterrows():
        if y < 50: 
            c.showPage()
            y = h-50
            c.setFont("Helvetica", 9)
        c.drawString(54, y, str(row["System Category"]))
        c.drawString(180, y, str(row["Model / Description"]))
        c.drawString(380, y, str(row["Scan Package"]))
        c.drawString(490, y, str(row["Count"]))
        y -= 18
    c.save()

    col1, col2 = st.columns(2)
    with col1: 
        st.download_button(
            "📥 Excel (.xlsx)", 
            excel_output.getvalue(), 
            "DrBuild_Takeoff.xlsx", 
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col2: 
        st.download_button(
            " PDF Report", 
            pdf_output.getvalue(), 
            "DrBuild_Report.pdf", 
            "application/pdf"
        )

else:
    status_box.empty()
    metrics_box.empty()
    progress_bar.empty()
    status_log.empty()
    table_box.empty()
    st.info("Upload legend and drawings to begin.")
    st.caption(
        " Click 'Preview Symbols' to see text removal results before scanning."
    )
