"""
Electrical Receptacle Counter - Streamlit Cloud App
"""

import streamlit as st
import numpy as np
import io
from PIL import Image
from datetime import datetime

# OpenCV import
try:
    import cv2
except ImportError:
    st.error("❌ OpenCV not loaded. Check requirements.txt and packages.txt")
    st.stop()

# Page config
st.set_page_config(
    page_title="Electrical Receptacle Counter",
    page_icon="🔌",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.success-box {
    padding: 2rem;
    background-color: #d4edda;
    border-radius: 1rem;
    border: 2px solid #28a745;
    color: #155724;
    text-align: center;
}
.info-box {
    padding: 2rem;
    background-color: #d1ecf1;
    border-radius: 1rem;
    border: 2px solid #17a2b8;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔌 Electrical Receptacle Counter</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider(
        "Detection Confidence",
        min_value=0.3,
        max_value=0.95,
        value=0.7,
        step=0.05,
        help="Higher = fewer but more accurate detections"
    )
    
    st.markdown("---")
    st.header("📋 Instructions")
    st.markdown("""
    1. Upload **Legend Page** (shows receptacle symbols)
    2. Upload **Power Plan** (page to count receptacles)
    3. Click **Detect & Count**
    4. Download results
    """)
    
    st.markdown("---")
    st.header("💡 Tips")
    st.markdown("""
    - Use clear, high-resolution scans
    - Symbols in legend should match power page
    - Lower confidence = find more receptacles
    - Higher confidence = fewer false positives
    """)

# Main columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Step 1: Upload Legend Page")
    legend_file = st.file_uploader(
        "Choose legend image",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload the electrical legend page showing receptacle symbols"
    )
    if legend_file:
        legend_image = Image.open(legend_file)
        st.image(legend_image, caption="Legend Page", use_column_width=True)
        st.success(f"✅ Loaded: {legend_file.name}")

with col2:
    st.subheader("🔌 Step 2: Upload Power Plan Page")
    power_file = st.file_uploader(
        "Choose power plan image",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload the power plan page to count receptacles"
    )
    if power_file:
        power_image = Image.open(power_file)
        st.image(power_image, caption="Power Plan Page", use_column_width=True)
        st.success(f"✅ Loaded: {power_file.name}")

# Processing
if legend_file and power_file:
    st.markdown("---")
    
    if st.button("🔍 Detect & Count Receptacles", type="primary", use_container_width=True):
        
        with st.spinner("🔄 Processing images... Please wait"):
            
            try:
                # Load images as arrays
                legend_array = np.array(Image.open(legend_file).convert('RGB'))
                power_array = np.array(Image.open(power_file).convert('RGB'))
                
                # Convert to grayscale
                legend_gray = cv2.cvtColor(legend_array, cv2.COLOR_RGB2GRAY)
                power_gray = cv2.cvtColor(power_array, cv2.COLOR_RGB2GRAY)
                
                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                legend_enhanced = clahe.apply(legend_gray)
                power_enhanced = clahe.apply(power_gray)
                
                # Binarize
                _, legend_binary = cv2.threshold(legend_enhanced, 127, 255, cv2.THRESH_BINARY_INV)
                _, power_binary = cv2.threshold(power_enhanced, 127, 255, cv2.THRESH_BINARY_INV)
                
                # Clean up with morphology
                kernel = np.ones((2,2), np.uint8)
                legend_binary = cv2.morphologyEx(legend_binary, cv2.MORPH_CLOSE, kernel)
                power_binary = cv2.morphologyEx(power_binary, cv2.MORPH_CLOSE, kernel)
                
                st.info("📖 Extracting symbols from legend...")
                
                # Extract symbols from legend (left portion)
                h, w = legend_binary.shape
                left_portion = legend_binary[:, :int(w * 0.4)]
                
                # Find connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    left_portion, connectivity=8
                )
                
                # Extract templates
                templates = []
                for i in range(1, num_labels):
                    x, y, bw, bh, area = stats[i]
                    
                    # Filter by size (receptacle symbols)
                    if 200 < area < 5000 and 15 < bw < 200 and 15 < bh < 200:
                        # Extract symbol with padding
                        pad = 5
                        y1 = max(0, y - pad)
                        y2 = min(h, y + bh + pad)
                        x1 = max(0, x - pad)
                        x2 = min(w, x + bw + pad)
                        
                        symbol = legend_binary[y1:y2, x1:x2]
                        
                        if symbol.shape[0] > 20 and symbol.shape[1] > 20:
                            templates.append({
                                'binary': symbol,
                                'size': (symbol.shape[1], symbol.shape[0])
                            })
                
                st.success(f"✅ Found {len(templates)} potential receptacle templates")
                
                if templates:
                    # Show templates
                    st.markdown("#### Extracted Templates:")
                    cols = st.columns(min(4, len(templates)))
                    for i, template in enumerate(templates[:8]):
                        with cols[i % 4]:
                            st.image(template['binary'], 
                                   caption=f"Template {i+1}", 
                                   use_column_width=True)
                
                st.info("🔍 Searching for receptacles in power plan...")
                
                # Detect in power page
                all_detections = []
                
                for template in templates[:10]:  # Limit to first 10 templates
                    template_img = template['binary'].astype(np.uint8)
                    h_t, w_t = template_img.shape
                    
                    # Multi-scale detection
                    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                        try:
                            scaled = cv2.resize(template_img, None, 
                                              fx=scale, fy=scale, 
                                              interpolation=cv2.INTER_LINEAR)
                        except:
                            continue
                        
                        h_s, w_s = scaled.shape
                        
                        if h_s > power_binary.shape[0] or w_s > power_binary.shape[1]:
                            continue
                        if h_s < 15 or w_s < 15:
                            continue
                        
                        # Template matching
                        result = cv2.matchTemplate(
                            power_binary, scaled, 
                            cv2.TM_CCOEFF_NORMED
                        )
                        
                        locations = np.where(result >= confidence)
                        
                        for pt in zip(*locations[::-1]):
                            all_detections.append({
                                'x': int(pt[0]),
                                'y': int(pt[1]),
                                'width': int(w_s),
                                'height': int(h_s),
                                'confidence': float(result[pt[1], pt[0]])
                            })
                
                # Remove overlapping detections
                if all_detections:
                    all_detections.sort(key=lambda x: x['confidence'], reverse=True)
                    final_detections = [all_detections[0]]
                    
                    for det in all_detections[1:]:
                        overlap = False
                        for existing in final_detections:
                            if (abs(det['x'] - existing['x']) < 30 and 
                                abs(det['y'] - existing['y']) < 30):
                                overlap = True
                                break
                        if not overlap:
                            final_detections.append(det)
                    
                    detections = final_detections
                else:
                    detections = []
                
                # Display results
                st.markdown("---")
                
                if detections:
                    st.markdown(f"""
                    <div class="success-box">
                        <h2>🎉 Detection Complete!</h2>
                        <h1 style="font-size: 4rem;">{len(detections)}</h1>
                        <h3>Receptacles Found</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Draw detections
                    result_img = power_array.copy()
                    for i, det in enumerate(detections):
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(
                            result_img,
                            (det['x'], det['y']),
                            (det['x'] + det['width'], det['y'] + det['height']),
                            color, 3
                        )
                        cv2.putText(
                            result_img,
                            f"#{i+1}",
                            (det['x'], det['y'] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1
                        )
                    
                    st.image(result_img, 
                           caption=f"Detected {len(detections)} Receptacles",
                           use_column_width=True)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        result_pil = Image.fromarray(result_img)
                        buf = io.BytesIO()
                        result_pil.save(buf, format='PNG', quality=95)
                        st.download_button(
                            label="📥 Download Result Image",
                            data=buf.getvalue(),
                            file_name=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    with col2:
                        import json
                        json_data = json.dumps({
                            'total_count': len(detections),
                            'detections': [
                                {
                                    'id': i+1,
                                    'x': d['x'],
                                    'y': d['y'],
                                    'confidence': d['confidence']
                                }
                                for i, d in enumerate(detections)
                            ]
                        }, indent=2)
                        
                        st.download_button(
                            label="📊 Download Data (JSON)",
                            data=json_data,
                            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    # Statistics
                    st.markdown("#### 📊 Statistics")
                    confidences = [d['confidence'] for d in detections]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Found", len(detections))
                    with col2:
                        st.metric("Avg Confidence", f"{np.mean(confidences):.1%}")
                    with col3:
                        st.metric("Max Confidence", f"{np.max(confidences):.1%}")
                
                else:
                    st.warning("""
                    <div class="info-box">
                        <h3>⚠️ No Receptacles Detected</h3>
                        <p>Try these solutions:</p>
                        <ul>
                            <li>Lower the confidence threshold (try 0.5 or 0.4)</li>
                            <li>Use higher resolution images</li>
                            <li>Ensure legend and power page are from same project</li>
                            <li>Check if symbols are clearly visible</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

else:
    if not legend_file and not power_file:
        st.info("""
        <div class="info-box">
            <h3>👆 Upload Images to Start</h3>
            <p>Upload both a legend page and power plan page to detect and count receptacles.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>🔌 Electrical Receptacle Counter | Built with Streamlit | Works on any browser</p>
</div>
""", unsafe_allow_html=True)
