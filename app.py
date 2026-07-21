"""
Electrical Receptacle Counter - Streamlit Cloud App
No OpenCV required - Uses Pillow for all image processing
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import io
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Electrical Receptacle Counter",
    page_icon="🔌",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 1rem;
}
.success-box {
    padding: 2rem;
    background-color: #d4edda;
    border-radius: 1rem;
    border: 3px solid #28a745;
    color: #155724;
    text-align: center;
    margin: 1rem 0;
}
.info-box {
    padding: 2rem;
    background-color: #d1ecf1;
    border-radius: 1rem;
    border: 2px solid #17a2b8;
    margin: 1rem 0;
}
.warning-box {
    padding: 2rem;
    background-color: #fff3cd;
    border-radius: 1rem;
    border: 2px solid #ffc107;
    color: #856404;
    margin: 1rem 0;
}
.stButton > button {
    font-size: 1.2rem;
    padding: 1rem 2rem;
    font-weight: 600;
}
.metric-box {
    text-align: center;
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔌 Electrical Receptacle Counter</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Automatically detect and count electrical receptacle symbols in architectural drawings</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    
    confidence_threshold = st.slider(
        "Detection Sensitivity",
        min_value=0.3,
        max_value=0.95,
        value=0.7,
        step=0.05,
        help="Higher values = fewer but more accurate detections. Lower values = find more symbols but may include false positives."
    )
    
    min_symbol_size = st.slider(
        "Minimum Symbol Size (pixels)",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
        help="Minimum area of connected pixels to consider as a symbol"
    )
    
    max_symbol_size = st.slider(
        "Maximum Symbol Size (pixels)",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        help="Maximum area of connected pixels to consider as a symbol"
    )
    
    search_scales = st.multiselect(
        "Search Scales",
        options=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        default=[0.8, 0.9, 1.0, 1.1, 1.2],
        help="Different sizes to search for symbols"
    )
    
    st.markdown("---")
    st.markdown("## 📋 How to Use")
    st.markdown("""
    **Step 1:** Upload the **Legend Page** that shows electrical symbols and their descriptions
    
    **Step 2:** Upload the **Power Plan Page** where you want to count receptacles
    
    **Step 3:** Click **Detect & Count Receptacles** button
    
    **Step 4:** Review results and download if needed
    """)
    
    st.markdown("---")
    st.markdown("## 💡 Tips for Better Results")
    st.markdown("""
    - Use **high-resolution** scans (300 DPI or higher)
    - Ensure **good contrast** between symbols and background
    - Legend symbols should **match** power plan symbols
    - Adjust **sensitivity** if missing symbols or getting false positives
    - **Clear, clean** drawings work best
    """)
    
    st.markdown("---")
    st.markdown("## 📊 Expected Symbol Types")
    st.markdown("""
    Common receptacle symbols:
    - ⭕ Circle with two lines (duplex)
    - ⭕ Circle with one line (single)
    - ▢ Square with lines
    - Other standard electrical symbols
    """)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="sub-header">📄 Step 1: Upload Legend Page</p>', unsafe_allow_html=True)
    st.markdown("*The legend page shows what receptacle symbols look like*")
    
    legend_file = st.file_uploader(
        "Choose legend image file",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'tif'],
        key='legend',
        help="Upload the electrical legend page that defines receptacle symbols"
    )
    
    if legend_file is not None:
        try:
            legend_image = Image.open(legend_file)
            if legend_image.mode != 'RGB':
                legend_image = legend_image.convert('RGB')
            
            st.image(legend_image, caption=f"Legend: {legend_file.name}", use_container_width=True)
            
            # Show image info
            file_size = len(legend_file.getvalue()) / 1024  # KB
            st.info(f"📁 File: {legend_file.name} | Size: {file_size:.1f} KB | Dimensions: {legend_image.size[0]}x{legend_image.size[1]} px")
        
        except Exception as e:
            st.error(f"Error loading legend image: {str(e)}")

with col2:
    st.markdown('<p class="sub-header">🔌 Step 2: Upload Power Plan Page</p>', unsafe_allow_html=True)
    st.markdown("*The power plan page where receptacles will be counted*")
    
    power_file = st.file_uploader(
        "Choose power plan image file",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'tif'],
        key='power',
        help="Upload the power plan page to count receptacle symbols"
    )
    
    if power_file is not None:
        try:
            power_image = Image.open(power_file)
            if power_image.mode != 'RGB':
                power_image = power_image.convert('RGB')
            
            st.image(power_image, caption=f"Power Plan: {power_file.name}", use_container_width=True)
            
            # Show image info
            file_size = len(power_file.getvalue()) / 1024  # KB
            st.info(f"📁 File: {power_file.name} | Size: {file_size:.1f} KB | Dimensions: {power_image.size[0]}x{power_image.size[1]} px")
        
        except Exception as e:
            st.error(f"Error loading power plan image: {str(e)}")

# Processing section
if legend_file is not None and power_file is not None:
    st.markdown("---")
    st.markdown('<p class="sub-header">🔍 Step 3: Detect Receptacles</p>', unsafe_allow_html=True)
    
    if st.button("🔍 Detect & Count Receptacles", type="primary", use_container_width=True):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load and preprocess images
            status_text.text("Loading images...")
            progress_bar.progress(10)
            
            legend_img = Image.open(legend_file).convert('L')  # Grayscale
            power_img = Image.open(power_file).convert('L')    # Grayscale
            power_color = Image.open(power_file).convert('RGB')  # Color for display
            
            # Enhance images
            status_text.text("Enhancing images...")
            progress_bar.progress(20)
            
            legend_img = legend_img.filter(ImageFilter.SHARPEN)
            power_img = power_img.filter(ImageFilter.SHARPEN)
            
            # Binarize images (black and white)
            status_text.text("Converting to black and white...")
            progress_bar.progress(30)
            
            threshold = 128
            legend_bin = legend_img.point(lambda x: 0 if x < threshold else 255, '1')
            power_bin = power_img.point(lambda x: 0 if x < threshold else 255, '1')
            
            legend_array = np.array(legend_bin)
            power_array = np.array(power_bin)
            
            # Extract symbols from legend (focus on left portion where symbols usually are)
            status_text.text("Extracting symbols from legend...")
            progress_bar.progress(40)
            
            height_legend, width_legend = legend_array.shape
            
            # Focus on left 40% of legend (where symbols typically are)
            left_width = int(width_legend * 0.4)
            left_region = legend_array[:, :left_width]
            
            # Find connected components (symbols)
            def find_connected_components(binary_array, min_area=50, max_area=50000):
                """Find connected components in binary image"""
                visited = np.zeros_like(binary_array, dtype=bool)
                components = []
                
                for y in range(binary_array.shape[0]):
                    for x in range(binary_array.shape[1]):
                        if binary_array[y, x] == 0 and not visited[y, x]:
                            # Flood fill to find component
                            stack = [(y, x)]
                            pixels = []
                            min_x, min_y = x, y
                            max_x, max_y = x, y
                            
                            while stack:
                                cy, cx = stack.pop()
                                if (0 <= cy < binary_array.shape[0] and 
                                    0 <= cx < binary_array.shape[1] and 
                                    binary_array[cy, cx] == 0 and 
                                    not visited[cy, cx]):
                                    
                                    visited[cy, cx] = True
                                    pixels.append((cy, cx))
                                    
                                    min_x = min(min_x, cx)
                                    min_y = min(min_y, cy)
                                    max_x = max(max_x, cx)
                                    max_y = max(max_y, cy)
                                    
                                    # Check 8-connected neighbors
                                    for ny, nx in [
                                        (cy-1, cx-1), (cy-1, cx), (cy-1, cx+1),
                                        (cy, cx-1), (cy, cx+1),
                                        (cy+1, cx-1), (cy+1, cx), (cy+1, cx+1)
                                    ]:
                                        stack.append((ny, nx))
                            
                            area = len(pixels)
                            if min_area <= area <= max_area:
                                width = max_x - min_x + 1
                                height = max_y - min_y + 1
                                
                                # Basic shape filtering
                                aspect_ratio = width / height if height > 0 else 0
                                
                                if 0.2 < aspect_ratio < 5.0:  # Not too elongated
                                    components.append({
                                        'bbox': (min_x, min_y, max_x, max_y),
                                        'area': area,
                                        'width': width,
                                        'height': height,
                                        'aspect_ratio': aspect_ratio,
                                        'pixels': pixels
                                    })
                
                return components
            
            # Extract symbols from legend
            legend_symbols = find_connected_components(
                left_region, 
                min_area=min_symbol_size, 
                max_area=max_symbol_size
            )
            
            status_text.text(f"Found {len(legend_symbols)} potential symbols in legend")
            progress_bar.progress(50)
            
            # Display extracted templates
            if legend_symbols:
                st.markdown("#### 📋 Extracted Symbol Templates from Legend:")
                template_cols = st.columns(min(4, len(legend_symbols)))
                
                for i, symbol in enumerate(legend_symbols[:8]):  # Show max 8 templates
                    x1, y1, x2, y2 = symbol['bbox']
                    
                    # Extract template image
                    template_array = legend_array[y1:y2+1, x1:x2+1]
                    template_img = Image.fromarray(template_array.astype(np.uint8) * 255)
                    
                    with template_cols[i % 4]:
                        st.image(template_img, 
                               caption=f"Template {i+1} ({symbol['width']}x{symbol['height']})",
                               use_container_width=True)
            
            # Search for symbols in power page
            status_text.text("Searching for receptacles in power plan...")
            progress_bar.progress(60)
            
            detections = []
            search_height, search_width = power_array.shape
            
            # For each template, search in power page
            for idx, symbol in enumerate(legend_symbols[:10]):  # Limit to first 10 templates
                x1, y1, x2, y2 = symbol['bbox']
                template = legend_array[y1:y2+1, x1:x2+1]
                template_h, template_w = template.shape
                
                # Skip if template is too small or too large
                if template_h < 10 or template_w < 10:
                    continue
                if template_h > search_height or template_w > search_width:
                    continue
                
                # Multi-scale search
                for scale in search_scales:
                    new_h = int(template_h * scale)
                    new_w = int(template_w * scale)
                    
                    if new_h < 10 or new_w < 10:
                        continue
                    if new_h > search_height or new_w > search_width:
                        continue
                    
                    # Resize template
                    template_pil = Image.fromarray(template.astype(np.uint8) * 255)
                    template_scaled = np.array(
                        template_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    ) // 255  # Convert back to 0/1
                    
                    # Sliding window search with step size
                    step_size = max(2, min(new_w, new_h) // 4)
                    
                    for y in range(0, search_height - new_h + 1, step_size):
                        for x in range(0, search_width - new_w + 1, step_size):
                            # Extract patch
                            patch = power_array[y:y+new_h, x:x+new_w]
                            
                            # Calculate similarity (normalized cross-correlation)
                            if patch.shape == template_scaled.shape:
                                # Count matching pixels
                                matching = np.sum(patch == template_scaled)
                                total = template_scaled.size
                                similarity = matching / total
                                
                                if similarity >= confidence_threshold:
                                    detections.append({
                                        'x': int(x),
                                        'y': int(y),
                                        'width': new_w,
                                        'height': new_h,
                                        'confidence': float(similarity),
                                        'template_idx': idx,
                                        'scale': scale
                                    })
                
                # Update progress
                progress = 60 + int((idx + 1) / min(10, len(legend_symbols)) * 30)
                progress_bar.progress(min(progress, 90))
                status_text.text(f"Searching with template {idx+1}/{min(10, len(legend_symbols))}... Found {len(detections)} candidates so far")
            
            # Remove overlapping detections (Non-Maximum Suppression)
            status_text.text("Removing duplicate detections...")
            progress_bar.progress(95)
            
            if detections:
                # Sort by confidence
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                
                # NMS
                final_detections = []
                used_areas = []
                
                for det in detections:
                    # Check overlap with existing detections
                    overlap = False
                    det_x1, det_y1 = det['x'], det['y']
                    det_x2, det_y2 = det['x'] + det['width'], det['y'] + det['height']
                    
                    for existing in final_detections:
                        ex_x1, ex_y1 = existing['x'], existing['y']
                        ex_x2, ex_y2 = existing['x'] + existing['width'], existing['y'] + existing['height']
                        
                        # Calculate intersection over area
                        ix1 = max(det_x1, ex_x1)
                        iy1 = max(det_y1, ex_y1)
                        ix2 = min(det_x2, ex_x2)
                        iy2 = min(det_y2, ex_y2)
                        
                        if ix2 > ix1 and iy2 > iy1:
                            intersection = (ix2 - ix1) * (iy2 - iy1)
                            det_area = det['width'] * det['height']
                            
                            if intersection / det_area > 0.3:  # 30% overlap threshold
                                overlap = True
                                break
                    
                    if not overlap:
                        final_detections.append(det)
                
                detections = final_detections
            
            progress_bar.progress(100)
            status_text.text("Detection complete!")
            
            # Display results
            st.markdown("---")
            
            if detections:
                # Success message with count
                st.markdown(f"""
                <div class="success-box">
                    <h2>🎉 Detection Complete!</h2>
                    <h1 style="font-size: 5rem; margin: 1rem 0;">{len(detections)}</h1>
                    <h3>Receptacle Symbols Detected</h3>
                    <p>Confidence threshold: {confidence_threshold:.0%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Draw detections on image
                result_image = power_color.copy()
                draw = ImageDraw.Draw(result_image)
                
                # Color coding based on confidence
                for i, det in enumerate(detections):
                    confidence = det['confidence']
                    
                    if confidence >= 0.8:
                        color = 'lime'  # Green for high confidence
                        width = 3
                    elif confidence >= 0.6:
                        color = 'yellow'  # Yellow for medium confidence
                        width = 2
                    else:
                        color = 'red'  # Red for low confidence
                        width = 2
                    
                    # Draw rectangle
                    draw.rectangle(
                        [det['x'], det['y'], 
                         det['x'] + det['width'], det['y'] + det['height']],
                        outline=color,
                        width=width
                    )
                    
                    # Draw number label
                    draw.text(
                        (det['x'] + 2, det['y'] - 15),
                        f"#{i+1}",
                        fill=color
                    )
                
                # Display result image
                st.image(result_image, 
                        caption=f"Power Plan with {len(detections)} Detected Receptacles",
                        use_container_width=True)
                
                # Download buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download image
                    buf = io.BytesIO()
                    result_image.save(buf, format='PNG', quality=95)
                    st.download_button(
                        label="📥 Download Result Image",
                        data=buf.getvalue(),
                        file_name=f"receptacle_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col2:
                    # Download JSON data
                    json_data = {
                        'total_count': len(detections),
                        'detection_settings': {
                            'confidence_threshold': confidence_threshold,
                            'min_symbol_size': min_symbol_size,
                            'max_symbol_size': max_symbol_size,
                            'scales_used': search_scales
                        },
                        'detections': [
                            {
                                'id': i + 1,
                                'x': d['x'],
                                'y': d['y'],
                                'width': d['width'],
                                'height': d['height'],
                                'confidence': d['confidence'],
                                'template_index': d['template_idx'],
                                'scale': d['scale']
                            }
                            for i, d in enumerate(detections)
                        ],
                        'timestamp': datetime.now().isoformat(),
                        'legend_file': legend_file.name,
                        'power_file': power_file.name
                    }
                    
                    st.download_button(
                        label="📊 Download Results (JSON)",
                        data=json.dumps(json_data, indent=2),
                        file_name=f"receptacle_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    # Statistics
                    st.markdown("#### 📈 Statistics")
                    confidences = [d['confidence'] for d in detections]
                    
                    st.metric("Average Confidence", f"{np.mean(confidences):.1%}")
                    st.metric("Max Confidence", f"{np.max(confidences):.1%}")
                    st.metric("Min Confidence", f"{np.min(confidences):.1%}")
                    
                    # Confidence distribution
                    high_conf = sum(1 for c in confidences if c >= 0.8)
                    med_conf = sum(1 for c in confidences if 0.6 <= c < 0.8)
                    low_conf = sum(1 for c in confidences if c < 0.6)
                    
                    st.markdown(f"""
                    - 🟢 High confidence (>80%): {high_conf}
                    - 🟡 Medium confidence (60-80%): {med_conf}
                    - 🔴 Low confidence (<60%): {low_conf}
                    """)
                
                # Detailed results table
                st.markdown("#### 📋 Detailed Detection List")
                
                # Create data for table
                table_data = []
                for i, det in enumerate(detections[:50]):  # Show first 50
                    table_data.append({
                        "ID": i + 1,
                        "X": det['x'],
                        "Y": det['y'],
                        "Width": det['width'],
                        "Height": det['height'],
                        "Confidence": f"{det['confidence']:.1%}",
                        "Scale": f"{det['scale']:.1f}x"
                    })
                
                if table_data:
                    st.dataframe(
                        table_data,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    if len(detections) > 50:
                        st.info(f"Showing first 50 of {len(detections)} detections")
                
            else:
                # No detections found
                st.markdown("""
                <div class="warning-box">
                    <h3>⚠️ No Receptacles Detected</h3>
                    <p>Try these troubleshooting steps:</p>
                    <ol>
                        <li><strong>Lower the sensitivity threshold</strong> (try 0.5 or 0.4)</li>
                        <li><strong>Add more search scales</strong> (include 0.5, 0.6, 1.3, 1.4)</li>
                        <li><strong>Adjust symbol size range</strong> (widen min/max values)</li>
                        <li><strong>Use higher resolution images</strong> (300 DPI or more)</li>
                        <li><strong>Ensure good contrast</strong> between symbols and background</li>
                        <li><strong>Verify legend and power plan</strong> are from the same project</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.exception(e)
            progress_bar.progress(100)
            status_text.text("Error occurred")

else:
    # No images uploaded yet
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <h3>👆 Upload Images to Begin Detection</h3>
        <p>Please upload both a <strong>Legend Page</strong> and a <strong>Power Plan Page</strong> above.</p>
        <p>The app will:</p>
        <ol>
            <li>Extract receptacle symbols from the legend</li>
            <li>Search for matching symbols in the power plan</li>
            <li>Count all detected receptacles</li>
            <li>Provide downloadable results</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🔌 <strong>Electrical Receptacle Counter</strong> v1.0</p>
    <p>Built with Streamlit • No installation required • Works in any browser</p>
    <p style="font-size: 0.8rem;">Uses Pillow for image processing (no OpenCV needed)</p>
</div>
""", unsafe_allow_html=True)
