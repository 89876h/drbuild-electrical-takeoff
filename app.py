
Or if using Streamlit Cloud, deploy with this in requirements.txt.
""")
else:
with st.spinner("🔄 Processing... This may take a moment"):
    try:
        # Initialize detector
        detector = ElectricalSymbolDetector()
        detector.min_match_threshold = confidence_threshold
        
        # Step 1: Extract symbols from legend
        st.info("📖 Step 1: Extracting symbols from legend...")
        templates = detector.extract_symbols_from_legend(legend_image)
        
        if templates:
            st.success(f"✅ Found {len(templates)} potential receptacle symbols")
            
            # Show extracted templates
            st.markdown("#### Extracted Symbol Templates:")
            template_cols = st.columns(min(4, len(templates)))
            for i, template in enumerate(templates):
                with template_cols[i % 4]:
                    st.image(
                        template['binary'],
                        caption=f"Template {i+1}",
                        use_container_width=True
                    )
        else:
            st.warning("⚠️ No symbols found in legend. Trying contour detection...")
        
        # Step 2: Count in power page
        st.info("🔍 Step 2: Counting receptacles in power plan...")
        
        with st.status("Processing power plan...", expanded=True) as status:
            detections = detector.count_receptacles(power_image)
            
            if detections:
                status.update(
                    label=f"✅ Found {len(detections)} receptacles!",
                    state="complete"
                )
            else:
                # Try contour-based detection as fallback
                status.update(
                    label="Template matching failed, trying alternative...",
                    state="running"
                )
                
                gray, binary = detector.preprocess_image(power_image)
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                
                detections = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 100 < area < 5000:
                        x, y, w, h = cv2.boundingRect(contour)
                        if 15 < w < 200 and 15 < h < 200:
                            detections.append({
                                'x': x, 'y': y,
                                'width': w, 'height': h,
                                'confidence': 0.5
                            })
                
                status.update(
                    label=f"✅ Found {len(detections)} potential receptacles (contour method)",
                    state="complete"
                )
        
        # Display results
        if detections:
            st.markdown("---")
            st.markdown(f"""
            <div class="success-box">
                <h2>🎉 Detection Complete!</h2>
                <h1 style="text-align: center; font-size: 3rem;">{len(detections)}</h1>
                <p style="text-align: center;">Receptacles Detected</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show result image
            result_image = draw_detections(power_image, detections)
            st.image(
                result_image,
                caption=f"Detected {len(detections)} Receptacles",
                use_container_width=True
            )
            
            # Download options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download result image
                result_pil = Image.fromarray(result_image)
                buf = io.BytesIO()
                result_pil.save(buf, format='PNG')
                st.download_button(
                    label="📥 Download Result Image",
                    data=buf.getvalue(),
                    file_name=f"receptacle_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            
            with col2:
                # Download JSON results
                json_data = {
                    'total_count': len(detections),
                    'detections': [
                        {
                            'id': i+1,
                            'x': d['x'],
                            'y': d['y'],
                            'width': d['width'],
                            'height': d['height'],
                            'confidence': d['confidence']
                        }
                        for i, d in enumerate(detections)
                    ],
                    'timestamp': datetime.now().isoformat(),
                    'confidence_threshold': confidence_threshold
                }
                
                st.download_button(
                    label="📊 Download Results (JSON)",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"receptacle_count_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col3:
                # Summary statistics
                st.markdown("#### 📊 Statistics")
                confidences = [d['confidence'] for d in detections]
                st.metric("Average Confidence", f"{np.mean(confidences):.2%}")
                st.metric("Max Confidence", f"{np.max(confidences):.2%}")
                st.metric("Min Confidence", f"{np.min(confidences):.2%}")
            
            # Detailed results table
            st.markdown("#### 📋 Detailed Detection List")
            df_data = []
            for i, det in enumerate(detections):
                df_data.append({
                    "ID": i+1,
                    "X": det['x'],
                    "Y": det['y'],
                    "Width": det['width'],
                    "Height": det['height'],
                    "Confidence": f"{det['confidence']:.2%}"
                })
            
            st.dataframe(df_data, use_container_width=True)
            
        else:
            st.warning("""
            <div class="warning-box">
                <h3>⚠️ No Receptacles Detected</h3>
                <p>Try these solutions:</p>
                <ul>
                    <li>Lower the confidence threshold</li>
                    <li>Ensure legend and power pages are from the same project</li>
                    <li>Use higher resolution images</li>
                    <li>Check if symbols are clearly visible</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.exception(e)
else:
st.info("""
<div class="info-box">
<h3>👆 Upload Both Images to Begin</h3>
<p>Upload a legend page and power plan page, then click "Detect and Count Receptacles"</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<p>Electrical Receptacle Counter v1.0 | Built with Streamlit and OpenCV</p>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
main()
