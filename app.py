def run_strict_takeoff_module_live(legend_img, drawing_imgs, package_name, category_filter, status_placeholder, metrics_placeholder, table_placeholder, accumulator):
    """
    Performs strict computer vision extraction with LIVE UI updates.
    Tracks scanned objects vs matches found in real-time.
    """
    if not drawing_imgs or not legend_img:
        return

    valid_drawings = []
    for d_img in drawing_imgs:
        if d_img is None:
            continue
        try:
            d_arr = np.array(d_img)
            if len(d_arr.shape) == 3:
                d_arr = cv2.cvtColor(d_arr, cv2.COLOR_RGB2GRAY)
            valid_drawings.append(d_arr)
        except Exception:
            continue

    if not valid_drawings:
        status_placeholder.warning(f"No valid drawings found for {package_name}")
        return

    extracted_items = extract_symbols_from_legend(legend_img, category_filter)
    if not extracted_items:
        status_placeholder.warning(f"No symbols detected for '{category_filter}'.")
        return

    total_drawings = len(valid_drawings)
    cumulative_scanned = 0
    cumulative_matches = 0
    
    # Process each drawing and update UI immediately
    for idx, d_arr in enumerate(valid_drawings, 1):
        drawing_matches = 0
        
        # Accumulate counts for each symbol
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
                print(f"Match error for {item['name']}: {e}")
                continue

        # Update cumulative counters
        cumulative_scanned += 1
        cumulative_matches += drawing_matches

        # UPDATE METRICS DASHBOARD (Real-time counters)
        metrics_placeholder.markdown(
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

        # UPDATE STATUS TEXT (4-line constraint)
        status_text = (
            f"**Scanning:** {package_name}\n"
            f"**Progress:** Drawing {idx}/{total_drawings} | +{drawing_matches} new matches\n"
            f"**Cumulative:** {cumulative_scanned} scanned → {cumulative_matches} total matches\n"
            f"**Status:** Matching scale variants..."
        )
        status_placeholder.info(status_text)

        # Dynamically update the table placeholder
        if accumulator:
            live_df = pd.DataFrame(list(accumulator.values()))
            try:
                table_placeholder.dataframe(
                    live_df,
                    column_config={
                        "Legend Icon": st.column_config.ImageColumn("Legend Symbol", width="small"),
                        "Count": st.column_config.NumberColumn("Verified Count", format="%d ⚡"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception:
                table_placeholder.dataframe(live_df.drop(columns=["Legend Icon"], errors="ignore"))


# --- MAIN EXECUTION BLOCK ---
if process_btn:
    if not legend_file:
        st.error("Please upload the Legend Sheet to perform scans.")
    elif not power_files and not lighting_files:
        st.warning("Please upload at least one Power or Lighting drawing file.")
    else:
        # Create persistent placeholders for live updates
        status_box = st.empty()
        metrics_box = st.empty()      # NEW: Metrics dashboard
        table_box = st.empty()
        
        try:
            legend_image = load_image_safely(legend_file)
            if legend_image is None:
                st.error("Failed to load legend image.")
                st.stop()

            power_images = [load_image_safely(f) for f in (power_files or []) if load_image_safely(f)]
            lighting_images = [load_image_safely(f) for f in (lighting_files or []) if load_image_safely(f)]

            accumulated_data = {}

            # Run Power Scan with Live Updates
            run_strict_takeoff_module_live(
                legend_image, power_images, "Power Package", "Power / Devices", 
                status_box, metrics_box, table_box, accumulated_data
            )
            
            # Run Lighting Scan with Live Updates
            run_strict_takeoff_module_live(
                legend_image, lighting_images, "Lighting Package", "Lighting", 
                status_box, metrics_box, table_box, accumulated_data
            )

            # Finalize
            df_summary = pd.DataFrame(list(accumulated_data.values())) if accumulated_data else pd.DataFrame()
            
            if not df_summary.empty:
                # Clear metrics and show final success
                metrics_box.success(f"✅ SCAN COMPLETE! Total: {sum(row['Count'] for row in accumulated_data.values())} verified matches across {df_summary['Scan Package'].nunique()} packages.")
                
                st.subheader(" Itemized Takeoff Schedule")
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
                        label="📄 Download Takeoff Report (PDF)",
                        data=pdf_data,
                        file_name="DrBuild_Verified_Report.pdf",
                        mime="application/pdf",
                    )
            else:
                metrics_box.warning("No elements were matched. Try adjusting thresholds.")
                
        except Exception as e:
            st.error(f"Critical Application Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("Upload your legend sheet and optional drawing files in the sidebar to begin.")
