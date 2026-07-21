def extract_electrical_symbols(legend_img, category_filter="All"):
    """
    Extracts electrical symbols using contour analysis.
    UPDATED: Lowered size thresholds to catch small legend symbols.
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
        
        # Threshold to binary
        _, binary = cv2.threshold(section, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        extracted_symbols = []
        item_counter = 1
        prefix = "Device" if category_filter == "Power / Devices" else "Fixture"
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # --- RELAXED SIZE FILTERS ---
            # Lowered from 100 to 30 to catch small legend symbols
            if area < 30: 
                continue
            # Lowered from 15 to 8 to catch thin/small symbols
            if w < 8 or h < 8: 
                continue
            # Keep max size limit to avoid catching entire text paragraphs
            if w > 150 or h > 150: 
                continue
            
            # Extract symbol region with padding
            pad = 5
            sy1 = max(0, y - pad)
            sy2 = min(section.shape[0], y + h + pad)
            sx1 = max(0, x - pad)
            sx2 = min(section.shape[1], x + w + pad)
            
            symbol_crop = section[sy1:sy2, sx1:sx2]
            
            # Check if this is likely text
            aspect_ratio = w / h
            density = area / (w * h)
            
            is_text = False
            
            # 1. Very thin vertical strokes (I, l, 1) - kept strict
            if aspect_ratio < 0.25:
                is_text = True
            
            # 2. Very wide horizontal elements (-, _) - kept strict
            if aspect_ratio > 5.0:
                is_text = True
            
            # 3. Low density check - RELAXED threshold
            # Only reject if VERY sparse AND small (likely a comma or period)
            if density < 0.10 and area < 50:
                is_text = True
            
            # 4. Hole detection for letters (P, O, D, B)
            # We check if there's a significant enclosed white space
            has_hole = False
            # Create mask for this specific contour to check internal holes
            mask = np.zeros(section.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Invert mask to find holes inside the symbol
            hole_mask = cv2.bitwise_not(mask)
            # Count white pixels in the bounding box of the symbol
            bbox_area = w * h
            white_in_bbox = np.sum(hole_mask[y:y+h, x:x+w] == 255)
            
            # If >30% of the bounding box is empty space INSIDE the contour, 
            # and the shape is roughly square/vertical, it's likely a letter like P/O/D
            hole_ratio = white_in_bbox / max(1, bbox_area)
            
            if hole_ratio > 0.30 and aspect_ratio < 1.6 and density < 0.45:
                # Additional check: Letters usually have simpler shapes
                # Approximate contour to see complexity
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Simple polygons with holes are often letters
                if len(approx) < 10: 
                    is_text = True
            
            if is_text:
                continue
            
            # This looks like a symbol! Normalize it
            target_size = 60
            scale = target_size / max(w, h)
            new_w = max(15, int(w * scale))
            new_h = max(15, int(h * scale))
            
            symbol_resized = cv2.resize(symbol_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create icon bytes for display
            pil_chip = Image.fromarray(symbol_resized)
            img_byte_arr = io.BytesIO()
            pil_chip.save(img_byte_arr, format="PNG")
            
            extracted_symbols.append({
                "category": category_filter,
                "name": f"{prefix} Type {item_counter}",
                "icon_bytes": img_byte_arr.getvalue(),
                "template": symbol_resized,
                "contour": contour,
                "bbox": (x, y, w, h),
                "area": area,
                "aspect_ratio": aspect_ratio,
            })
            item_counter += 1
        
        print(f"Extracted {len(extracted_symbols)} symbols from legend")
        return extracted_symbols
        
    except Exception as e:
        st.error(f"Symbol extraction failed: {str(e)}")
        traceback.print_exc()
        return []
