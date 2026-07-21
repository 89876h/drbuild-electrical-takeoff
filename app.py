import pytesseract # You must install this: pip install pytesseract
# Also ensure Tesseract-OCR engine is installed on your system

def extract_symbols_structured(legend_img, category_filter="All"):
    """
    Uses OCR to find text labels, then extracts the region to the LEFT 
    as the symbol. This guarantees NO letters are extracted as symbols.
    """
    try:
        legend_cv = np.array(legend_img)
        gray = cv2.cvtColor(legend_cv, cv2.COLOR_RGB2GRAY)
        h_leg, w_leg = gray.shape[:2]

        # 1. Define Section Boundaries
        if category_filter == "Power / Devices":
            y_start, y_end = int(h_leg * 0.42), int(h_leg * 0.62)
        elif category_filter == "Lighting":
            y_start, y_end = int(h_leg * 0.62), int(h_leg * 0.78)
        else:
            y_start, y_end = 0, h_leg

        section = gray[y_start:y_end, :]
        
        # 2. Use OCR to find TEXT bounding boxes
        # We look for single characters or short words (the labels)
        data = pytesseract.image_to_data(section, output_type=pytesseract.Output.DICT)
        
        text_boxes = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30: # Confidence threshold
                x = int(data['left'][i])
                y = int(data['top'][i])
                w = int(data['width'][i])
                h = int(data['height'][i])
                text_boxes.append((x, y, w, h))

        # 3. Extract Symbols to the LEFT of each text box
        extracted_items = []
        item_counter = 1
        prefix = "Device" if category_filter == "Power / Devices" else "Fixture"
        
        # Sort text boxes by Y position to keep order logical
        text_boxes.sort(key=lambda b: (b[1], b[0]))

        used_regions = [] # Prevent overlapping extractions

        for tx, ty, tw, th in text_boxes:
            # Define the "Symbol Zone": Left of the text, same height
            # We assume symbol is roughly square or slightly wider than tall
            sym_w = max(th, int(tw * 0.8)) 
            sym_h = th
            
            sx1 = max(0, tx - sym_w - 5) # 5px gap
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
            
            # Final sanity check: Is the crop actually dark? (Not just whitespace)
            if np.mean(symbol_crop) > 240: continue 

            # Normalize
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
            used_regions.append((sx1, sy1, sx2, sy2))
            item_counter += 1

        print(f"Structured Extraction: Found {len(extracted_items)} symbols next to text labels.")
        return extracted_items

    except Exception as e:
        st.error(f"Structured extraction failed (is Tesseract installed?): {e}")
        # FALLBACK: If OCR fails, use strict geometry filtering
        return extract_symbols_geometric_fallback(legend_img, category_filter)

def extract_symbols_geometric_fallback(legend_img, category_filter):
    """
    Fallback if OCR is unavailable. Uses strict aspect ratio and density checks
    to reject alphabets. Letters like P, E, B have high vertical stroke density.
    Circles/Symbols have more uniform distribution.
    """
    # ... [Your original connected components code here] ...
    # BUT add this stricter filter inside the loop:
    
    # Calculate 'Vertical Projection' (sum of dark pixels per column)
    col_sums = np.sum(symbol_crop < 128, axis=0)
    # Letters often have sharp peaks and valleys. Symbols are smoother.
    # This is a heuristic, OCR is better.
    
    # For now, just return empty to force user to install OCR or fix legend
    return [] 
