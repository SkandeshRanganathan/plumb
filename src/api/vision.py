import cv2
import numpy as np
import base64

def analyze_broadcast_frame(b64_img):
    """
    Analyzes a base64 image from the live broadcast to extract tactical insights.
    Returns detected pitch conditions and fielder setup heuristics.
    """
    try:
        # Handle Data URI scheme if present (e.g., "data:image/jpeg;base64,...")
        if ',' in b64_img:
            b64_img = b64_img.split(',')[1]
            
        img_data = base64.b64decode(b64_img)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Failed to decode OpenCV image"}
            
        h, w, _ = img.shape
        
        # --- 1. PITCH CLASSIFICATION ---
        # Heuristic: The cricket pitch usually occupies the center-bottom of the frame 
        # in the standard wide broadcast camera angle.
        crop_y_start = int(h * 0.4)
        crop_y_end = int(h * 0.85)
        crop_x_start = int(w * 0.35)
        crop_x_end = int(w * 0.65)
        
        pitch_roi = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        
        if pitch_roi.size == 0:
            return {"status": "success", "pitch_type": "Unknown", "dominant_color": "N/A"}
            
        # Calculate the average RGB color of the pitch area
        avg_color_per_row = np.average(pitch_roi, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        avg_b, avg_g, avg_r = avg_color
        
        pitch_type = "Standard True Pitch"
        
        # Computer Vision Heuristics for Pitch Color
        # Green > Red usually indicates live grass on the wicket
        if avg_g > (avg_r + 15) and avg_g > avg_b:
            pitch_type = "Green Seaming Pitch"
        # High Red and Green (Yellow/Brownish) with low blue indicates dry dirt
        elif avg_r > 120 and avg_g > 110 and avg_b < 100:
            pitch_type = "Dry Spinning Pitch"
        # Overall dark indicates dampness or very dark soil
        elif avg_r < 90 and avg_g < 90 and avg_b < 90:
            pitch_type = "Dark Damp Pitch"
            
        # --- 2. OUTFIELD FIELDER ESTIMATION (Experimental) ---
        # We can look for non-green objects in the green outfield to estimate fielders
        # For a quick prototype, we just count edges in the outer third of the screen
        outfield_roi = img[int(h*0.2):int(h*0.5), int(w*0.7):w]
        gray = cv2.cvtColor(outfield_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (outfield_roi.shape[0] * outfield_roi.shape[1])
        
        field_setup = "Standard"
        if edge_density > 0.05:
            field_setup = "Defensive (Many fielders deep)"
        else:
            field_setup = "Attacking (Fielders in the ring)"
            
        return {
            "status": "success",
            "pitch_type": pitch_type,
            "field_setup": field_setup,
            "avg_rgb": f"R:{int(avg_r)} G:{int(avg_g)} B:{int(avg_b)}"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
