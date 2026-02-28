from ultralytics import YOLO
import easyocr
import cv2
import re
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ===== Load Models =====
model = YOLO(r"E:\yolo License Plate Recognition\License-Plate-Recognition\lisence_detector.pt")
reader = easyocr.Reader(['en'], gpu=False)


source = "E:\yolo License Plate Recognition\License-Plate-Recognition\download.jpg"   # 
if not os.path.isabs(source):
    source = os.path.join(os.path.dirname(__file__), source)

# ===== Check File Type =====
image_ext = ['.jpg', '.jpeg', '.png']
video_ext = ['.mp4', '.avi', '.mov', '.mkv']

# Check if file exists
if not os.path.exists(source):
    print(f"Error: File '{source}' not found!")
    exit(1)

file_ext = os.path.splitext(source)[1].lower()

# ==========================================
# 🖼 IMAGE MODE
# ==========================================
if file_ext in image_ext:
    image = cv2.imread(source)
    
    if image is None:
        print(f"Error: Could not read image file '{source}'")
        exit(1)
    
    results = model(image)

    if results[0].boxes is None or len(results[0].boxes) == 0:
        print("No license plates detected in the image.")
    else:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = image[y1:y2, x1:x2]

            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2)

            result = reader.readtext(gray)

            if result:
                text = "".join([res[1] for res in result])
                text = re.sub(r'[^A-Z0-9]', '', text.upper())
                print("Detected Plate:", text)

                # Draw box and text
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(image, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,0), 2)

        # Show once after loop (fallback to saving if display not available)
        try:
            cv2.imshow("Number Plate Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            out_path = os.path.join(os.path.dirname(__file__), "detection_result.jpg")
            cv2.imwrite(out_path, image)
            print(f"Display not available. Saved result to {out_path}")
# ==========================================
# 🎥 VIDEO MODE
# ==========================================
elif file_ext in video_ext:
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{source}'")
        exit(1)

    # ===== Setup Video Writer =====
    out_path = os.path.join(os.path.dirname(__file__), "detection_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]

                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2, fy=2)

                result = reader.readtext(gray)

                if result:
                    text = "".join([res[1] for res in result])
                    text = re.sub(r'[^A-Z0-9]', '', text.upper())

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, text, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0,255,0), 2)

        # ✅ Always write frame
        writer.write(frame)

        # Optional display
        cv2.imshow("Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Video saved successfully at: {out_path}")
else:
    print("Unsupported file format!")