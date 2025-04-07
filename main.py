import cv2
from ultralytics import YOLO
import datetime
import firebase_admin
from firebase_admin import credentials, db

if not firebase_admin._apps:
    cred = credentials.Certificate('C:/Users/ADMIN/PycharmProjects/bienxe/baidoxethongminh-a8054-firebase-adminsdk-fbsvc-aac5d4a671.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://baidoxethongminh-a8054-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

print("Firebase đã kết nối thành công!")

plate_detector = YOLO('C:/Users/ADMIN/PycharmProjects/bienxe/model3.pt')
char_detector = YOLO('C:/Users/ADMIN/PycharmProjects/bienxe/model6.pt')

image_path = 'BIEN11.jpg'
image = cv2.imread(image_path)

if image is None:
    print("❌ Không thể đọc ảnh!")
    exit()

image_resized = cv2.resize(image, (640, 384))
results = plate_detector(image, conf=0.3)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

for r in results:
    boxes = r.boxes
    class_names = r.names

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0])

        if confidence < 0.5:
            continue

        print(f"📍 Đã phát hiện biển số: ({x1}, {y1}) - ({x2}, {y2}), Conf: {confidence:.2f}")
        cropped_plate = image[y1:y2, x1:x2]
        char_results = char_detector(cropped_plate, conf=0.3)

        char_boxes = []

        for r in char_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                char_name = char_detector.names[cls]
                char_boxes.append((x1, y1, char_name))

        char_boxes.sort(key=lambda x: (x[1], x[0]))

        rows = []
        current_row = []

        plate_height = cropped_plate.shape[0]
        line_threshold = plate_height * 0.2

        current_y = char_boxes[0][1]

        for x1, y1, char_name in char_boxes:
            if abs(y1 - current_y) > line_threshold:
                rows.append(current_row)
                current_row = []
                current_y = y1
            current_row.append((x1, char_name))

        if current_row:
            rows.append(current_row)

        for row in rows:
            row.sort(key=lambda x: x[0])

        for i, row in enumerate(rows):
            print(f"📏 Hàng {i + 1} có {len(row)} ký tự:", [c for _, c in row])

        final_text = []
        for i, row in enumerate(rows):
            row_text = "".join([char[1] for char in row])

            if len(rows) == 1 and len(row) > 3:
                if len(row[3:]) == 5:
                    row_text = row_text[:3] + "*" + row_text[3:6] + "." + row_text[6:]
                else:
                    row_text = row_text[:3] + "*" + row_text[3:]

            elif i == 0 and len(row) == 4:
                row_text = row_text[:2] + "-" + row_text[2:]

            elif i == 1 and len(row) == 5:
                row_text = row_text[:3] + "." + row_text[3:]

            final_text.append(row_text)

        if len(final_text) > 1:
            final_text = final_text[0] + "*" + final_text[1]
        else:
            final_text = final_text[0]

        new_plate = final_text
        file_path = r"C:/Users/ADMIN/PycharmProjects/bienxe/name_car/name_car.txt"

        def is_plate_exists(file_path, plate):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.strip() == plate:
                            return True
            except FileNotFoundError:
                return False
            return False

        if not is_plate_exists(file_path, new_plate):
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(new_plate + "\n")
            print(f"Đã lưu kết quả vào file: {file_path}")
        else:
            print("Biển số đã tồn tại trong file, không lưu lại.")

        db.reference("license_plate").update({"name": final_text})
        print("Biển số xe nhận diện:", final_text)

        cv2.putText(cropped_plate, final_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("License Plate with Character Boxes", cropped_plate)
        cv2.waitKey(0)

cv2.destroyAllWindows()