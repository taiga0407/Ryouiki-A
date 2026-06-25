import cv2
from ultralytics import YOLO

# 学習済みモデル
model = YOLO("best260408.pt")

# 動画ファイルを開く
cap = cv2.VideoCapture("ex5-26.mp4")

frame_count = 0  # フレーム数

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # 推論
    results = model.predict(frame, conf=0.55, verbose=False)

    # 検出結果
    boxes = results[0].boxes

    helmet_count = 0

    for box in boxes:
        cls_id = int(box.cls[0])

        # person
        if cls_id == 0:
            helmet_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                thickness=3
            )

    # 30フレームごとに表示
    if frame_count % 30 == 0:
        print(f"Frame {frame_count}: Helmet = {helmet_count}")

    # 人数表示
    cv2.putText(
        frame,
        f"Helmets: {helmet_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("YOLO Person Detection", frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()