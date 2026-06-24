import cv2
from ultralytics import YOLO

# モデル読み込み
model = YOLO("yolov8x.pt")

# 画像読み込み
img = cv2.imread("ex2-26.png")

# -------------------
# 検出対象領域（ROI）
# -------------------
roi_x1 = 0
roi_y1 = 250
roi_x2 = 1000
roi_y2 = 550

roi = img[roi_y1:roi_y2, roi_x1:roi_x2]

# 人物のみ検出
results = model.predict(
    roi,
    conf=0.05,
    classes=[0],  # personのみ
    verbose=False
)

boxes = results[0].boxes

person_count = 0

for box in boxes:

    person_count += 1

    # ROI内での座標
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # 元画像の座標へ変換
    x1 += roi_x1
    x2 += roi_x1
    y1 += roi_y1
    y2 += roi_y1

    # 赤枠描画
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )

# ROI範囲を確認したい場合
# cv2.rectangle(
#     img,
#     (roi_x1, roi_y1),
#     (roi_x2, roi_y2),
#     (255, 0, 0),
#     2
# )

# 人数表示
cv2.putText(
    img,
    f"Persons: {person_count}",
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 0, 0),
    2
)

print("人物数:", person_count)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()