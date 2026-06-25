# https://ultralytics.com/images/bus.jpg 中の物体を抽出し，その領域を元の画像に描画せよ
import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict("https://ultralytics.com/images/ex1-26.png", conf=0.2)

# 入力画像
img = results[0].orig_img

# 認識した物体領域を取得する．
boxes = results[0].boxes

person_count = 0

for box in boxes:
    # クラスIDを取得
    cls_id = int(box.cls[0])

    # person(クラスID=0)ならカウント
    if cls_id == 0:
        person_count += 1

    # 物体領域のxy座標を取得する．
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        thickness=3,
    )

# 人数を表示
print("人物数:", person_count)

# 画像にも表示
cv2.putText(
    img,
    f"Persons: {person_count}",
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 0, 0),
    2
)

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()