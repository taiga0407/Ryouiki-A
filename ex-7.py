import cv2
import math
from ultralytics import YOLO

# 学習済みモデル
model = YOLO("best260408.pt")

# 動画
cap = cv2.VideoCapture("ex5-26.mp4")

# 前フレームの中心座標
prev_centers = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 赤枠ヘルメット数
    moving_count = 0

    results = model.track(
        frame,
        persist=True,
        conf=0.55,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes.id is not None:

        for box, track_id in zip(boxes, boxes.id):

            track_id = int(track_id)

            cls_id = int(box.cls[0])

            # 青ヘルメットのみ
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 中心座標
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            move_dist = 0

            if track_id in prev_centers:
                px, py = prev_centers[track_id]

                move_dist = math.sqrt(
                    (cx - px) ** 2 +
                    (cy - py) ** 2
                )

            # 4ピクセル以上移動
            if move_dist >= 4:
                color = (0, 0, 255)  # 赤
                moving_count += 1
            else:
                color = (0, 255, 0)  # 緑

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                3,
                color,
                -1
            )

            prev_centers[track_id] = (cx, cy)

    # チーム状態判定
    if moving_count >= 3:
        team_state = "TEAM MOVING"
        text_color = (0, 0, 255)
    else:
        team_state = "TEAM STATIONARY"
        text_color = (0, 255, 0)

    # 右上に表示
    cv2.putText(
        frame,
        team_state,
        (frame.shape[1] - 300, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2
    )

    cv2.imshow("Helmet Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()