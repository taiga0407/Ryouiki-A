import cv2
import math
from ultralytics import YOLO

# 学習済みモデル
model = YOLO("best260408.pt")

# パラメータ
CONF = 0.55          # 信頼度
MOVE_THRESH = 4      # 移動と判定する距離[pixel]
MOVING_THRESH = 3    # TEAM MOVINGとなる人数
STOP_FRAMES = 30     # TEAM STATIONARY継続フレーム数


def detect_snap(video_path):
    cap = cv2.VideoCapture(video_path)

    prev_centers = {}
    stationary_frames = 0
    prev_team_state = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        moving_count = 0

        results = model.track(
            frame,
            persist=True,
            conf=CONF,
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

                # 移動判定
                if move_dist >= MOVE_THRESH:
                    moving_count += 1

                prev_centers[track_id] = (cx, cy)

        # チーム状態判定
        if moving_count >= MOVING_THRESH:
            team_state = "TEAM MOVING"
        else:
            team_state = "TEAM STATIONARY"

        # STATIONARY継続時間を数える
        if team_state == "TEAM STATIONARY":
            stationary_frames += 1
        else:
            # STATIONARYが一定時間以上続いた後にMOVINGになった瞬間
            if prev_team_state == "TEAM STATIONARY" and stationary_frames >= STOP_FRAMES:
                cap.release()
                return frame_number

            stationary_frames = 0

        prev_team_state = team_state

    cap.release()

    # 見つからなかった場合
    return -1


# 2本の動画でスナップフレームを取得
frame1 = detect_snap("ex5-26.mp4")
frame2 = detect_snap("ex9-26.mp4")

print("ex5-26.mp4 :", frame1)
print("ex9-26.mp4 :", frame2)

if frame1 != -1 and frame2 != -1:
    print("Difference :", abs(frame1 - frame2))
else:
    print("スナップフレームが検出できませんでした。")