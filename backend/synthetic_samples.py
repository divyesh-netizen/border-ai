import os
import cv2
import numpy as np
import math

def draw_pedestrian(img, cx, cy, height, is_thermal=False, stride=0.0):
    head_r = int(height * 0.12)
    head_center = (int(cx), int(cy - height * 0.38))
    
    torso_top = (int(cx), int(cy - height * 0.28))
    torso_bottom = (int(cx), int(cy + height * 0.08))
    torso_w = int(height * 0.24)

    leg_offset = int(math.sin(stride) * (height * 0.18))
    
    if is_thermal:
        body_color = (235, 250, 255) # Bright hot thermal LWIR
        core_color = (255, 255, 255)
    else:
        # High-contrast night security guard / patrol clothing (dark jacket with reflective trim)
        body_color = (75, 80, 95)
        core_color = (200, 220, 240)

    # Torso
    cv2.rectangle(img, (int(cx - torso_w//2), torso_top[1]), (int(cx + torso_w//2), torso_bottom[1]), body_color, -1)
    if not is_thermal:
        # Reflective stripe on jacket (realistic CCTV feature)
        cv2.line(img, (int(cx - torso_w//2), torso_top[1] + 12), (int(cx + torso_w//2), torso_top[1] + 12), core_color, 3)

    # Head
    cv2.circle(img, head_center, head_r, body_color, -1)

    # Legs
    cv2.line(img, (int(cx - 5), torso_bottom[1]), (int(cx - leg_offset), int(cy + height * 0.48)), body_color, int(height * 0.11))
    cv2.line(img, (int(cx + 5), torso_bottom[1]), (int(cx + leg_offset), int(cy + height * 0.48)), body_color, int(height * 0.11))

    # Arms
    arm_offset = int(math.cos(stride) * (height * 0.15))
    cv2.line(img, (int(cx - torso_w//2), torso_top[1] + 8), (int(cx - torso_w//2 - arm_offset), torso_top[1] + int(height * 0.24)), body_color, int(height * 0.09))
    cv2.line(img, (int(cx + torso_w//2), torso_top[1] + 8), (int(cx + torso_w//2 + arm_offset), torso_top[1] + int(height * 0.24)), body_color, int(height * 0.09))


def generate_sample_surveillance_videos(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    visible_path = os.path.join(output_dir, "sample_cctv_night.mp4")
    thermal_path = os.path.join(output_dir, "sample_thermal_night.mp4")

    width, height = 640, 480
    fps = 20
    duration_sec = 14
    total_frames = fps * duration_sec

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_vis = cv2.VideoWriter(visible_path, fourcc, fps, (width, height))
    writer_therm = cv2.VideoWriter(thermal_path, fourcc, fps, (width, height))

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        timecode = f"2026-08-29 23:14:{int(t):02d}.{int((t%1)*100):02d}"

        # 1. Low-Light Background (Clean gradient ground & dark sky)
        vis_frame = np.full((height, width, 3), (20, 22, 26), dtype=np.uint8)
        # Ground plane
        cv2.rectangle(vis_frame, (0, int(height*0.48)), (width, height), (32, 35, 42), -1)
        # Perimeter fence line
        cv2.line(vis_frame, (0, int(height*0.48)), (width, int(height*0.48)), (55, 60, 70), 2)
        for fx in range(20, width, 40):
            cv2.line(vis_frame, (fx, int(height*0.40)), (fx, int(height*0.48)), (45, 50, 58), 1)

        # 2. Thermal Background
        therm_frame = np.full((height, width, 3), (35, 25, 20), dtype=np.uint8)
        cv2.rectangle(therm_frame, (0, int(height*0.48)), (width, height), (50, 38, 28), -1)
        cv2.line(therm_frame, (0, int(height*0.48)), (width, int(height*0.48)), (75, 58, 42), 2)

        # Object A: Person 1 (Patrol moving left to right)
        p1_x = int(50 + t * 42) % (width + 60) - 30
        p1_y = int(height * 0.65)
        draw_pedestrian(vis_frame, p1_x, p1_y, height=115, is_thermal=False, stride=t * 6.5)
        draw_pedestrian(therm_frame, p1_x, p1_y, height=115, is_thermal=True, stride=t * 6.5)

        # Object B: Person 2 (Stationary loiterer near border fence - Sustained Presence)
        p2_x = 220
        p2_y = int(height * 0.60)
        draw_pedestrian(vis_frame, p2_x, p2_y, height=100, is_thermal=False, stride=math.sin(t*0.6)*0.2)
        draw_pedestrian(therm_frame, p2_x, p2_y, height=100, is_thermal=True, stride=math.sin(t*0.6)*0.2)

        # Object C: Patrol Vehicle (moving across perimeter after 2.5 seconds)
        if t > 2.5:
            veh_x = int(width - (t - 2.5) * 80)
            veh_y = int(height * 0.72)
            veh_w, veh_h = 160, 75
            if -180 < veh_x < width + 50:
                # Visible vehicle
                cv2.rectangle(vis_frame, (veh_x, veh_y), (veh_x + veh_w, veh_y + veh_h), (65, 70, 85), -1)
                cv2.rectangle(vis_frame, (veh_x + 35, veh_y - 25), (veh_x + 120, veh_y), (55, 60, 75), -1)
                # Headlight beams
                cv2.line(vis_frame, (veh_x, veh_y + 40), (max(0, veh_x - 90), veh_y + 60), (180, 220, 255), 3)

                # Thermal vehicle
                cv2.rectangle(therm_frame, (veh_x, veh_y), (veh_x + veh_w, veh_y + veh_h), (120, 150, 190), -1)
                cv2.rectangle(therm_frame, (veh_x + 35, veh_y - 25), (veh_x + 120, veh_y), (100, 130, 170), -1)
                cv2.circle(therm_frame, (veh_x + 25, veh_y + 35), 22, (250, 255, 255), -1)
                cv2.circle(therm_frame, (veh_x + 35, veh_y + veh_h - 4), 16, (230, 250, 255), -1)
                cv2.circle(therm_frame, (veh_x + 125, veh_y + veh_h - 4), 16, (230, 250, 255), -1)

        # Object D: Animal (crossing background)
        anim_x = int(480 - t * 25)
        anim_y = int(height * 0.44)
        if 30 < anim_x < width:
            cv2.ellipse(vis_frame, (anim_x, anim_y), (28, 15), 0, 0, 360, (50, 55, 65), -1)
            cv2.circle(vis_frame, (anim_x - 22, anim_y - 8), 8, (50, 55, 65), -1)
            cv2.line(vis_frame, (anim_x - 12, anim_y + 10), (anim_x - 14, anim_y + 24), (50, 55, 65), 2)
            cv2.line(vis_frame, (anim_x + 12, anim_y + 10), (anim_x + 14, anim_y + 24), (50, 55, 65), 2)

            cv2.ellipse(therm_frame, (anim_x, anim_y), (28, 15), 0, 0, 360, (210, 235, 250), -1)
            cv2.circle(therm_frame, (anim_x - 22, anim_y - 8), 8, (230, 245, 255), -1)

        # Header Watermark
        cv2.putText(vis_frame, f"CAM-04 PERIMETER-NORTH | {timecode}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 220), 1)
        cv2.putText(vis_frame, "MODE: VISIBLE CCTV LOW-LIGHT", (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        cv2.putText(therm_frame, f"CAM-04-IR THERMAL LWIR | {timecode}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1)
        cv2.putText(therm_frame, "MODE: THERMAL INFRARED (LLVIP ALIGNED)", (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        writer_vis.write(vis_frame)
        writer_therm.write(therm_frame)

    writer_vis.release()
    writer_therm.release()
    print(f"[SyntheticGenerator] Re-generated high-contrast surveillance video at {output_dir}")
    return visible_path, thermal_path
