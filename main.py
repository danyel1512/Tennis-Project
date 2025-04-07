import cv2 as cv
import time
from utils import (read_video,save_video,get_court_layout)
from trackers import TrackPlayer,TrackBall
from court_line_detector import LineDetector
from minimap import DrawMinimap

def main():
    #Start timer to count FPS
    start_time = time.time()

    #Read video
    input_video_path = "input_images/input_video.mp4"
    video_frames = read_video(input_video_path)

    #Detecting keypoints in the video
    keypoints_detector = LineDetector("models/keypoints_model.pth")
    keypoint_detected = keypoints_detector.predict(video_frames[0])
    court_layout = get_court_layout(keypoint_detected)
    print(f"Number of keypoints detected: {len(keypoint_detected)}")
    print(f"{keypoint_detected}")
    print(f"Court layout: {court_layout}")

    #Detecting players and tennis ball in the video
    player_tracker = TrackPlayer(model_path="models/yolov8x.pt")
    ball_tracker = TrackBall(model_path="models/yolov8_tennisball_best.pt") 
    minimap = DrawMinimap(video_frames[0])

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detection.pkl",
                                                     court_layout=court_layout)
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path='tracker_stubs/ball_detection.pkl')
    
    ball_detections = ball_tracker.interpolate_ball_position(ball_detections)
    print(f"Number of ball detections: {len(ball_detections)}")

    #Choose players
    player_detections,other_detection,role_assignments = player_tracker.assign_and_filter_roles(court_layout,player_detections)

    print(f"role assignment: {role_assignments[0]}")
    print(f"player detections: {player_detections[0]}")
    print(f"other detections: {other_detection[0]}")

    #Convert court positions to minimap positions
    player_minimap_detections,ball_minimap_detections = minimap.convert_bbox_to_minimap_coor(player_detections,ball_detections,keypoint_detected)

    #Draw stuff
    ##Draw bounding box around the players
    output_video_frames = player_tracker.draw_player_bounding_box(video_frames,player_detections)
    output_video_frames = player_tracker.draw_others_bounding_box(output_video_frames,other_detection,role_assignments)
    output_video_frames = ball_tracker.draw_bounding_box(output_video_frames,ball_detections)
                                                           
    ##Draw keypoints on the video
    output_video_frames = keypoints_detector.draw_keypoints_on_video(output_video_frames,
                                                          keypoint_detected)
    
    ##Draw minimap on the video
    output_video_frames = minimap.draw_minimap(output_video_frames)

    #Draw points on the minimap
    output_video_frames = minimap.draw_points_on_minimap(output_video_frames,player_minimap_detections,(0,0,255))
    output_video_frames = minimap.draw_points_on_minimap(output_video_frames,ball_minimap_detections,(0,255,0))

    #Draw fps in video (for future real time)
    end_time = time.time()
    total_time = end_time-start_time
    fps = len(output_video_frames) / total_time #FPS = No. of frames / Time(s)
    #for frame in (output_video_frames):
        #cv.putText(frame,f"{int(fps)} fps",(10,30),cv.FONT_HERSHEY_COMPLEX,1,(160,160,0),3)

    #Draw frame counter of the video
    for i, frame in enumerate(output_video_frames):
        cv.putText(frame,f"{i}",(10,30),cv.FONT_HERSHEY_COMPLEX,1,(0,110,110),3)

    #Save the video
    save_video(output_video_frames, "output videos/output_video.avi")

if __name__ == "__main__":
    main() 