from ultralytics import YOLO
import cv2 as cv
import pickle
import pandas as pd

class TrackBall:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_position(self,ball_positions):
        """
        Interpolates missing ball positions by usinng pd.DataFrame
        Args:
            ball_positions (list): List of ball positions in the video
        Returns
            ball_positions (list): List of updated ball positions in the video after interpolation
        """
        ball_positions = [x.get(1,[]) for x in ball_positions]
        ball_positions_df = pd.DataFrame(ball_positions,columns=["x1","y1","x2","y2"])

        #Interpolate missing values
        ball_positions_df = ball_positions_df.interpolate()
        ball_positions_df = ball_positions_df.bfill()
        print(f"\nBall Position:\n{ball_positions_df}")

        #Convert back to list form from dataframe
        ball_positions = [{1:x} for x in ball_positions_df.to_numpy().tolist()]
    
        return ball_positions
    
    #Detect and track the ball in the video
    def detect_frames(self,frames,read_from_stub=False,stub_path=None):
        """
        Detects and tracks ball in the video
        Args:
            frames (list): List of frames as an array
            read_from_stub (bool): If True, put data into a stub. Default = False
            stub_path (str): path to save the output so that we dont have to run the detector again

        Returns:
            ball_detection (list): List of dictionaries with ball id and bounding box coordinates
        """
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections
 
        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(ball_detections,f)
        
        return ball_detections

    #Detect and track the ball in the frame
    def detect_frame(self,frame):
        """
        Detects and tracks ball in the frame
        Args:
            frame (array): Input frame
        Returns:
            ball_dict (dict): Dictionary with ball id and bounding box coordinates
        """
        results = self.model.predict(frame,conf=0.2)[0] #Runs tracking on the input frame

        ball_dict = {} #Dictionary to store ball id and bounding box coordinates
        for box in results.boxes:
            result = box.xyxy.tolist()[0] #Add bounding box coordinates to the dictionary
            ball_dict[1] = result
        
        return ball_dict
    
    def draw_bounding_box(self,video_frames,ball_detections):
        """
        Draws bounding box around the ball in the video
        Args:
            video_frames (list): List of frames as an array
            player_detections (list): List of dictionaries with ball id and bounding box coordinates
        Returns:
            output_video_frames (list): List of frames with bounding box around the players
        """
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv.putText(frame, f"Tennis Ball",(int(bbox[0]),int(bbox[1] -10 )),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_video_frames.append(frame)

        return output_video_frames
        
        
  