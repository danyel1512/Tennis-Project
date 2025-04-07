from ultralytics import YOLO
import cv2 as cv
import pickle
from utils import centre_of_bbox,measure_dist,get_foot_position,measure_xy_dist,distance_point_to_segment

class TrackPlayer:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    #Assign roles in the video based on proximity of the keypoints
    def assign_and_filter_roles(self,court_layout,player_detections):
        """
        Assigns roles to players based on their proximity to the court lines
        Args:
            court_layout (dict): Dictionary of court description
            player_detections (list): List of dictionaries with player id and bounding box coordinates
        Returns:
            filtered_players_detection (list): List of dictionaries with player id and bounding box coordinates
            filtered_others_detection (list): List of dictionaries with other people id and bounding box coordinates
            assign_roles_per_frame (list): List of dictionaries with player id and their roles
        """
        detections_first_frame = player_detections[18] #Detect the 17th frame since ball kid is only present in the 17th frame
        filtered_players_detection = [] #Filter only the players
        filtered_others_detection = [] #Filter ball kids, judges...
        assign_roles_per_frame = []

        #Assign roles
        role_assignments = self.calculate_player_dist_from_court(court_layout,detections_first_frame)

        #Create dictionary to seperate players and other people
        for player_dict in player_detections:
            players = {}
            others = {}

            for track_id in player_dict:
                role = role_assignments.get(track_id, "Unknown")
                if role == "Player":
                    players[track_id] = player_dict[track_id]
                else:
                    others[track_id] = player_dict[track_id]
            
            assign_roles_per_frame.append(role_assignments)
            filtered_others_detection.append(others)
            filtered_players_detection.append(players)
        
        return filtered_players_detection,filtered_others_detection,assign_roles_per_frame
            
    def calculate_player_dist_from_court(self,court_layout,player_dict):
        """
        Calculates the distance of the players from the court lines and assigns roles based on the distance
        Args:
            court_layout (dict): Dictionary of court description
            player_dict (dict): Dictionary of player id and bounding box coordinates
        Returns:
            role_assignemnts (dict): Dictionary of player id and their roles
        """
        role_assignemnts = {}

        top_baseline_length = court_layout["top_baseline"]
        btm_baseline_length = court_layout["bottom_baseline"]
        net_left = court_layout["net_left"]
        net_right = court_layout["net_right"]
        
        for track_id,bbox in player_dict.items():
            #get the middle of the bounding box
            x_middle_bbox = get_foot_position(bbox=bbox)
            x_foot,y_foot = x_middle_bbox

            dist_from_top_baseline = distance_point_to_segment(x_middle_bbox,*top_baseline_length) #* means unpack the tuple
            dist_from_btm_baseline = distance_point_to_segment(x_middle_bbox,*btm_baseline_length)
            dist_from_net_left = measure_dist(x_middle_bbox,net_left)
            dist_from_net_right = measure_dist(x_middle_bbox,net_right)
            net_middle_x = (net_left[0] + net_right[0]) / 2
            print(f"Track ID {track_id}")
            print(f"Foot Position: {x_middle_bbox}")
            print(f"dist_from_top_baseline: {dist_from_top_baseline}")
            print(f"dist_from_btm_baseline: {dist_from_btm_baseline}")
            print(f"dist_from_net_left: {dist_from_net_left}")
            print(f"dist_from_net_right: {dist_from_net_right}")
            print(f"net_mid_x: {net_middle_x}, foot_x: {x_foot}")

            if dist_from_top_baseline < 80 or dist_from_btm_baseline < 80:
                role = "Player"
            elif 0 < dist_from_top_baseline < 250  or  0 < dist_from_btm_baseline < 250:
                role = "Line Judge"
            elif dist_from_net_right < 300:
                role = "Umpire"
            elif dist_from_net_left < 300:
                role = "Ball kid"
            else:
                role = "Unknown"

            role_assignemnts[track_id] = role

        return role_assignemnts
    
    #Detect and track the players in the video
    def detect_frames(self,frames,read_from_stub=False,stub_path=None,court_layout=None):
        """
        Detects and tracks players in the video
        Args:
            frames (list): List of frames as an array
            read_from_stub (bool): If True, put data into a stub. Default = False
            stub_path (str): path to save the output so that we dont have to run the detector again

        Returns:
            player_detection (list): List of dictionaries with player id and bounding box coordinates
        """
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
     
        for frame in frames:
            player_dict = self.detect_frame(frame,court_layout)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(player_detections,f)
        
        return player_detections

    #Detect and track the players in the frame
    def detect_frame(self,frame,court_layout=None):
        """
        Detects and tracks players in the frame
        Args:
            frame (array): Input frame
            court_layout (dict): Dictionary of court description

        Returns:
            player_dict (dict): Dictionary with player id and bounding box coordinates
        """
        results = self.model.track(frame,persist=True)[0] #Runs tracking on the first frame
        id_name_dict = results.names
        player_dict = {} #Dictionary to store player id and bounding box coordinates

        for box in results.boxes:
            track_id = int(box.id.tolist()[0]) #Add track id of each box to the dictionary
            result = box.xyxy.tolist()[0] #Add bounding box coordinates to the dictionary
            object_cls_id = box.cls.tolist()[0] #Add object class id to the dictionary
            object_cls_name = id_name_dict[object_cls_id] #Get the object class name

            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict
    
    def draw_player_bounding_box(self,video_frames,player_detections):
        """
        Draws bounding box around the players in the video
        Args:
            video_frames (list): List of frames as an array
            player_detections (list): List of dictionaries with player id and bounding box coordinates
        Returns:
            output_video_frames (list): List of frames with bounding box around the players
        """
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10)),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
    
    def draw_others_bounding_box(self,video_frames,other_detections,role_assignments):
        """
        Draws bounding box around the other people in the video
        Args:
            video_frames (list): List of frames as an array
            other_detections (list): List of dictionaries with player id and bounding box coordinates
            role_assignments (list): List of dictionaries with player id and their roles
        Returns:
            output_video_frames (list): List of frames with bounding box around the other people
        """
        output_video_frames = []
        for frame, other_dict, roles in zip(video_frames,other_detections,role_assignments):

            #Draw bounding box
            for track_id,bbox in other_dict.items():
                role = roles.get(track_id, "unknown")
                x1,y1,x2,y2 = bbox
                if role == "Umpire":
                    cv.putText(frame, f"{role} ID: {track_id}",(int(bbox[0]),int(bbox[1] -10)),cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                elif role == "Line Judge":
                    cv.putText(frame, f"{role} ID: {track_id}",(int(bbox[0]),int(bbox[1] -10)),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 165), 2)
                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 165), 2)
                else:
                    cv.putText(frame, f"{role} ID: {track_id}",(int(bbox[0]),int(bbox[1] -10)),cv.FONT_HERSHEY_SIMPLEX, 0.9, (128, 0, 128), 2)
                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (128, 0, 128), 2)
            
            output_video_frames.append(frame)

        return output_video_frames
    

        
        
  