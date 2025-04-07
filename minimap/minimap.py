import cv2 as cv
import numpy as np
import sys
import constants
from utils import (convert_pixels_to_meters,
                   convert_meters_to_pixels,
                   get_foot_position,
                   get_closest_keypoint_index,
                   get_height_of_bbox,
                   measure_xy_dist,
                   centre_of_bbox,
                   measure_dist)

class DrawMinimap():
    """
    Draw Minimap on the video
    """
    def __init__(self,frame):
        """
        Initialize the minimap
        Args:
            frame (array): Input frame
        """
        self.minimap_w = 250
        self.minimap_h = 600
        self.buffer = 50
        self.padding = 20

        self.background_position(frame)
        self.minimap_position()
        self.draw_court_keypoints()
        self.define_court_lines()
    
    def background_position(self,frame):
        """
        Get Position of the minimap background
        """
        frame = frame.copy()

        self.x_backgnd_end_pos = frame.shape[1] - self.buffer
        self.y_backgnd_end_pos = self.buffer + self.minimap_h
        self.x_backgnd_start_pos = self.x_backgnd_end_pos - self.minimap_w
        self.y_backgnd_start_pos = self.y_backgnd_end_pos - self.minimap_h

    def minimap_position(self):
        """
        Get Position of the minimap
        """
        self.minimap_x_end_pos = self.x_backgnd_end_pos - self.padding
        self.minimap_y_end_pos = self.y_backgnd_end_pos - self.padding
        self.minimap_x_start_pos = self.x_backgnd_start_pos + self.padding
        self.minimap_y_start_pos = self.y_backgnd_start_pos + self.padding
        self.court_width = self.minimap_x_end_pos - self.minimap_x_start_pos
        
    def convert_meters_to_pixels(self,meters):
        return convert_meters_to_pixels(meters,
                                        constants.DOUBLES_LINE_WIDTH,
                                        self.minimap_w)
    
    def draw_court_keypoints(self):
        """
        Draw court keypoints on the minimap
        """
        #Create a list of 28 0's
        draw_keypoints = [0]*28

        #Define points on the court
        ##Doubles line (top left side) (x,y)
        draw_keypoints[0], draw_keypoints[1] = self.minimap_x_start_pos,self.minimap_y_start_pos

        ##Doubles line (top right side) (x,y)
        draw_keypoints[2], draw_keypoints[3] = self.minimap_x_end_pos,self.minimap_y_start_pos

        ##Doubles line (bottom left side)
        draw_keypoints[4] = self.minimap_x_start_pos
        draw_keypoints[5] = self.minimap_y_start_pos + self.convert_meters_to_pixels(constants.HALF_COURT_LENGTH*2)

        ##Doubles line (bottom right side)
        draw_keypoints[6] = draw_keypoints[0] + self.court_width
        draw_keypoints[7] = draw_keypoints[5]

        ##Singles line (top left side)
        draw_keypoints[8] = draw_keypoints[0] + self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_WIDTH)
        draw_keypoints[9] = draw_keypoints[1]

        ##Singles line (bottom left side)
        draw_keypoints[10] = draw_keypoints[4] + self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_WIDTH)
        draw_keypoints[11] = draw_keypoints[5]

        ##Singles line (top right side)
        draw_keypoints[12] = draw_keypoints[2] - self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_WIDTH)
        draw_keypoints[13] = draw_keypoints[3]

        ##Singles line (bottom right side)
        draw_keypoints[14] = draw_keypoints[6] - self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_WIDTH)
        draw_keypoints[15] = draw_keypoints[7]

        #Service box (top left side)
        draw_keypoints[16] = draw_keypoints[8]
        draw_keypoints[17] = draw_keypoints[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_WIDTH)

        #Service box (top right side)
        draw_keypoints[18] = draw_keypoints[12]
        draw_keypoints[19] = draw_keypoints[13] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_WIDTH)

        #Service box (bottom left side)
        draw_keypoints[20] = draw_keypoints[10]
        draw_keypoints[21] = draw_keypoints[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_WIDTH)
        
        ##Service box (bottom right side)
        draw_keypoints[22] = draw_keypoints[14]
        draw_keypoints[23] = draw_keypoints[15] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_WIDTH)

        ##Service box (top middle)
        draw_keypoints[24] = (draw_keypoints[16] + draw_keypoints[18])/2
        draw_keypoints[25] = draw_keypoints[17]

        ##Service box (bottom middle)
        draw_keypoints[26] = (draw_keypoints[20] + draw_keypoints[22])/2
        draw_keypoints[27] = draw_keypoints[21]

        self.draw_keypoints = draw_keypoints
   
    def define_court_lines(self):
        self.lines = [
           (0,1),
           (1,3),
           (3,2),
           (2,0),
           (4,5),
           (6,7),
           (8,9),
           (10,11),
           (12,13)
        ]

    def draw_background(self,frame):
        """
        Draw minimap background
        Args:
            frame (array): Input frame
        Returns:
            transparent_backgnd (array): Returns the frame with minimap background
        """
        frame_copy = frame.copy()
        background = np.zeros_like(frame,np.uint8)

        #Draw background
        cv.rectangle(background,
                     (self.x_backgnd_start_pos,self.y_backgnd_start_pos),
                     (self.x_backgnd_end_pos,self.y_backgnd_end_pos),
                     (255,255,255),
                     -1)
        
        alpha = 0.2 #Transparency factor

        transparent_backgnd = cv.addWeighted(frame_copy,1-alpha,background,alpha,0)
        #transparent_backgnd = cv.cvtColor(transparent_backgnd,cv.COLOR_BGR2RGB)

        return transparent_backgnd
    
    def draw_court_lines(self,frame):
        """
        Draw court lines on the minimap
        Args:
            frame (array): Input frame
        Returns:
            frame (array): Returns the frame with court lines
        """
        for i in range(0,len(self.draw_keypoints),2):
            Cx = self.draw_keypoints[i]
            Cy = self.draw_keypoints[i+1]
            cv.circle(frame,(int(Cx),int(Cy)),5,(255,0,0),-1)

        for i, j in self.lines:
            start_point = (int(self.draw_keypoints[i*2]),int(self.draw_keypoints[i*2+1]))
            end_point = (int(self.draw_keypoints[j*2]),int(self.draw_keypoints[j*2+1]))
            cv.line(frame,start_point,end_point,(0,0,0),2)

        #Draw net
        net_start_point = (self.draw_keypoints[0],int(self.draw_keypoints[1]+self.convert_meters_to_pixels(constants.HALF_COURT_LENGTH)))
        net_end_point = (self.draw_keypoints[2],int(self.draw_keypoints[3]+self.convert_meters_to_pixels(constants.HALF_COURT_LENGTH)))
        cv.line(frame,net_start_point,net_end_point,(255,0,0),4)

        return frame
    
    def draw_ball_position(self,frame,ball_position,court_layout):
        """
        Draw ball position on the minimap
        Args:
            frame (array): Input frame
            ball_position (list): List of ball bounding box coordinates
        Returns:
            frame (array): Returns the frame with ball position
        """
        
        #Get centre of the ball
        x1,y1,x2,y2 = ball_position
        cX = (x1+x2)/2
        cY = (y1+y2)/2

        x_offset = cX - court_layout["left"]
        y_offset = max(0,cY - court_layout["top_baseline"])

        ref_w_x = court_layout["right"] - court_layout["left"]
        ref_h_x = court_layout["bottom_baseline"] - court_layout["top_baseline"]

        x_meters = convert_pixels_to_meters(x_offset,constants.DOUBLES_LINE_WIDTH,ref_w_x)
        y_meters = convert_pixels_to_meters(y_offset,constants.HALF_COURT_LENGTH*2,ref_h_x)

        x_minimap = self.minimap_x_start_pos + self.convert_meters_to_pixels(x_meters)
        y_minimap = self.minimap_y_start_pos + self.convert_meters_to_pixels(y_meters)

        # cv.circle(frame,(int(x_minimap),int(y_minimap)),5,(0,255,0),-1)
        # cv.circle(frame,(self.minimap_x_start_pos+20,self.minimap_y_start_pos+20),5,(0,255,0),-1)

        return frame

    def draw_minimap(self,frames):
        """
        Draw minimap on the video
        Args:
            frames (list): List of frames as an array
        Returns:
            output_frames (list): Returns the list of frames with minimap
        """
        output_frames = []

        for i, frame in enumerate(frames):
            frame = self.draw_background(frame)
            frame = self.draw_court_lines(frame)
            output_frames.append(frame)

        return output_frames
    
    def get_minimap_coor(self,object_position,closest_keypoint,closest_keypoint_idx,player_height_pixels,player_height_meters):
        """
        Get minimap coordinates of the player
        Args:
            object_position (tuple): Position of the player
            closest_keypoint (tuple): Closest keypoint to the player
            closest_keypoint_idx (int): Index of the closest keypoint
            player_height_pixels (int): Height of the player in pixels
            player_height_meters (int): Height of the player in meters
        Returns:
            minimap_play_pos (tuple): Returns the minimap coordinates of the player
        """
        #Distance from player to the closest keypoint on court
        dist_from_keypoint_x_pixels ,dist_from_keypoint_y_pixels = measure_xy_dist(object_position,closest_keypoint)
        
        #Convert pixels to m
        dist_from_keypoint_x_meters  = convert_pixels_to_meters(dist_from_keypoint_x_pixels,
                                                                player_height_meters,
                                                                player_height_pixels)
        
        dist_from_keypoint_y_meters  = convert_pixels_to_meters(dist_from_keypoint_y_pixels,
                                                                player_height_meters,
                                                                player_height_pixels)
        
        #Convert to minicourt coor
        x_minimap_dist_pixels = self.convert_meters_to_pixels(dist_from_keypoint_x_meters)
        y_minimap_dist_pixels = self.convert_meters_to_pixels(dist_from_keypoint_y_meters)
        closest_minimap_keypoint = (self.draw_keypoints[closest_keypoint_idx*2],self.draw_keypoints[closest_keypoint_idx*2+1]) #(x,y)

        minimap_play_pos = (closest_minimap_keypoint[0]+x_minimap_dist_pixels,
                            closest_minimap_keypoint[1]+y_minimap_dist_pixels)
        
        return minimap_play_pos

    def convert_bbox_to_minimap_coor(self,player_boxes,ball_boxes,original_court_keypoints):
        """
        Convert player and ball bounding box to minimap coordinates
        Args:
            player_boxes (list): List of dictionaries with player id and bounding box coordinates
            ball_boxes (list): List of dictionaries with ball id and bounding box coordinates
            original_court_keypoints (list): List of court keypoints
        Returns:
            output_player_bbox (list): Returns the list of dictionaries with player id and minimap coordinates
            output_ball_bbox (list): Returns the list of dictionaries with ball id and minimap coordinates
        """
        player_height = {
            1: constants.PLAYER_1_HEIGHT,
            2: constants.PLAYER_2_HEIGHT}

        output_player_bbox = []
        output_ball_bbox = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = centre_of_bbox(ball_box)

            #Measures the dist btwn center of the player bounding box and the ball position and return the key of the minimun distance calculated
            closest_player_id_to_ball = min(player_bbox.keys(),key=lambda x:measure_dist(ball_position,centre_of_bbox(player_bbox[x])))

            output_player_bbox_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                #Get closest court keypoints in pixels
                closest_keypoint_index = get_closest_keypoint_index(foot_position,original_court_keypoints,[0,2,12,13])
                closest_keypoint = (original_court_keypoints[closest_keypoint_index*2],original_court_keypoints[closest_keypoint_index*2+1])

                #Get player height in pixels
                frame_idx_min = max(0,frame_num-20)
                frame_idx_max = min(len(player_boxes),frame_num+50)
                bbox_height_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range(frame_idx_min,frame_idx_max)]
                max_player_height_pixels = max(bbox_height_in_pixels)
                minimap_player_pos = self.get_minimap_coor(foot_position,
                                                           closest_keypoint,
                                                           closest_keypoint_index,
                                                           max_player_height_pixels,
                                                           player_height[player_id])
                
                output_player_bbox_dict[player_id] = minimap_player_pos

                if closest_player_id_to_ball == player_id:
                    #Get closest court keypoints in pixels
                    closest_keypoint_index = get_closest_keypoint_index(ball_position,original_court_keypoints,[0,2,12,13])
                    closest_keypoint = (original_court_keypoints[closest_keypoint_index*2],original_court_keypoints[closest_keypoint_index*2+1])

                    minimap_ball_pos = self.get_minimap_coor(ball_position,
                                                               closest_keypoint,
                                                               closest_keypoint_index,
                                                               max_player_height_pixels,
                                                               player_height[player_id])
                    
                    output_ball_bbox.append({1:minimap_ball_pos})
            output_player_bbox.append(output_player_bbox_dict)

        print(f"{output_player_bbox_dict}")
        return output_player_bbox,output_ball_bbox

    def draw_points_on_minimap(self,frames,positions,colour):
        """
        Draw points on the minimap
        Args:
            frames (list): List of frames as an array
            positions (list): List of positions
            Colour (tuple); Colour of the points
        Returns:
            frames (list): Returns the list of frames with points on the minimap
            """
        for frame_num,frame in enumerate(frames):
            if frame_num >= len(positions):
                continue
            for _, position in positions[frame_num].items():
                cX,cY = position
                cX = int(cX)
                cY = int(cY)
                # Draw the points on the frame
                cv.circle(frame,(cX,cY),5,colour,-1)

        return frames


                    




