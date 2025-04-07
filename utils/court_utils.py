from math import dist
import numpy as np

def get_court_layout(keypoints):
    """
    Extracts net and baseline positions from flat keypoint list.
    Args:
        keypoints (list): [x1,y1,x2,y2,...,x14,y14]
    Returns:
        dict: Dictionary of court layout info
    """
    #Parse the keypoints into a coordinate pair
    kps = np.array([(int(keypoints[i]), int(keypoints[i+1])) for i in range(0, len(keypoints), 2)]) #kps = [(x1,y1),(x2,y2),...,(x14,y14)]
    
    layout = {
        "top_baseline": (kps[0], kps[1]),
        "bottom_baseline": (kps[2], kps[3]),
        #"net": ((kps[0]-kps[2]),(kps[1]-kps[3])),
        #"left": kps[0],
        #"right": kps[1],
        #"keypoints": kps,      
        "net_left": ((kps[0][0] + kps[2][0]) / 2, (kps[0][1] + kps[2][1]) / 2), #midpoint formula
        "net_right": ((kps[1][0] + kps[3][0]) / 2, (kps[1][1] + kps[3][1]) / 2) #midpoint formula
    }

    return layout