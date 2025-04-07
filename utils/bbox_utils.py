import numpy as np

def centre_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    cX = int((x1+x2)/2)
    cY = int((y1+y2)/2)

    return (cX,cY)

#Based on euclidean distance
def measure_dist(p1,p2):
    return((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)**0.5

def get_foot_position(bbox):
    """
    Get the foot position (center of the length of the bbox) of the player based on their bounding box
    Args:
        bbox (tuple): Tuple containing the bounding box coordinates (x1, y1, x2, y2).
    Returns:
        tuple: Tuple containing the foot position (x, y).
    """
    x1,y1,x2,y2 = bbox

    return(int((x1+x2)/2),y2)

def get_middle_height_of_box(bbox):
    x1,y1,x2,y2 = bbox

    return (x2,int((y1+y2)/2))

def get_closest_keypoint_index(point,keypoints,keypoint_index):
    """
    Get the index of the closest keypoint to a given point.
    Args:
        point (tuple): Tuple containing the coordinates of the point (x, y).
        keypoints (list): List of keypoints coordinates.
        keypoint_index (list): List of indices of keypoints.
    Returns:
        keypoint_idx (int): Index of the closest keypoint.
    """
    closest_dist = float('inf')
    keypoint_id = keypoint_index[0]
    for keypoint_idx in keypoint_index:
        keypoint = keypoints[keypoint_idx*2],keypoints[keypoint_idx*2+1]
        distance = abs(point[1]-keypoint[1])

        if distance < closest_dist:
            closest_dist = distance
            keypoint_id = keypoint_idx

    return keypoint_id

def get_height_of_bbox(bbox):
    """
    Gets the height of the bounding box of the player
    Returns:
        int: Height of the bounding box of the player
    """
    return bbox[3] - bbox[1] #max y - min y

def measure_xy_dist(p1,p2):
    """
    Gets the distance between 2 points
    Returns:
        int: x and y distance between 2 points
    """
    return abs(p2[0]-p1[0]),abs(p2[1]-p1[1]) 

def centre_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))

def distance_point_to_segment(p0, p1, p2):
    """
    Calculates the distance from point p0 to the line segment (p1-p2).
    If the perpendicular projection is outside the segment, return distance to nearest endpoint.
    """

    # Convert to numpy arrays
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)

    line_vec = p2 - p1
    point_vec = p0 - p1

    line_len_squared = np.dot(line_vec, line_vec)

    if line_len_squared == 0:
        return np.linalg.norm(p0 - p1)  # p1 and p2 are the same point

    # Project point onto the line (scalar projection)
    t = np.dot(point_vec, line_vec) / line_len_squared

    if t < 0:
        closest = p1
    elif t > 1:
        closest = p2
    else:
        closest = p1 + t * line_vec

    return np.linalg.norm(p0 - closest)


