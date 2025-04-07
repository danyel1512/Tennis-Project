import cv2 as cv

#Function to read the video
def read_video(video_path):
    """
    Reads MP4 video file and returns frames as a list
    Args:
        video_path (str): Path to video
    Returns:
        frames (list): List of frames as an array
    """
    frames = []
    cap = cv.VideoCapture(video_path)

    if(cap.isOpened()==False):
        print("Cannot open video file")
    
    while True:
        #Capture frame by frame
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    #Release the video capture object
    cap.release()
        
    return frames

#Function to save video
def save_video(output_video_frames,output_video_path):
    """
    Saves frames as a video fie
    Args:
        output_video_path (str): Path to save video
        output_video_frames (list): List of frames as an array
    Returns:
        None
    """
    #Get dimensions of the frames
    frame_height,frame_width,_ = output_video_frames[0].shape

    #Define coedc and create video writer object
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(output_video_path,fourcc,30,(frame_width,frame_height))
    for frame in output_video_frames:
        out.write(frame)

    out.release()

