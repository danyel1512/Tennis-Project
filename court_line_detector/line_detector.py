import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import cv2 as cv

class LineDetector:
    def __init__(self,model_path):
        """
        Create NN based on resnet50 architecture
        Args:
            model_path (str): Model Path to load state_dict of pre-trained model
        """
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features,
                                  out_features=14*2)
        self.model.load_state_dict(torch.load(model_path,map_location="cpu"))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Offical normalisation values for ImageNet trained models
        ])

    def predict(self,image):
        """
        Predict the first image/frame only since its a still video
        Args:
            image (any): 1 frame of the video
        Returns:
            keypoints (list): Returns list of keypoints of the image
        """
        img_rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0)

        #Make a prediction to predict the keypoints of the court
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        #Get the keypoints of the image
        keypoints = predictions.squeeze().cpu().numpy()
        original_h,original_w,_ = img_rgb.shape

        #Scale keypoints based on the original image size
        keypoints[::2] *= original_w/225.0
        keypoints[1::2] *= original_h/225.0

        return keypoints
    
    def draw_keypoints(self,image,keypoints):
        """
        Draw keypoints on a single frame
        Args:
            image (any): Image of the video
            keypoints (list): List of keypoints of the court in the image
        Returns:
            image: Returns image with the keypoints annotated on the image
        """
        #Loop over each keypoint
        for i in range(0,len(keypoints),2):
            x,y = int(keypoints[i]),int(keypoints[i+1])
            cv.circle(image,(x,y),5,(255,0,0),-1)
        
        return image
    
    def draw_keypoints_on_video(self,video_frames,keypoints):
        """
        Draw keypoints on the whole video
        Args:
            video_frame (any): Video to draw the keypoints
            keypoints (list): Keypints of the video
        Returns:
            output_video_frames: Returns the video with the keypoints annotated
        """
        output_video_frames = []

        for frame in video_frames:
            frame = self.draw_keypoints(frame,keypoints)
            output_video_frames.append(frame)
        
        return output_video_frames

            