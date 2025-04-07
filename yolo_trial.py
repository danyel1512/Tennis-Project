from ultralytics import YOLO

#Load YOLOv8 model
model_tennisball = YOLO(r"C:\Users\danye\OneDrive\Documents\Personal Projects\Tennis match realtime analysis\training\tennis-ball-detection-1\yolov8_tennisball_best.pt")
model = YOLO("yolov8m")

#make sample prediction
result = model.track(r"C:\Users\danye\OneDrive\Documents\Personal Projects\Tennis match realtime analysis\input_images\input_video.mp4",
                       save = True,
                       project = r"C:\Users\danye\OneDrive\Documents\Personal Projects\Tennis match realtime analysis\runs\detect")
#print(result)

#Show bounding box results
# for box in result[0].boxes:
#     print(box)

print(result.names)