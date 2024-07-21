import cv2
import os
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

image_folder = 'plots'
video_name = 'movie.avi'

files = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = natural_sort(files)
#print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#video = cv2.VideoWriter(video_name,fourcc, 12, (width,height))
video = cv2.VideoWriter(video_name, 0, 12, (width,height))

for image in images:
 #   print("Processing:", os.path.join(image_folder, image))
    video.write(cv2.imread(os.path.join(image_folder, image)))

if video.isOpened():
    print("Video created successfully!")
else:
    print("Error creating video!")


cv2.destroyAllWindows()
video.release()
os.system('ffmpeg -i '+video_name+' '+video_name[:-3]+'mp4')
os.system('rm '+video_name)
