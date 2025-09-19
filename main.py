import os
from utils.video_helpers import extract_frames  

extract_frames("Couch film of John F. Kennedy assassination.mp4", "couch_film", 10, 20)

for frame in os.listdir("couch_film"):
    # TODO: process each frame
    pass
