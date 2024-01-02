import imageio
import pathlib
import datetime
from PIL import Image
import os
from numpy import asarray

path = os.path.dirname(os.path.abspath(__file__))+'/'
path_save = path
path_load = path + "Test_1/"
scene = "MORMultiGaitRobotScene"

initial_count = 0
for path in pathlib.Path(path_load).iterdir():
    if path.is_file():
        initial_count += 1

video = imageio.get_writer(path_save
                             + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                             + ".mp4", format='mp4', mode='I', fps=100)

for i in range(1, initial_count+1):
    str_num = "0"*(8-len(str(i))) + str(i)
    file = path_load+ scene + "_" + str_num +".png"
    image = asarray(Image.open(file))
    video.append_data(image)
