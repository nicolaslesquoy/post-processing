import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import numpy as np

class Calibration:
    path_to_reference = r"/home/nlesquoy/Documents/Cours/EAEP-209/TP MAE 2022/Calibration/"
    path_to_calibration_output = r"/home/nlesquoy/Documents/Cours/post-processing/src/calibration.json"
    path_to_intermediary = r"/home/nlesquoy/Documents/Cours/post-processing/src/intermediary.txt"

    def calibration(path_to_image):
        fig = plt.figure()
        img = np.asarray(Image.open(path_to_image))
        with open(Calibration.path_to_intermediary,"a") as f:
            name = os.path.basename(path_to_image) 
            f.write(name + "\n")
            f.close()
        plt.imshow(img)

        def onclick(event):
            with open(Calibration.path_to_intermediary,"a") as f:
                f.write('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %(event.button, event.x, event.y, event.xdata, event.ydata) + "\n")
                f.close()
            print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %(event.button, event.x, event.y, event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, '.',c='b')
            fig.canvas.draw()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    
    def iterate(path_to_folder):
        for filename in os.listdir(path_to_folder):
            f = os.path.join(path_to_folder, filename)
            # checking if it is a file
            if os.path.isfile(f):
                Calibration.calibration(f)
        return None
    
    def create_reference_file(path_to_calibration_file):
        output_dict = {}
        with open(Calibration.path_to_intermediary,"r") as f:
            data = f.readlines()
            f.close()
        counter = 0
        while counter < len(data):
            line = data[counter]
            if 'jpg' in data[counter]:
                intermediate_list = []
                counter = counter + 1
                while 'button=1' in data[counter]:
                    contents = line.split('')
                    for element in contents:
                        pass

                


# Calibration.iterate(Calibration.path_to_reference)
Calibration.create_reference_file("truc")

# class CalibrationTest:
#     """Class to generate calibraton files [test]"""
#     path_to_reference = r"/home/nlesquoy/Documents/Projets/EAEP-209/TP MAE 2022/Calibration/" # Move to config file

#     def prepare_image(path):
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         img = cv2.imread(path)
#         plt.plot(img)
        
    
# """     def rotate_image(image, angle):
#         image_center = tuple(np.array(image.shape[1::-1]) / 2)
#         rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#         result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#         return result
#          """
# """     def prepare_image(path):
#         for entry in os.scandir(path):
#             filename = os.path.basename(entry)
#             print(filename)
#             if filename in ["0.jpg","1.jpg","2.jpg","3.jpg","4.jpg"]:
#                 img = cv2.imread(CalibrationTest.path_to_reference + "/{0}".format(filename))
#                 rotated = CalibrationTest.rotate_image(img,180)
#                 cv2.imwrite("{0}_r.jpg".format(filename.strip("jpg")),rotated) """
    
# fig = plt.figure()
# ax = fig.add_subplot(111)
# img = np.asarray(Image.open(Calibration.path_to_reference + "/0.jpg"))
# implot = plt.imshow(img)

# def onclick(event):
#     with open(r"/home/nlesquoy/Documents/Cours/post-processing/src/intermediary.txt","a") as f:
#         f.write('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %(event.button, event.x, event.y, event.xdata, event.ydata))
#     print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#           (event.button, event.x, event.y, event.xdata, event.ydata))
#     plt.plot(event.xdata, event.ydata, '.',c='b')
#     fig.canvas.draw()

# cid = fig.canvas.mpl_connect('button_press_event', onclick)
# plt.show()
