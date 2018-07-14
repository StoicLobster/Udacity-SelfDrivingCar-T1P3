import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

images = []
steer_data = []
folder_list = []
data_path = '..\..\Training_Data\\'
#folder_list.append('Full_Loop_CCW\\')
#folder_list.append('Full_Loop_CW\\')
folder_list.append('Full_Loop_Fast_CCW\\')
folder_list.append('Full_Loop_Slow_CCW\\')
folder_list.append('Full_Loop_Fast_CW\\')
folder_list.append('Full_Loop_Slow_CW\\')
folder_list.append('Target_Areas\\')
folder_list.append('TargetAreas2\\')
folder_list.append('Recover_Left_CCW\\')
folder_list.append('Recover_Left_CW\\')
folder_list.append('Recover_Right_CCW\\')
folder_list.append('Recover_Right_CW\\')
for folder in folder_list:
	path = data_path + folder
	lines = []
	print('Reading CSV Lines for Folder: ' + path)
	with open(path + 'driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	print('Reading Images for Folder: ' + path)
	for line in lines:
		#Center Image
		source_path = line[0]
		filename = source_path.split('\\')[-1]		
		current_path = path + 'IMG\\' + filename
		img = cv2.imread(current_path)
		if img is not None:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
			images.append(img)
			steer = float(line[3])
			steer_data.append(steer)
		else:
			print(current_path + ' Does Not Exist')
		#Left Image
		source_path = line[1]
		filename = source_path.split('\\')[-1]		
		current_path = path + 'IMG\\' + filename
		img = cv2.imread(current_path)
		if img is not None:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
			images.append(img)
			steer = float(line[3]) + 0.3
			steer_data.append(steer)
		else:
			print(current_path + ' Does Not Exist')
		#Right Image
		source_path = line[2]
		filename = source_path.split('\\')[-1]		
		current_path = path + 'IMG\\' + filename
		img = cv2.imread(current_path)
		if img is not None:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
			images.append(img)
			steer = float(line[3]) - 0.3
			steer_data.append(steer)
		else:
			print(current_path + ' Does Not Exist')

X_Train_Raw = np.array(images)
y_Train_Raw = np.array(steer_data)
print(X_Train_Raw.shape)
# plt_idx = 0
# plt.figure(plt_idx)
# plt.imshow(images[plt_idx])
# plt_idx = 1021
# plt.figure(plt_idx)
# plt.imshow(images[plt_idx])
# plt_idx = 7211
# plt.figure(plt_idx)
# plt.imshow(images[plt_idx])
# plt_idx = 9234
# plt.figure(plt_idx)
# plt.imshow(images[plt_idx])
# plt_idx = 11853
# plt.figure(plt_idx)
# plt.imshow(images[plt_idx])
# plt_idx = 15644
# plt.figure(plt_idx)
# plt.imshow(images[plt_idx])
# plt_idx = 18356
# plt.figure(plt_idx)
# plt.imshow(images[plt_idx])
# plt.show()
print('Saving X_Train...')
np.save('X_Train_Raw',X_Train_Raw)
print('Saving y_Train...')
np.save('y_Train_Raw',y_Train_Raw)