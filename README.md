Installation:
	All the commands below may have different based on the OS you are using.
	
	Language: python3.10
	
	Please make sure you have installed following Libraries in your python3.10 version:
		pip3.10 install tensorflow
		pip3.10 install opencv-python
		pip3.10 install keras
		pip3.10 install numpy
	
	Note: When installing tensorflow with Windows, you may get the 'longpath error'. 
	To solve this error, please follow below steps: 
		  1. turn on your Registry Editor
		  2. go to Computer\HKEY_LOOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
		  3. Select (double click) LongPathEnabled
		  4. change the value from 0 to 1
		  (you may need to restart you PC)

Instruction:
	1. To run the code of this project, please open a terminal/command line in the same directory with this file.
	2. Enter the command: python3.10 cnn_model.py
		2.1: The files model.json and model.h5 will saved in the same directory
	3. Enter the command: python3.10 cv_image.py
		3.1 Make sure the input Sudoku image (my_sudoku_image.png) is in the same directory
		3.2 The image process_image.jpg will saved in the same directory
	4. Enter the command: python3.10 digit_image.py
		4.1 Make sure the model.json, model.h5, and process_image.jpg are in the same directory
