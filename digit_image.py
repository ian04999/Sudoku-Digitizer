import sys
import numpy as np
import cv2
from keras.models import model_from_json
from keras.models import load_model

# load the saved CNN model which created by cnn_model.py
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
 
def output(a):
    sys.stdout.write(str(a))
    
# evaluate loaded model on test data
def identify_number(image):
    image_resize = cv2.resize(image, (28,28))
    image_rr = image_resize.reshape(-1,28,28)
    model_pred = loaded_model.predict(image_rr, verbose = 0)
    prediction = (model_pred > 0.5).astype("int64")

    return np.where(prediction[0] == 1)[0][0]
    
def extract_number(sudoku):
    sudoku = cv2.resize(sudoku, (450,450))

    # split sudoku
    grid = np.zeros([9,9])
    for i in range(9):
        for j in range(9):
            image = sudoku[i*50+3:(i+1)*50-3,j*50+3:(j+1)*50-3]
            if image.sum() > 25000:
                grid[i][j] = identify_number(image)
            else:
                grid[i][j] = 0
    return grid.astype(int)

def display_sudoku(sudoku):
    for i in range(9):
        for j in range(9):
            cell = sudoku[i][j]
            if cell == 0 or isinstance(cell, set):
                output('.')
            else:
                output(cell)
            if (j + 1) % 3 == 0 and j < 8:
                output(' |')

            if j != 8:
                output('  ')
        output('\n')
        if (i + 1) % 3 == 0 and i < 8:
            output("--------+----------+---------\n")
			
image_path = "./process_image.jpg"
img = cv2.imread(image_path,0)


grid = extract_number(img)
print('Sudoku:')
display_sudoku(grid.tolist())

cv2.imshow('sudoku', img)
cv2.waitKey()
