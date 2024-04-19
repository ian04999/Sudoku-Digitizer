import numpy as np
import operator
import cv2 	

# this function is used to smooth the image by Gaussian Blur and thresholding
def gaussian_thresholding(img, is_dilate):
    # boolean variable that used to check for doing dilate or not
    is_dilate = is_dilate
    # smooth the image with Gaussian blur with a kernal size of 9.
    smooth = cv2.GaussianBlur(np.copy(img), (9, 9), 0)
    
    # using adaptive threshold with 11 nearest neighbour
    threshold = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold = cv2.bitwise_not(threshold, threshold)
    
    if is_dilate:
        # increase the size of the grid's lines.
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        dilate = cv2.dilate(threshold, kernel)
        return dilate
    return threshold

# this function is used to find the contours on the image
def contours_detection(img):
    # finding the contours from the image input 
	contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)  
	polygon = contours[0] 

	# largest value of (x + y) 
	bottom_r, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    # smallest value of (x + y) 
	top_l    , _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    # smallest value (x - y) 
	bottom_l , _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    # largest value (x - y) 
	top_r   , _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    # return the 4 corner points as an array 
	return [polygon[top_l][0], polygon[top_r][0], polygon[bottom_r][0], polygon[bottom_l][0]]

# this function is used to calculate the distance between 2 point 
def distance(p1, p2):
	x_bar = p2[0] - p1[0]
	y_bar = p2[1] - p1[1]
	return np.sqrt((x_bar ** 2) + (y_bar ** 2))

# this function is used to warp the image to showing only the Sudoku grid
def warping(img, corners):
	top_l, top_r, bottom_r, bottom_l = corners[0], corners[1], corners[2], corners[3]
	src = np.array([top_l, top_r, bottom_r, bottom_l], dtype='float32')
	# get the maximum side from the corners 
	side = max([
		distance(top_l, top_r),
		distance(top_l, bottom_l),
        distance(bottom_r, top_r),
		distance(bottom_r, bottom_l)		
	])
	
	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
	# transformation matrix 
	m = cv2.getPerspectiveTransform(src, dst)

	# transform the original image
	return cv2.warpPerspective(img, m, (int(side), int(side)))

# this function is used to determine the top left and bottom right points of the boxes in each inner grids
def inner_grid(img):
	box = []
	side = img.shape[:1]
	side = side[0]/9

	for j in range(9):
		for i in range(9):
			p1 = (i * side, j * side)  
			p2 = ((i + 1) * side, (j + 1) * side) 
			box.append((p1, p2)) # (top left corner, bottom right corner)
	return box

# this function is used to cut a box from the inputed image using the top left and bottom right points
def cut_box(img, box):
	return img[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])]

# this function is used to scale and centre the image onto a new background
def scale_and_centre(img, size, margin=0, background=0):
	h, w = img.shape[:2]

	def centre_pad(length):
		if length % 2 == 0:
			side1 = int((size - length)/2)
			side2 = side1
		else:
			side1 = int((size - length)/2)
			side2 = side1 + 1
		return side1, side2

	def scale(r, x):
		return int(r * x)

	if h > w:
		t_pad = int(margin/2)
		b_pad = t_pad
		ratio = (size - margin)/h
		w, h = scale(ratio, w), scale(ratio, h)
		l_pad, r_pad = centre_pad(w)
	else:
		l_pad = int(margin/2)
		r_pad = l_pad
		ratio = (size - margin)/w
		w, h = scale(ratio, w), scale(ratio, h)
		t_pad, b_pad = centre_pad(h)

	img = cv2.resize(img, (w, h))
	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
	return cv2.resize(img, (size, size))

# this function is used to find the feature (digit) from the inputed image
def find_feature(inp_img, scan_tl=None, scan_br=None):
	img = inp_img.copy()  
	h, w = img.shape[:2]

	max_area = 0
	seed_point = (None, None)

	if scan_tl is None:
		scan_tl = [0, 0]

	if scan_br is None:
		scan_br = [w, h]

	# loop through the image
	for x in range(scan_tl[0], scan_br[0]):
		for y in range(scan_tl[1], scan_br[1]):
			# operate if the box is white 
			if img.item(y, x) == 255 and x < w and y < h:  
				area = cv2.floodFill(img, None, (x, y), 64)
				if area[0] > max_area:  # get the maximum bound area (the grid)
					max_area = area[0]
					seed_point = (x, y)

	# make everythin gray 
	for x in range(w):
		for y in range(h):
			if img.item(y, x) == 255 and x < w and y < h:
				cv2.floodFill(img, None, (x, y), 64)

	mask = np.zeros((h + 2, w + 2), np.uint8) 

	# highlight the main feature
	if all([p is not None for p in seed_point]):
		cv2.floodFill(img, mask, seed_point, 255)

	top, bottom, left, right = h, 0, w, 0

	for x in range(w):
		for y in range(h):
			if img.item(y, x) == 64:  # hide the non main feature
				cv2.floodFill(img, mask, (x, y), 0)

			if img.item(y, x) == 255:
				top = y if y < top else top
				bottom = y if y > bottom else bottom
				left = x if x < left else left
				right = x if x > right else right

	bbox = [[left, top], [right, bottom]]
	return img, np.array(bbox, dtype='float32'), seed_point

# this function is used to get a digit from a Sudoku grid
def extract_digit(img, box, size):
    # get the digit box from the whole grid
	digit = cut_box(img, box)  

	# find the feature (the number) in middle of the box 
	h, w = digit.shape[:2]
	margin = int(np.mean([h, w])/2.5)
	_, bbox, seed = find_feature(digit, [margin, margin], [w - margin, h - margin])
	digit = cut_box(digit, bbox)

	w = bbox[1][0] - bbox[0][0]
	h = bbox[1][1] - bbox[0][1]

	if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
		return scale_and_centre(digit, size, 4)
	else:
		return np.zeros((size, size), np.uint8)

# this function is used to get the digits from the boxes of the grid and build as an array
def get_digits(img, squares, size):
    digits = []
    img = gaussian_thresholding(np.copy(img), False)
    for s in squares:
        digits.append(extract_digit(img, s, size))
    return digits

# this function is used to show the final result of the Sudoku gird image
def show_result_img(digits, colour=255):
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    img = np.concatenate(rows)
    return img


# this function is used to get the result by calling this function
def processing(img_path):
    org_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    smoothed = gaussian_thresholding(org_img, True)
    
    contours = contours_detection(smoothed)
    warped = warping(org_img, contours)
    squares = inner_grid(warped)
    digits = get_digits(warped, squares, 28)
    fin_img = show_result_img(digits)

    return fin_img

if __name__ == '__main__':
    img_path = "./my_sudoku_image.png"
    result = processing(img_path)
    cv2.imwrite("process_image.jpg", result)
    cv2.imshow("result", result)
    cv2.waitKey()
