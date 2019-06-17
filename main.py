import cv2
import numpy as np

objectTolerance = 0.5

# Prepare the image we will look in
imageToMatch = cv2.imread("C:\\Users\\Peter\\Documents\\pythoncode\\computervision\\images\\find_waldo.jpeg")
imageToMatchGray = cv2.cvtColor(imageToMatch, cv2.COLOR_BGR2GRAY)

# Prepare the object we will look for
objectToFind = cv2.imread("C:\\Users\\Peter\\Documents\\pythoncode\\computervision\\images\\waldo_head.png", 0)
w, h = objectToFind.shape[::-1]

# Process Image with Object
compiledImageToMatch = cv2.matchTemplate(imageToMatchGray, objectToFind, cv2.TM_CCOEFF_NORMED)
matchedObjects = np.where(compiledImageToMatch >= objectTolerance)

# Draw squares around matched objects
for pt in zip(*matchedObjects[::-1]):
	cv2.rectangle(imageToMatch, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)

# Show the result
cv2.imshow('Detected Objects', imageToMatch)

# These lines make sure window, doesn't close immediately
cv2.waitKey()
cv2.destroyAllWindows()