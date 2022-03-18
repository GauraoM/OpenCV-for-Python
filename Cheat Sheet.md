#### Reading Image
image = cv2.imread("image")

#### Capturing Video
cap = cv2.VideoCapture(0)

#### Blank image
blank = np.zeros((500,500,3), dtype="uint8") # Blank image

#### Drawing rectangle 
cv.rectangle(image, pt1, pt2, color, thickness)

cv.rectangle(blank, (0,0), (250,250),(0,255,0), thickness=1)

#### Drawing circle 
cv.circle(image, centre, radius, color, thickness)

cv.circle(blank, (250,250),40,(0,255,0), thickness=1)

#### Drawing rectangle (blank, pt1, pt2, color, thickness)
cv.line(blank, (0,0), (250,250),(0,255,0), thickness=1)

#### Putting a text 
cv.putText(image, text, origin,fontface, fontscale, color, thickness)

cv.putText(blank, "Hello",(255,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0),2 )

#### Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#### Creating blur image
cv.GaussainBlur(src, ksize(oddno), sigmaX, dst, sigmaY)

blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)

#### Edge Detection 
cv.canny(image, threshold1, threshold2)

canny = cv.Canny(img, 125, 175)

#### Dilate
cv.dilate(src, kernel, dst=None, anchor=None, iterations=None)

dilated = cv.dilate(canny, (3,3), iterations=2)

#### Erode 
cv.erode(src, kernel, dst=None, anchor=None, iterations=None)

dilated = cv.erode(dilated, (3,3), iterations=2)

#### REsized
cv.resize(images, dst_size, interpolation = cv_INTER_CUBIC)

resized = cv.resize(image, (500,500))

#### Cropping 
cropped = img[500:200, 200:400]

#### Flipping
cv.flip(img, 1-flip horizontal/0-flip vertical/-1-flip both)

flip = cv.flip(img,1)

#### Find contours 
contours, hierarchies = cv.findContours(image, mode, method)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

#### Threshlod (Binaries the image)
ret, thresh = cv.threshold(image, pt1, pt2, thresholdingTechnique)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

#### BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#### BGR to LAB
hsv = cv.cvtColor(img, cv.COLOR_BGR2LAB)

#### Splitting color channels
bgr = cv.split(img)

#### Merging color channels
merged = cv.merge([b,g,r]) 

#### Median Blurring(img, kernel_size)
median = cv.medianBlur(img, 3)

#### bilteral filter
cv.bilateral(img, d, sigmacolor, sigmaspace)

bilteral = cv.bilateralFilter(img, 3, 15, 15)

#### Bitwise AND(Intersection)
bitwise_and = cv.bitwise_and(reactangle,circle)

#### Bitwise OR(Superimpose)
bitwise_or = cv.bitwise_or(reactangle,circle)

#### Bitwise Not(Invertion)
bitwise_not = cv.bitwise_not(rectangle)

#### Calculate Histogram
images : it is the source image of type uint8 or float32 represented as [img].
channels : it is the index of channel for which we calculate histogram. For grayscale image, its value is [0] and color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.
mask : mask image. To find histogram of full image, it is given as “None”.
histSize : this represents our BIN count. For full scale, we pass [256].
ranges : this is our RANGE. Normally, it is [0,256].

cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

gray_hist = cv.calcHist([gray], [0], None, [256], [0,256])

#### Thresholding

##### Simple

threshold, thresh = cv.threshold(source, thresholdValue, maxVal, thresholdingTechnique)

threshold, thresh = cv.threshold(source, thresholdValue, maxVal, cv.THRESH_BINARY)

##### Inverse
threshold, thresh = cv.threshold(source, thresholdValue, maxVal, thresholdingTechnique)

threshold, thresh_inv = cv.threshold(source, thresholdValue, maxVal, cv.THRESH_BINARY_INV)

##### Adaptive
cv2.adaptiveThreshold(source, maxVal, adaptiveMethod, thresholdType, blocksize, constant)

adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)

#### Gradients

##### Laplacian
laplacian = cv.Laplacian(src, ddepth)

lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))

##### Sobel

Sobel(src, dst, ddepth, dx, dy, CV_SCHARR, scale, delta, borderType).

sobelx = cv.Sobel(img,cv.CV_64F,1,0) 
sobely = cv.Sobel(img,cv.CV_64F,0,1)
