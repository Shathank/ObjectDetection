import cv2
from matplotlib import pyplot as plt

# Opening image
img = cv2.imread("image.jpg")


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

stop_data = cv2.CascadeClassifier('stop_data.xml')

found = stop_data.detectMultiScale(img_gray,
                                   minSize=(20, 20))
# no sign
amount_found = len(found)

if amount_found != 0:

    # sign in the image
    for (x, y, width, height) in found:
        # We draw a green rectangle around
        cv2.rectangle(img_rgb, (x, y),
                      (x + height, y + width),
                      (0, 255, 0), 5)

# Creates the environment of
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()