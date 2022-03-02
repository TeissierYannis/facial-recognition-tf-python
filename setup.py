import cv2
import os
import uuid

# import tensorflow dependencies
import tensorflow as tf

# Avoid OOM Errors by settings the GPU memory

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Create directories
if not os.path.exists(POS_PATH):
    os.makedirs(POS_PATH)
if not os.path.exists(NEG_PATH):
    os.makedirs(NEG_PATH)
if not os.path.exists(ANC_PATH):
    os.makedirs(ANC_PATH)

# http:/vis-www.cs.umass.edu/lfw/lfw.tgz
if not os.path.exists('lfw'):
    os.system('wget http://vis-www.cs.umass.edu/lfw/lfw.tgz')
    os.system('tar -xvf lfw.tgz')
    os.system('rm lfw.tgz')

# Uncompress (tar -xf lfw.tgz)
# Move the images to the data folder data/negative

for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)

# Collect positive and Anchor images
# Enable a connection with the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Get size of the frame
    height, width, channels = frame.shape
    # Crop the frame to a center square
    frame = frame[int(height / 2) - 280:int(height / 2) + 280, int(width / 2) - 280:int(width / 2) + 280]
    # resize the frame to a 250x250 image
    frame = cv2.resize(frame, (250, 250))

    # Collect anchors
    if cv2.waitKey(1) & 0xFF == ord('a'):
        # Create the unique file path
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid4()))
        # Write an anchor image
        cv2.imwrite(imgname, frame)

    # Collect positive images
    if cv2.waitKey(1) & 0xFF == ord('p'):
        # Create the unique file path
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid4()))
        # Write an positive image
        cv2.imwrite(imgname, frame)

    # Display the frame
    cv2.imshow('Image collection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
