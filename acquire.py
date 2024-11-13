import numpy as np
import matplotlib.pyplot as plt
import cam

h, ia = cam.setup_cam()
# In here we have to find a way to set the amount of frame we want to capture
# or is it just done with a for loop?

try:
    with ia.fetch_buffer() as buffer:
        img = buffer.payload.components[0].data
        img = np.reshape(img, (buffer.payload.components[0].height, buffer.payload.components[0].width))
        print(img)
        print(img.max())
        plt.imshow(img, cmap='gray')
        plt.show()
finally:
    cam.disconnect_cam(h, ia)
