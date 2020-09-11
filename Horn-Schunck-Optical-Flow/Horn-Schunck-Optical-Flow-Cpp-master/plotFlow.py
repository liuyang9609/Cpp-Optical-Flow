import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('imgPath', help='Image Path')
    parser.add_argument('uMatrixPath', help='U Matrix Path')
    parser.add_argument('vMatrixPath', help='V Matrix Path')
    parser.add_argument('savePath', help='V Matrix Path')
    args = parser.parse_args()
        
    img = mpimg.imread(args.imgPath)

    # Reading u and v matrices 
    uFile = cv2.FileStorage(args.uMatrixPath, cv2.FILE_STORAGE_READ)
    u = uFile.getNode("u matrix").mat()
    
    vFile = cv2.FileStorage(args.vMatrixPath, cv2.FILE_STORAGE_READ)
    v = vFile.getNode("v matrix").mat()
    
    (size_x, size_y) = u.shape
    delta = 10
    
    x,y = np.meshgrid(np.arange(0, size_y, delta), np.arange(0, size_x, delta))
    u_ds = u[0:size_x:delta, 0:size_y:delta]
    v_ds = v[0:size_x:delta, 0:size_y:delta]
    
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.quiver(x, y, u_ds,v_ds, color="yellow")
    plt.savefig(args.savePath)
    plt.show()
    
if __name__ == "__main__":
    main()
