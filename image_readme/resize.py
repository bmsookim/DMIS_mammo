import cv2

img_lst = ['mass_0.png', 'mass_1.png']

for img_path in img_lst:
   img = cv2.imread(img_path)
   rsz = cv2.resize(img, (224, 224))
   print(rsz.shape)

   cv2.imwrite(img_path, rsz)
