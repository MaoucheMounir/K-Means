import matplotlib . pyplot as pl
from KMeans import *


im = plt.imread("logo_cropped2.png ")[: ,: ,:3] #on garde que les
#3 premieres composantes , la transparence est inutile
im_h, im_l , _ = im.shape

pixels = im.reshape((im_h*im_l ,3)) #transformation en matrice n*3, n nombre de pixels
imnew = pixels.reshape((im_h,im_l ,3)) #transformation inverse

km = KMeans(2, pixels)
km.fit()

pixels2 = []
print(pixels.shape)
for i, point in enumerate(pixels):
    pixels2.append(km.get_nouvelle_couleur_du_point(point))
print(pixels2)
pixels2 = np.array(pixels2)
print(pixels2)

imnew = pixels2.reshape((im_h,im_l ,3)) #transformation inverse
plt.imshow(imnew)

#print(pixels2)

