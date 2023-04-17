from data_prep import *
import numpy as np
from PIL import Image

if __name__ == '__main__':
    a = np.zeros((2000, 2000, 3), dtype=np.uint8)
    im = Image.fromarray(a, "RGB")
    im.show()


    # imgs, counts, masks = h5_to_sep_lists()
    # train_dl, test_dl = get_loaders(imgs, counts, masks)
    #
    # # aug_example_iter = iter(train_dl)
    # # aug_images, _, aug_masks = next(aug_example_iter)
    # #
    # f, ax = plt.subplots(2, 4)
    #
    # for i in range(4):
    #     ax[0, i].imshow(np.swapaxes(imgs[i], 0, 2))
    #     ax[1, i].imshow(masks[i], cmap='gray')
    #
    # f.set_figheight(5)
    # f.set_figwidth(15)
    # plt.show()
