from matplotlib import pyplot as plt
import cv2

"""
A set of utility functions handling opencv image formats and color-spaces
"""

def patch(img, kwargs):
  """
  A patching function that:
    - Defaults cmap to grayscale if detects images with only 1 channel.
    - Defaults cmap to rgb if detects images with 3 channels and no cmap defined.
    - Converts opencv default BGR format to RGB if detects images with 3 channels.
    - Converts opencv HSV to RGB
  """

  grayscale = {'cmap':'gray', 'vmin':0, 'vmax':255}
  
  cmap_patched = kwargs.copy()
  if len(img.shape) == 2:
    # num channels == 1
    # Defaulting cmap to grayscale
    if 'cmap' not in kwargs:
      cmap_patched.update(grayscale)

  img_patched = img
  if len(img.shape) == 3:
    if 'cmap' not in kwargs:
      # Changing BGR opencv format to RGB
      if img.shape[2] == 4:
        img_patched = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
      else:
        img_patched = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # matplotlib expects hsv in [0, 1] range, simply convert opencv HSV to RGB format
    if 'cmap' in kwargs and kwargs['cmap'] == 'hsv':
      img_patched = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

  return img_patched, cmap_patched


def imshow(img, **kwargs):
  """
  imshow wrapper for matplotlib.pyplot.imshow
  """
  patched_img , patched_cmap = patch(img, kwargs)
  plt.imshow(patched_img, **patched_cmap)
  plt.axis('off')

def show_images(images, titles=None, **kwargs):
  num_images = len(images)
  fig, axs = plt.subplots(1, num_images, figsize=(12, 6))
  if titles is None:
    titles = [None for _ in images]
  for ax, img, title in zip(axs, images, titles):

    patched_img , patched_cmap = patch(img, kwargs)
    ax.imshow(patched_img, **patched_cmap)
    ax.axis('off')
    ax.set_title(title)

def plot_transform(r, s, label=None, title=None, fig=None):
  if fig is None:
    plt.figure(figsize=(5, 5))
  if not isinstance(s, list):
    ss = [s]
  else:
    ss = s

  legend = True
  if label is None:
    legend = False
    ls = [None] * len(ss)
  else:
    if not isinstance(label, list):
      ls = [label] * len(ss)
    else:
      ls = label

  for s, lbl in zip(ss, ls):
    plt.plot(r, s, label=lbl)

  plt.grid()
  plt.xlabel("r")
  plt.ylabel("s")
  plt.title(title)
  if legend:
    plt.legend()
  plt.ylim(0, 256)
  plt.xlim(0, 256)
