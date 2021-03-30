# segment_image

### Abstract

- A lib can be used to segment image https://pypi.org/project/segment-image/
- The first version just has k_means algorithm.

### install

- python >= 3.6.5
- install
```
$ pip install segment-image
```


### how to use

```
img = cv2.imread("./sample_imgs/lena.png")
k = 2 # number of segments
k_means = segment_image.Kmeans(img, k)
```

### example

- https://github.com/MyDuan/segment_image/blob/main/examples/example_of_kmeans.py
- results:

![kmeans_re](https://user-images.githubusercontent.com/19246998/113019886-04c81500-91bd-11eb-8075-016c64f5161b.png)

