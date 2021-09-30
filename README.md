# segment_image

### Abstract

- A lib can be used to segment image https://pypi.org/project/segment-image/
- The first version just has k_means algorithm.

### install

- python >= 3.6
- install
```
$ pip install segment-image
```


### how to use 

- `cv2`need to be installed by
`
pip install opencv-python
`

- Kmeans
```
import segment_image
import cv2

img = cv2.imread("./sample_imgs/lena.png")
k = 2 # number of segments
k_means = segment_image.Kmeans(img, k)
iteration = 10
convergence_radius = 1e-6
k_means.run(iteration, convergence_radius)

```

- gvf_snake
```
import segment_image
import cv2

img = cv2.imread("./sample_imgs/star.png")
max_iteration_gvf = 1000
max_iteration_snake = 1000
gvf_snake = segment_image.GVFSnake(img, max_iteration_gvf, max_iteration_snake)
gvf_snake.run(save=True)

```

### example

- https://github.com/MyDuan/segment_image/blob/main/examples/example_of_kmeans.py
- https://github.com/MyDuan/segment_image/blob/main/examples/example_of_gvf_snake.py
- results:
    - kmeans:

    ![kmeans_re](https://user-images.githubusercontent.com/19246998/113019886-04c81500-91bd-11eb-8075-016c64f5161b.png)

    - gvf_snake
    <img width="300" alt="star_re" src="https://user-images.githubusercontent.com/19246998/135444069-82e60a2d-7f99-4266-990e-5c4e99ccf76b.png">

    
