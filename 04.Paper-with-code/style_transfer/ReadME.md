# Style transfer

## 1. Introduction
 paper : https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
 논문에서는 어떤 이미지의 style을 다른 이미지로 전달하는것을 textuer transfer 문제라고 한다.
 기존의 texture transfer에서도 가능했지만, 이는 low-level feature만을 사용해서 문제점이 있었다.
 이 연구에서는 high-level feature에서 high-level semantic feature을 추출하여 사용할 수 있기 때문에 좀 더 의미있는 style transfer이 가능하다고 한다.
 ![image](https://user-images.githubusercontent.com/102507688/185020424-a3dfb0a1-a830-4cec-a1ab-8b9c1ca5f537.png)

## 2.Deep image representations
  * base : normalised된 16개의 convolutional layers, 5개의 pooling layers로 구성된 VGG19 network를 baisis model로 사용, fully connected layer들을 사용하지 않았다.
  
  ### Content representation
   $N_l : 한 layer의 filter 수(channel 수)$ 
   $M_l : feature map의 내적$ 
   $F^l_{i,g} : F \in R^{N_lXM_l}$  
   $p : 원본 이미지$
   $x : 생성된 이미지$   
   $P^l: 원본 이미지 layer l feature map $
   $F^l : 생성된 이미지 layer l feature map$
 
 
 Content Loss : 
 
 ![image](https://user-images.githubusercontent.com/102507688/185025370-7d4d1b91-47ea-4185-830b-a19f23f0683f.png)
 
 
 ![image](https://user-images.githubusercontent.com/102507688/185032861-b5f1ee9b-5f13-4ea6-b130-6598f049b9c1.png)

 network의 higher layer는 이미지의 물체나 배치에 대한 high-level content를 잡아낸다. 그러나 정확한 픽셀값을 기대하기는 어렵다.
 이 연구에서는 higher layer의 feature map을 content representation에 활용한다.
 
 ### Style representation
 * input 이미지의 style representation을 얻으려면, texture information을 잡아내는 feature space를 활용해야한다.
 이 때 활용하는 것이 Gramm matrix이다.
 
 $G^l \in R^{N_lXM_l}$
 
 $G^l_{ij} = \sigma F^l_{ik}F^l_{jk}$
 
 
 * 이 때 G는 layer l에서 vectorized feature map i, j 의 내적을 한 것이다.
 원본 이미지의 Gram matrics와 생성된 이미지의 Gram matrics 간 mean-squared distance를 최소화 하도록 Loss를 구성한다.
 
$$a : style original image$$
$$x : 생성된 image$$
$$A^l : style original image layer l feature map$$
$$F^l : 생성된 image layer l feature map$$

![image](https://user-images.githubusercontent.com/102507688/185036722-2254f37f-1a7f-4de8-ae54-0180f0266af3.png)


![image](https://user-images.githubusercontent.com/102507688/185036763-0e56b1c0-5892-489f-a057-dca43e15273c.png)

