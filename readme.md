# Data reduction strategy (code)

## Main codes 
- [main.py](main.py) : 대용량의 이미지들에서 feature(color, local_binary_pattern, edge_magnitude)를 추출하여 KMedoids 를 이용해서 
대표 값만 추출하여 10bit(형태는 16bit) yuv420 형태로 저장해주는 코드.
- [random_sample_with_resize.py](random_sample_with_resize.py) : random 하게 down scale 하여 저장하는 코드.

## etc
[custom_kmedoids](custom_kmedoids) : 실제로 사용하지는 않음. 
[read_hdr.py](read_hdr.py) : 24bit 의 영상을 8bit 영상으로 rendering 하는 코드. 

## Sequence
1. hdr 영상들 중 gt 영상만 모두 모아주자. 
   HDR-Eye   
   HDR-Real   
   Code used : get_only_hdr_gt.py   
   SingleHDR_training_data(HDR-Real)  
   SingleHDR_training_data(HDR-Synth)  


2. random_sample_with_size.py 에서 DB 들을 random 하게 resize 해서 저장해준다.


3. re_name_and_move.py 를 이용해서 따로따로 만들어진 DB 들을 하나의 폴더에 합쳐주면서, 두개의 폴더로 나눠준다.



