from Apartment import Apartment
from Vile import Vile

f = open("Apartment.txt", 'r')
N=int(f.readline())
apartments=[]
for i in range(N):
    mylist = (f.readline().split(' '))
    a = Apartment(mylist[0], int(mylist[1]), int(mylist[2].strip()))
    apartments.append(a) ## class append..?
f.close()

# 클래스 리스트 정렬
max_index=0
# magic method가 작동하는 것을 직관적으로 파악해보기 위해 for문을 이용하여 최댓값을 찾아보세요!
for j in range(N-1):
    if apartments[j] > apartments[max_index]:
        max_index = j
# 혹시 같은 인덱스가 있으면 같이 출력한다.
print(apartments[max_index])

for j in range(N):
    if apartments[j] == apartments[max_index]:
        print(apartments[j])

##### Vile
import numpy as np

f = open("Vile.txt", 'r')
N=int(f.readline())
viles=[]
for i in range(N):
    mylist = (f.readline().split(' '))
    a = Vile(mylist[0], int(mylist[1]), int(mylist[2]), mylist[3].strip())
    viles.append(a)
f.close()

# Vile에 대한 최대평수 출력은 max를 활용해보세요~!
size = [viles[i].size_vile for i in range(len(viles))]
print(viles[np.argmax(size)])





