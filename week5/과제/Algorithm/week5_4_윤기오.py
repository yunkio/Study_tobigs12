from math import sqrt
N = int(input())
mylist = [sum([(input().replace("-1", "-")).split(" ") for i in range(N)], [])]

def devide(mylist) : 
    devided = []
    for i in range(3):
        for j in range(3):
            tem=[]
            for num in range(len(mylist)):
                if i*len(mylist)//3 <= num < (i+1)*len(mylist)//3:
                    if j*int(sqrt(len(mylist))/3) <= num%int(sqrt(len(mylist))) < (j+1)*int(sqrt(len(mylist))/3):
                        tem.extend(mylist[num])
            devided.append(tem)
    return(devided)

s,o,t = 0,0,0
for i in mylist:
    if (i.count('1') == len(i)) : t += 1
    elif (i.count('0') == len(i)) : o += 1
    elif (i.count('-') == len(i)) : s += 1
    else : 
        mylist.extend(devide(i))
print(s)
print(o)
print(t)