
q = []
for i in range(int(input())):
    q.append(input())
q = (" ".join(q)).split(" ")
target = input().split(" ")

abcd = ('abcdefghijklmnopqrstuvwxyz')
abcd_list = list(abcd)

x=0
P=list()
C=list()
for i in q:
    x = x+1
    if x % 2 == 1:
        P.append(q[x-1])
    else:
        C.append(q[x-1])

P2 = list()
for i in P:
    word=list()
    for letter in i:
        if abcd.index(letter) >= 1:
            word.append(abcd_list[abcd.index(letter) - 1])
        else:
            word.append('z')
    word = "".join(word)
    P2.append(word)

x=0
dic={}
for i in P2:
    dic[C[x]] = P2[x]
    x=x+1

answer=list()
for i in target:
    answer.append(dic[i])
print(" ".join(answer))