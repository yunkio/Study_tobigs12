num = int(input())
q = (" ".join(input().split('\n'))).split(" ")
target = input()

x=0
P=list()
C=list()
for i in q:
    x = x+1
    if x % 2 == 1:
        P.append(q[x-1])
    else:
        C.append(q[x-1])

abcd = ('abcdefghijklmnopqrstuvwxyz')
abcd_list = list(abcd)

P2 = list()
x=0
for i in P:
    word=list()
    x = x+1
    for letter in P[x-1]:
        if abcd.index(letter) > 1:
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

target = target.split(" ")

x=0
answer=list()
for i in target:
    answer.append(dic[target[x]])
    x=x+1

print(" ".join(answer))


