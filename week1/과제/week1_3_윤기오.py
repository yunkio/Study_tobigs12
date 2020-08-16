f = open("phone_book.txt", 'r')
all = f.read()
numbers = all.split('\n')

for i in numbers:
    y=list()
    y.append(i.split('-'))

mynumber = list()
for i in numbers:
    mynumber.append(i.split('-'))

result = list()
for (first, middle, last) in mynumber:
    if first == '010' and int(middle) > int(last):
        result.append(''.join(mynumber[x]))
    x=x+1

answer=list()
for i in result:
    answer.append(i+'\n')

f = open("new_phone_book.txt", 'w')
f.writelines(answer)
f.close()



