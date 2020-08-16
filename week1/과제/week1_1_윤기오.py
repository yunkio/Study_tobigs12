a = int(input())
b = (input()).split(" ")

result = list()
for i in b:
    if i.find('9') < 0:
        result.append(i);
result = list(map(int, result))

if len(result) >= a:
    print(result[a-1]**2)
else:
    print(999999999)


