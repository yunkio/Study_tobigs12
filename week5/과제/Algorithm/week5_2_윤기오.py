total, now = 0, 0
for letter in input():
    if letter == "O":
        now += 1
        total = total + now
    else:
        now = 0
print(total)