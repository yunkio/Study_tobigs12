m,n = int(input()),int(input())
floor = [list(range(1, n+1))]
for x in range(m) :
    floor.append([sum(floor[x][:room]) for room in range(1, n+1)])
print(floor[-1][-1])