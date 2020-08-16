import itertools
for case in list(itertools.combinations([int(input()) for i in range(9)], 7)):
    if sum(case) == 100:
        for person in sorted(list(case)):
            print(person)