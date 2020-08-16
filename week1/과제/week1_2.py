key = dict()
for i in range(int(input())):
    P, C = input().split()
    P = "".join('z' if c == 'a' else chr(ord(c) - 1) for c in P)
    key[C] = P
print(" ".join(key[i] for i in input().split()))