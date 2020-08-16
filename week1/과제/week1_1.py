N = int(input())
num = list(filter(lambda x: '9' not in x, input().split()))
print('9' * 9 if len(num) < N else int(num[N - 1]) ** 2)