A, B = (input().split())
A_list, B_list = [input() for i in range(int(A))], [input() for i in range(int(B))]
Pass = [result.capitalize() for result in B_list if result in A_list]
print(len(Pass), "\n".join(sorted(Pass)), sep="\n")