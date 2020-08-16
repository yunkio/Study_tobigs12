import pandas as pd
data = pd.read_csv(input())
data["IQ"] = pd.DataFrame(data["IQ"]).apply(pd.to_numeric, errors='coerce') # 결측치 처리
M, F = (data[data['Gender'] == 'M']), (data[data['Gender'] == 'F']) # 남자,여자
R = (data["IQ"].isna().sum() / len(data))*100 # R값 구하기
M.fillna(int(M["IQ"].mean()), inplace=True), F.fillna(int(F["IQ"].mean()), inplace=True) # 결측치 대체
E = (sum(abs((M["Height"] - 70) - M["IQ"])) + sum(abs((F["Height"] - 60) - F["IQ"])))/len(data) # E
H = ((sum(M["Height"].between(168, 172)) + sum(F["Height"].between(158, 162)))/len(data))*100 # H
print(int(R), int(E), int(H))
if (R <= 25) and (E <= 5) and (H <= 50):
    print("GodDaeWoong")
else: 
    print("GodHyeIn")