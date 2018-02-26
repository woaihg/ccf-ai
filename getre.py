import pandas as pd

lawlist=[59, 338, 363, 267, 383, 42, 341, 357, 128, 55, 79, 65, 209, 225, 57, 61, 43, 348, 31, 386, 343, 144, 70, 275, 77, 269, 30, 64, 293, 69, 358, 356, 271, 210, 359, 73, 27, 176, 63, 213, 76, 52, 385, 41, 382, 198, 62, 143, 266, 56, 68, 342, 393, 274, 205, 303, 292, 345, 58, 264, 47, 17, 36, 263, 86, 238, 45, 93, 192, 37, 175, 354, 141, 277, 224, 280, 214, 234, 44, 78, 75, 25, 23, 239, 71, 177, 196, 240, 133, 312, 336, 72, 276, 347, 22, 12, 19, 26, 67, 38, 53, 140, 390, 18, 344]
for law in lawlist:
    print(law)
    a=pd.read_csv("data/127_1.csv")
    b=a[a['law']!=law]
    b=b.reset_index()
    c=pd.read_csv("re/"+str(law)+".csv")
    d=pd.concat([b[['row_id','law']],c],axis=0)
    d.to_csv("data/127_1.csv",index=None)


a=pd.read_csv("data/testrow.csv")
law=pd.read_csv("data/127_1.csv")
money=pd.read_csv("data/money-11-2.csv")
output = open("result1207_1.csv", 'w')
row_id=set(money['row_id'])
for tt in range(len(a)):
    row=a.iloc[tt]['row_id']
    print(row)
    _money=money[money['row_id']==row]
    _law=law[law['row_id']==row]
    m=_money.iloc[0]['money']
    line="{\"id\":\""+str(row)+"\",\"penalty\":"+str(m)+",\"laws\":["
    if len(_law)>0:
        for lw in _law['law']:
            line=line+str(lw)+','
    else:
        line=line+str(-1)+','
    line=line[0:-1]
    line=line+"]}\n"
    output.write(line)

output.close()
