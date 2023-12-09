import numpy as np
with open("./point.txt" ,"r") as file:
    # print(file.readlines())
    lps = file.readlines()
    
    len_lps = len(lps)
    p1, p2 = 0,0
    for lp in lps:
        p1 += int(lp.strip().split(",")[0][1:])
        p2 += int(lp.strip().split(",")[1][:-1])
    
    minp1, minp2 = p1/len_lps, p2/len_lps
    var1, var2   = 0,0
    for lp in lps:
        var1 += (minp1 - int(lp.strip().split(",")[0][1:]))**2
        var2 += (minp2 - int(lp.strip().split(",")[1][:-1]))**2

    print(minp1, minp2)
    print(np.sqrt(var1/minp1) , np.sqrt(var2/minp2))
    