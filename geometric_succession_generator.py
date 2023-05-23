# symmetrical w.r.t. 128.
# 'total' changes the length of the list generated
# 'shift' to finetune the numbers

shift = 2.5
total = 5

b = 128**(1/(total-1+shift))

geom_succession = []
l = []

clip = lambda x: min(max(x,0),255)

for i in range(total):
    d = int(b**(i+shift))
    geom_succession.append(d)
    l.append((clip(128-d),50,clip(128+d),50))

print(geom_succession)
print(l)

# (120,50,136,50),(100,50,156,50)