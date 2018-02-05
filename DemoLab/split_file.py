from collections import defaultdict

f = open("input/order.csv")
pid_record = defaultdict(list)
l3_record = defaultdict(set)
head = f.readline()
while True:
    s = f.readline()
    if len(s) < 3:
        break
    a = s.split(",")
    pid_record[a[4]].append(s)
    l3_record[a[6]].add(a[4])

for key, items in pid_record.items():
    with open("input/"+key+".csv", "w") as w:
        w.write(head)
        for item in items:
            w.write(item)

with open("input/categories_l3.csv", "w") as w:
    w.write("l3_gds_group_cd,product_id\n")
    for key, items in l3_record.items():
        for item in items:
            w.write(key+","+item+"\n")

f.close()


