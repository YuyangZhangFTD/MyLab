from collections import defaultdict

f = open("input/order.csv")
record = defaultdict(list)
head = f.readline()
while True:
    s = f.readline()
    if len(s) < 3:
        break
    a = s.split(",")
    record[a[4]].append(s)

for key, items in record.items():
    with open("input/"+key+".csv", "w") as w:
        w.write(head)
        for item in items:
            w.write(item)


