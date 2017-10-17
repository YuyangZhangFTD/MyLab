fp_train = "input/avazu/train.csv"

# divide train set by day
# divided files
fwx = dict()
fwy = dict()
for i in range(21, 31):
    fwx[str(i)] = open("input/x_day"+str(i)+".csv", "w")
    fwy[str(i)] = open("input/y_day" + str(i) + ".csv", "w")

header=["id","label","hour","C1","banner_pos","site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21"]

feature_dict = dict(zip(header, [[] for __ in range(len(header))]))

with open(fp_train) as f:
    f.readline()    # read column name
    while True:
        line = f.readline().split(",")
        if len(line) < 3:
            break
        hour = line[2][4:6]
        line[2] = line[2][-2:]
        fwx[hour].write(
            line[0] + ',' + ','.join(line[2:])
        )
        for i in range(2, len(line)):
            pass
        # write label file
        fwy[hour].write(
            ','.join(line[:2]) + '\n'
        )

# close file
for i in range(21, 31):
    fwx[str(i)].close()
    fwy[str(i)].close()

# feature
# header=["id","hour","C1","banner_pos","site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21"]

