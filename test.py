from utils import measure_dist,centre_of_bbox
import numpy as np
a = np.array([(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18),(19,20),(21,22),(23,24),(25,26),(27,28)])
print(a[1]-a[0])
b = [1,2,3,4,5,6,7,8]
c = [599.33,310.63, #0
     1309.3,310.45, #1
     339.4,853.02, #2
     1568.7,851.71, #3
     688.38,310.58, #4
     494.07,852.89, #5
     1220.4,310.46, #6
     1414.7,851.69, #7
     659.8,90.25, #8
     1249,389.88, #9
     560.99,666.01, #10
     1348.4,665.21, #11
     954.17,389.98, #12
     954.54,665.73] #13

#print(len(c))

bbox1 = [480.2823791503906, 751.597412109375, 624.8380126953125, 930.6015014648438]
bbox2 = [1030.5684814453125, 201.68313598632812, 1095.9012451171875, 308.51904296875]
bbox3 = [1494.946044921875, 77.82566833496094, 1544.8236083984375, 195.40792846679688]
bbox4 = [1609.7119140625, 276.98779296875, 1681.4844970703125, 389.99664306640625]
bbox5 = [387.4237060546875, 72.42877197265625, 424.358642578125, 198.62344360351562]

x_total = 0
for x in range(0,len(c),2):
    x_total += c[x]
x_center_of_court = x_total / (len(c)/2)

y_total = 0
for y in range(1,len(c),2):
    y_total += c[y]
y_center_of_court = y_total / (len(c)/2)
#print(f"({x_center_of_court},{y_center_of_court})")

# print(a[2][1], a[3][1]) # 6 8
# print(a[2],a[3]) #(5,6),(7,8)
# print(int(a[0][1]+a[1][1])/2) #3

# for i in range(0,len(a),2):
#     print(i) #0 2 4 6 8 10 12

