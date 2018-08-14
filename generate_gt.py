import csv
import os
import pdb
'''
words = lines.split(',')
            imid = words[0]
            classid = self._class_to_ind_image[words[1]]
            x1 = float(words[2])
            y1 = float(words[3])
            x2 = float(words[4])
            y2 = float(words[5])
            crowd = int(words[6])
            height = float(words[7])            
            width = float(words[8])
'''
classid = '/m/01g317'
x1 = float(0.1)
y1 = float(0.1)
x2 = float(0.2)
y2 = float(0.2)
crowd = int(0)
height = float(0.1)
width = float(0.1)
f = open('./data/openimage/annotations/test_challenge_2018.csv', 'w')
csvwriter = csv.writer(f, delimiter=',')
for im_name in os.listdir('./data/openimage/images/test_challenge_2018'):
    imid = im_name.split('.')[0]
    csvwriter.writerow([imid, classid, x1, y1, x2, y2, crowd, height, width])
f.close()
