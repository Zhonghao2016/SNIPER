import pdb
import csv
import json
from os import listdir
from os.path import isfile, join
import numpy as np
 
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

sequence = []
def dict_generator(indict, pre=None):
    global sequence
    pre = pre[:] if pre else []
    if 'Subcategory' in indict.keys():
        for item in indict['Subcategory']:
            dict_generator(item, pre=pre+[indict['LabelName']])
    sequence.append(pre + [indict['LabelName']])


def read_hierarchy():
    with open('bbox_labels_600_hierarchy.json') as json_data:
        jd = json.load(json_data)
        json_data.close()
    hierarchy_dict = {}
    dict_generator(jd)
    for item in sequence:
        if len(item) < 3:
            hierarchy_dict[item[-1]] = []
            continue
        hierarchy_dict[item[-1]] = item[1:-1]
    return hierarchy_dict

def combine_two():
    i = 0
    f_write = open('submit.csv', 'w')
    f = open('test2018_output1.csv', 'r')
    for line in f:
        f_write.write(line)
    f.close()
    # manually check the file to see whether it needs a '\n'
    # f_write.write('\n')
    f = open('test2018_output2.csv', 'r')
    for line in f:
        f_write.write(line)
    f.close()

    f_write.close()


def process():
    """
    add hierarchy to the output file, and convert the output file format to the one can be submitted
    """
    thre = 0.0115
    mypath = '../data/openimages/images/test_challenge_2018/'
    f = open('submit1.csv', 'r')
    f_write = open('submit3.csv', 'w')

    f_write.write('ImageId,PredictionString\n')
    hierarchy = read_hierarchy()
    i = 0
    j = 0
    name_all = []
    for line in f:
        #if i == 0:
        #    i = 1
        #    continue
        j += 1
        line_split = line.split(',')
        if len(line_split) == 1:
            f_write.write(line)
            continue

        name = line_split[0]
        content = line_split[1].split(' ')
        con_write = []
        for k in range(len(content)/6):
            if float(content[k*6+1]) > thre:
                temp = [content[k*6], content[k*6+1]]
                for w in range(2, 6):
                    if float(content[k*6+w]) > 1.0:
                        temp.append(str('1.0'))
                    else:
                        temp.append(str(float(content[k*6+w])))
                for item in temp:
                    con_write.append(item)
                for item in hierarchy[temp[0]]:
                    con_write.append(item)
                    for w in range(1, 6):
                        con_write.append(temp[w])

        name_all.append(name)
        f_write.write(name+','+' '.join(con_write)+'\n')

    
    onlyfiles = [files for files in listdir(mypath) if isfile(join(mypath, files))]

    for image_id in onlyfiles:
        if image_id[:-4] in name_all:
            continue
        j += 1
        f_write.write(line[:-4]+',\n')

    f.close()
    f_write.close()
    print(j)

def combine():
    """
    supplement the output file to have all the image id in validation set
    """
    mango_im = open('mango.csv', 'r')
    #all_im = open('submit2.csv', 'r')
    write_file = open('mango1.csv', 'w')
    mypath = '../data/openimages/images/validation/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #csvwriter = csv.writer(write_file, delimiter=',')
    exist = []
    for line in mango_im:
        line_split = line.split(',')
        exist.append(line_split[0])
        write_file.write(line)
    for line in onlyfiles:
        if line[:-4] in exist:
            continue
        write_file.write(line[:-4]+',\n')        
    mango_im.close()
    #all_im.close()
    write_file.close()


def check():
    """
    check whether the number of lines of the submission file
    """
    f = open('submit2.csv', 'r')
    i = 0
    j = 0
    for line in f:
        i += len(line.split(' ')) / 6.
        j += 1
        #if j == 11111:
        #    print(line[-1])
    print(j)
    print(i/99999)
    #print(i)

if __name__ == "__main__":
    process()
    #read_hierarchy()
    #check()
    #combine_two()
