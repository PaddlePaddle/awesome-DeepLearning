import os
root_path = "/home/aistudio/data/data128448/index_motorcycle/"
dirs = os.listdir(root_path)
dir_dict = {"person":"人","motorcycle":"电瓶车/摩托车","bicycle":"自行车","others":"其他"}
with open("index_label.txt","w") as f: 
    for dir in dirs:
        path = root_path + dir + "/"
        print(path)
        filenames = os.listdir(path)
        for filename in filenames:
            f.write(path+filename+"	"+dir+"\n")