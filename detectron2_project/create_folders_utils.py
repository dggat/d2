
import os

def exists_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Directory " , directory ,  " Created ")
    else:    
        print("Directory " , directory ,  " already exists")






if __name__ == "__main__":

    directory = 'C:\\Users\\djblack7\\detectron2_v2\\detectron_repo\\detections\\dotay'
    exists_directory(directory)