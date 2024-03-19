"FUNCTIONS FOR GENERATION OF DATASETS FOR NERF FROM OMNIVERSE CODE AND BLENDER"

import numpy as np
import os
import shutil
import json


def sort_and_name_images(using_blender = True, training_split = 0.9, img_folder = "imgs", project_data_dir = '/home/eherrin@ad.ufl.edu/code/coral_nerf/coral'):
    '''Generates training and test image data from omniverse or blender and stores it
      in a data folder in your project's directory. For example, in project "coral_nerf" the data folder
      is "coral". Within coral the folder "imgs" is created which will hold the training and test data
      
      Returns: number of files transfered'''
      #FIXME: ensure/ fix omniverse case: needs to be validated
    
    img_folder = project_data_dir + "/" + img_folder

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    png_files = [f for f in os.listdir(project_data_dir) if f.endswith('.png')] # Note: These will be unsorted!

    if using_blender == True:
        # When using blender we sort numerically since images start at 0000.png
        png_files = sorted(png_files, key=lambda x: int(x.split('.')[0]))

    train_count = int(training_split * len(png_files))
    test_count = len(png_files) - train_count

    train_counter = 0
    test_counter = 0

    for i, file_name in enumerate(png_files):
        # We now rename all png files to be training and test images
        if i < train_count:
            os.rename(os.path.join(project_data_dir, file_name), os.path.join(project_data_dir,f"train_{train_counter}"+".png"))
            train_counter += 1
        else:
            os.rename(os.path.join(project_data_dir, file_name), os.path.join(project_data_dir,f"test_{test_counter}"+".png"))
            test_counter += 1

    png_files = [f for f in os.listdir(project_data_dir) if f.endswith('.png')] # this line is needed as files in previous location no longer exist
    files_moved_count = sum(1 for file in png_files if shutil.move(os.path.join(project_data_dir, file), os.path.join(img_folder, file)))

    return files_moved_count

    # Code creates a train folder containing a pose folder which contains training and test pose.txt files

def omniverse_pose_and_intrinsics(training_split=0.9,
                                  train_folder = 'train',
                                  project_data_dir='/home/eherrin@ad.ufl.edu/code/coral_nerf/coral'):
    '''
    Creates a data folder (train_folder) containing pose and intrinsics data for omniverse data which is
    given as .json files and converted to .txt files. Training and test data is stored together in train_folder/poses
    and train_folder/intrinsics. Train folder can be named however you like
    '''
    
    train_folder = project_data_dir +'/' + train_folder

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        if not os.path.exists(train_folder+'/pose'):
            os.makedirs(train_folder+'/pose')
        if not os.path.exists(train_folder+'/intrinsics'):
            os.makedirs(train_folder+'/intrinsics')

    # Takes intrinsics data and pose data from json files and puts into individual .txt files
    json_files = [f for f in os.listdir(project_data_dir) if f.endswith('.json')]

    for json_file in json_files:
        with open(os.path.join(project_data_dir, json_file), 'r') as file:
            data = json.load(file)
            mat = data.get('cameraViewTransform') # loading pose data
            # WHY: c2w comes in as an array (1,16) of T mat COLUMNS, when it needs to be ROWS. This switches the 
            # order of the list uch that it can be properly written into the .txt files we use for storing our data
            mat = [mat[0], mat[4], mat[8], mat[12],
                  mat[1], mat[5], mat[9], mat[13],
                  mat[2], mat[6], mat[10], mat[14],
                  mat[3], mat[7], mat[11], mat[15]]
            
            w2c = np.array(mat).reshape(4,4)
            # we inverse to go from w2c to c2w
            c2w = np.linalg.inv(w2c) 

            c2w = [c2w[0,0],c2w[0,1],c2w[0,2],c2w[0,3],
                  c2w[1,0],c2w[1,1],c2w[1,2],c2w[1,3],
                  c2w[2,0],c2w[2,1],c2w[2,2],c2w[2,3],
                  c2w[3,0],c2w[3,1],c2w[3,2],c2w[3,3]]
            print(c2w)

            res = data.get('renderProductResolution') # loading intrinsics info # FIXME: Should be center of image, NOT the resolution... so half the resolution
            focal = data.get('cameraFocalLength')
            intrinsics = [focal,0.0,res[0]/2,0.0,0.0,focal,res[1]/2,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0]

            if c2w:
                output_file_path = os.path.join(train_folder+'/pose', f"{json_file.split('.')[0]}.txt")
                output_file_path_intr = os.path.join(train_folder+'/intrinsics', f"{json_file.split('.')[0]}.txt")

                with open(output_file_path, 'w') as output_file:
                    for item in c2w:
                        output_file.write(str(item)+ '\n') 

                with open(output_file_path_intr, 'w') as output_file_intr:
                    for item in intrinsics:
                        output_file_intr.write(str(item)+'\n')

    # splitting into train and test for pose and intrinsics data
    pose_files = [f for f in os.listdir(train_folder+'/pose') if f.endswith('.txt')]
    intrinsics_files = [f for f in os.listdir(train_folder+'/intrinsics') if f.endswith('.txt')]

    train_count = int(training_split * len(pose_files)) # 3-12-24 noticed training/test split was hardcoded -resolved
    test_count = len(pose_files) - train_count

    # test and train split for pose data
    train_counter = 0
    test_counter = 0

    for i, file_name in enumerate(pose_files):
        if i < train_count:
            os.rename(os.path.join(train_folder+'/pose', file_name), os.path.join(train_folder+'/pose',f"train_{train_counter}"+".txt"))
            train_counter += 1
        else:
            os.rename(os.path.join(train_folder+'/pose', file_name), os.path.join(train_folder+'/pose',f"test_{test_counter}"+".txt"))
            test_counter += 1

    # test and train split for intrinsics data
    train_counter = 0
    test_counter = 0

    for i, file_name in enumerate(intrinsics_files):
        if i < train_count:
            os.rename(os.path.join(train_folder+'/intrinsics', file_name), os.path.join(train_folder+'/intrinsics',f"train_{train_counter}"+".txt"))
            train_counter += 1
        else:
            os.rename(os.path.join(train_folder+'/intrinsics', file_name), os.path.join(train_folder+'/intrinsics',f"test_{test_counter}"+".txt"))
            test_counter += 1      

def blender_pose_and_intrinsics(training_split=0.9,
                                focal = 50,
                                  train_folder = 'train',
                                  project_data_dir='/home/eherrin@ad.ufl.edu/code/coral_nerf/cube'):
    '''
    Creates a data folder (train_folder) containing pose and intrinsics data for blender data which is
    given as a .json file and converted to .txt files. Training and test data is stored together in train_folder/poses
    and train_folder/intrinsics. Train folder can be named however you like.
    '''
    train_folder = project_data_dir +'/' + train_folder
    json_files = [f for f in os.listdir(project_data_dir) if f.endswith('.json')]

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        if not os.path.exists(train_folder+'/pose'):
            os.makedirs(train_folder+'/pose')
        if not os.path.exists(train_folder+'/intrinsics'):
            os.makedirs(train_folder+'/intrinsics')

    with open(os.path.join(project_data_dir, json_files[0]), 'r') as file:

        data = json.load(file) # load the json file containing all the poses

        # For each frame/ pose in the json file, we extract it.
        for i, frame in enumerate(data['frames']):
            mat = frame['transform_matrix']

            # puts transform value on each line of the text file generated
            c2w =  [mat[0][0], mat[0][1], mat[0][2], mat[0][3],
                    mat[1][0], mat[1][1], mat[1][2], mat[1][3],
                    mat[2][0], mat[2][1], mat[2][2], mat[2][3],
                    mat[3][0], mat[3][1], mat[3][2], mat[3][3]]
        
            # Instrinsics Information

            cxy = [200,200] # Where the center of the image is
            intrinsics = [focal,0.0,cxy[0],0.0,0.0,focal,cxy[1],0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0]

            if c2w:
                # get rid of 'train/' at the beginning of each frame name, which we use for indexing within the json file
                output_file_path = os.path.join(train_folder+'/pose', f"{data['frames'][i]['file_path'].replace('train/','').split('.')[0]}.txt")
                output_file_path_intr = os.path.join(train_folder+'/intrinsics', f"{data['frames'][i]['file_path'].replace('train/','').split('.')[0]}.txt")
                
                with open(output_file_path, 'w') as output_file:
                    for item in c2w:
                        output_file.write(str(item)+ '\n') 
                
                with open(output_file_path_intr, 'w') as output_file_intr:
                    for item in intrinsics:
                        output_file_intr.write(str(item)+'\n')

    # splitting into train and test for pose and intrinsics data
    pose_files = [f for f in os.listdir(train_folder+'/pose') if f.endswith('.txt')]
    intrinsics_files = [f for f in os.listdir(train_folder+'/intrinsics') if f.endswith('.txt')]

    train_count = int(training_split * len(pose_files)) # 3-12-24 noticed training/test split was hardcoded -resolved
    test_count = len(pose_files) - train_count

    # test and train split for pose data
    train_counter = 0
    test_counter = 0

    for i, file_name in enumerate(pose_files):
        if i < train_count:
            os.rename(os.path.join(train_folder+'/pose', file_name), os.path.join(train_folder+'/pose',f"train_{train_counter}"+".txt"))
            train_counter += 1
        else:
            os.rename(os.path.join(train_folder+'/pose', file_name), os.path.join(train_folder+'/pose',f"test_{test_counter}"+".txt"))
            test_counter += 1

    # test and train split for intrinsics data
    train_counter = 0
    test_counter = 0

    for i, file_name in enumerate(intrinsics_files):
        if i < train_count:
            os.rename(os.path.join(train_folder+'/intrinsics', file_name), os.path.join(train_folder+'/intrinsics',f"train_{train_counter}"+".txt"))
            train_counter += 1
        else:
            os.rename(os.path.join(train_folder+'/intrinsics', file_name), os.path.join(train_folder+'/intrinsics',f"test_{test_counter}"+".txt"))
            test_counter += 1   



def omniverse_seperate_training_and_test_pose_and_intrinsics(
                                  train_folder = 'train',
                                  project_data_dir='/home/eherrin@ad.ufl.edu/code/coral_nerf/coral'):
    '''Moves test data out of training folder to a new test folder'''
    
    test_folder = project_data_dir + '/test'

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        if not os.path.exists(test_folder+'/pose'):
            os.makedirs(test_folder+'/pose')
        if not os.path.exists(test_folder+'/intrinsics'):
            os.makedirs(test_folder+'/intrinsics')

    # reading in pose and intrinsics from training data folder
    pose_files = os.listdir(project_data_dir + '/' + train_folder + '/pose')
    intrinsic_files = os.listdir(project_data_dir + '/' + train_folder + '/intrinsics')

    for filename in pose_files:
        if filename.startswith("test"):
            source_file = os.path.join(project_data_dir + '/' + train_folder + '/pose', filename)
            destination_file = os.path.join(test_folder + '/pose', filename)
            shutil.move(source_file, destination_file)
    
    for filename in intrinsic_files:
        if filename.startswith("test"):
            source_file = os.path.join(project_data_dir + '/' + train_folder + '/intrinsics', filename)
            destination_file = os.path.join(test_folder + '/intrinsics', filename)
            shutil.move(source_file, destination_file)