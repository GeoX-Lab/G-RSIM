# -*- coding:utf-8 -*-
'''
Files tools.

Version 1.0  2018-10-25 16:36:55
by QiJi Refence:
TODO:
1. xxx

'''
# import sys
import os
import re


# **********************************************
# ***************     ****************
# **********************************************
def rename_file(file_name, ifID=0, addstr=None, extension=None):
    '''Rename a file.
    Args:
        file_name: The name/path of file.
        ifID: 1 - only keep the number(ID) in old_name
            Carefully! if only keep ID, file_name can't be path.
        addstr: The addition str add between name and extension
        extension: Set the new extension(kind of image, such as: 'png').
    '''
    savename, extn = os.path.splitext(file_name)  # extn content '.'
    if ifID:
        # file_path = os.path.dirname(full_name)
        ID_nums = re.findall(r"\d+", savename)
        ID_str = str(ID_nums[0])
        for i in range(len(ID_nums)-1):
            ID_str += ('_'+(ID_nums[i+1]))
        savename = ID_str

    if addstr is not None:
        savename += '_' + addstr

    if extension is not None:
        extn = '.' + extension

    return savename + extn


def mkdir_of_dataset(data_dir):
    '''
    Create folders of DL datasets according to the standard structure:
        ├── "dataset_name"(data_dir)
        |   ├── Log: log the train details
        |   ├── Model
        |       ├── checkpoints
        |   ├── SAVE: save some checkpoints and results
        |   ├── BackUp: backup some data or code
        |   ├── train
        |   ├── train_labels
        |   ├── val
        |   ├── val_labels
        |   ├── test
        |   ├── test_labels
    '''
    dir_list = ['/Log', '/Model', '/Model/checkpoints',
                '/SAVE', '/BackUp', '/train', '/train_labels',
                '/val', '/val_labels', '/test', '/test_labels']
    for a_dir in dir_list:
        if not os.path.exists(data_dir+a_dir):
            os.mkdir(data_dir+a_dir)
            print('make %s' % (a_dir))


def mkdir_of_classifyresult(out_dir, class_list):
    '''
    Make dir of (classification)reslut
    Args:
        out_dir: Dir of output
        class_list: list of class name
    '''
    for a_class in class_list:
        a_dir = out_dir + "/" + a_class
        if os.path.exists(a_dir):
            os.mkdir(a_dir)


def filelist(floder_dir, ifPath=False, extension=None):
    '''
    Get names(or whole path) of all files(with specify extension)
    in the floder_dir and return as a list.

    Args:
        floder_dir: The dir of the floder_dir.
        ifPath:
            True - Return whole path of files.
            False - Only return name of files.(Defualt)
        extension: Specify extension to only get that kind of file names.

    Returns:
        namelist: Name(or path) list of all files(with specify extension)
    '''
    namelist = sorted(os.listdir(floder_dir))

    if ifPath:
        for i in range(len(namelist)):
            namelist[i] = os.path.join(floder_dir, namelist[i])

    if extension is not None:
        n = len(namelist)-1  # orignal len of namelist
        for i in range(len(namelist)):
            if not namelist[n-i].endswith(extension):
                namelist.remove(namelist[n-i])  # discard the files with other extension

    return namelist


def filepath_to_name(full_name, extension=False):
    '''
    Takes an absolute file path and returns the name of the file with(out) the extension.
    '''
    file_name = os.path.basename(full_name)
    if not extension:  # if False then discard extension
        file_name = os.path.splitext(file_name)[0]
    return file_name


# **********************************************
# ************ Main functions ******************
# **********************************************
def rename_files(input_dir, out_dir=None, num=1):
    ''' Rename all the files in a floder by number.
    Args:
        input_dir: Original folder directory
        out_dir: Renamed files output directory(optional)
        num: The starting number of renamed files(default=1)
    '''
    # Log the rename record
    folder_dir = os.path.dirname(input_dir)
    folder_name = os.path.basename(input_dir)
    target = open(folder_dir+"/ReName Log_%s.txt" % folder_name, 'w')

    file_names = sorted(os.listdir(input_dir))
    extension = os.path.splitext(file_names[0])[1]  # Plan A
    if out_dir is None:
        out_dir = input_dir
    for name in file_names:
        newname = ("%.5d" % (num))+extension  # Plan A
        # newname = rename_file(name, ifID=1)  # Plan B
        # TODO: 文件名中文乱码，下同
        os.rename(input_dir+'/'+name, input_dir+'/'+newname)
        target.write(name + '\tTo\t' + newname + '\n')
        num += 1

    target.close()
    print("Finish rename.")


def main():
    pass


if __name__ == '__main__':
    # main()
    # mkdir_of_dataset('/home/tao/Data/RBDD')
    pass
