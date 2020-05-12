
import os
import fnmatch


import numpy as np

pathin = "/media/shrutikshirsagar/Data/Multimodal_journa/MSF_prepared_RECOLA"
pathout = "/media/shrutikshirsagar/Data/Multimodal_journa/MSF_bow_a"
temporary = "/media/shrutikshirsagar/Data/Multimodal_journa/temporary"


def load_labels(filename, col_labels=1, skip_header=True, delim=','):
    # Reads column col_labels from csv file (arbitrary data type)
    # filename:      csv-filename
    # col_labels:    Index of the target column (indexing starting with 1, default: 1)
    # skip_header:   Skip the first line of the given csv file (default: True)
    # delim:         Delimiter (default: ';')
    
    labels = []
    with open(filename, 'r') as csv_file:
        print(filername)
        if skip_header:
            next(csv_file)
        for line in csv_file:
            cols = line.split(delim)
            labels.append(cols[col_labels-1].rstrip())
    return np.array(labels)


def load_features(filename, skip_header=True, skip_instname=True, delim=',', num_lines=0):
    # Reads a csv file (only numbers, except for first item if skip_instname=True) with given delimiter
    # filename:      csv-filename
    # skip_header:   Skip the first line of the given csv file (default: True)
    # skip_instname: Skip the first column/attribute of the given csv file, e.g., the filename (default: True)
    # delim:         Delimiter (default: ';')
    # num_lines:     Number of lines in the CSV file. If given, the function is faster.
    # 
    # Return: numpy array (float)
    # 
    # This function is 6.3 times faster than loadtxt (also without number of lines given):
    #  data = np.loadtxt(open(filename, 'r'), delimiter=delim)
    
    if num_lines==0:
        num_lines = get_num_lines(filename,skip_header)
    
    data = np.empty((num_lines,get_num_columns(filename,skip_header,skip_instname,delim)), float)
    print('data shaope', data.shape)
    with open(filename, 'r') as csv_file:
        if skip_header:
            next(csv_file)
        c = 0
        for line in csv_file:
            offset = 0
            if skip_instname:
                offset = line.find(delim)+1
            data[c,:] = np.fromstring(line[offset:], dtype=float, sep=delim)
            c += 1
    print(data.shape)
    return data


# Helper functions
def get_num_lines(filename,skip_header):
    with open(filename, 'r') as csv_file:
        if skip_header:
            next(csv_file)
        c = 0
        for line in csv_file:
            c += 1
    return c

def get_num_columns(filename,skip_header,skip_instname,delim):
    with open(filename, 'r') as csv_file:
        if skip_header:
            csv_file.readline()
        line = csv_file.readline()
        offset = 0
        if skip_instname:
            offset = line.find(delim)+1
        cols = np.fromstring(line[offset:], dtype=float, sep=delim)
    return len(cols)

def save_features(filename, data, append=False, instname='', header='', delim=';', precision=8):
    # Write a csv file
    # filename: csv-filename
    # data:     Data given as numpy array
    # append:   If True, file (filename) is appended, otherwise, it is overwritten
    # instname: If non-empty string is given, instname is added as a first element in each row
    # header:   If non-empty string is given, print header line as a string
    # delim:    Delimiter (default: ';')
    # delim:    Floating point precision (default: 8)
    
    mode = 'w'
    if append:
        mode = 'a'
        if os.path.isfile(filename):
            header = ''  # do never write header if file already exists and is appended
    
    with open(filename, mode) as csv_file:
        if len(header)>0:
            csv_file.write(header + '\n')
        
        for row in data:
            if len(instname)>0:
                csv_file.write('\'' + instname + '\'' + delim)
            csv_file.write(np.array2string(row, max_line_width=100000, precision=precision, separator=delim)[1:-1].replace(' ','') + '\n')  # check max_line_width



import os.path
def  bow_all_folder(pathin, pathout):
    dirs = os.listdir("%s/%s"%(pathin, dirname))
    print('directory_open', dirs)
    
    dirFiles1 = dirs #list of directory files
    dirFiles1.sort()
    print(dirFiles1)
    if not os.path.exists("%s/%s" % (pathout, dirname)):
    
        os.makedirs("%s/%s" % (pathout, dirname))
    s = '/'
    new_path = os.path.join(pathin, dirname)
    print('new_path', new_path )
    out_path = os.path.join(pathout, dirname)
    print(out_path)
    folder_audio_features = new_path+'/'
    print('folder_audio', folder_audio_features)
    folder_boaw = out_path+'/'
    print('folder output', folder_boaw)
    temp_folder = os.path.join(temporary, dirname)  # To store codebook files
    temporary_folder = temp_folder + '/'
    print('tem_folder', temporary_folder)
    # Window size
    jar_openxbow = 'openXBOW.jar'  # https://github.com/openXBOW/openXBOW
    window_size  = 4.0

    # Do NOT modify
    hop_size = 0.04  # hop size of the labels

    openxbow_options_audio           = '-attributes nt1[223] -writeName -writeTimeStamp -t ' + str(window_size) + ' ' + str(hop_size)
    #openxbow_options_visual          = '-attributes nt1[18] -writeName -writeTimeStamp -t ' + str(window_size) + ' ' + str(hop_size)
    openxbow_options_codebook_audio  = '-size 500 -a 1 -log -standardizeInput'
    #openxbow_options_codebook_visual = '-size 500 -a 1 -log'

    # Get all files
    files_train = fnmatch.filter(os.listdir(folder_audio_features), 'Train_*')  # filenames are the same for all modalities
    files_devel = fnmatch.filter(os.listdir(folder_audio_features), 'Devel_*')
    #files_test  = fnmatch.filter(os.listdir(folder_audio_features), 'Test_*')
    files_train.sort()
    files_devel.sort()
    #files_test.sort()

    # Create folder, remove existing files
    if not os.path.exists(temporary_folder):
        os.mkdir(temporary_folder)
    if not os.path.exists(folder_boaw):
        os.mkdir(folder_boaw)

    for fn in ['Train_audio.csv', 'Codebook_audio.txt']:
       if os.path.exists(temporary_folder + fn):
          os.remove(temporary_folder + fn)

    # Create codebooks
    print('Concatenating training feature files ...')
    codebook_audio  = temporary_folder + 'Codebook_audio.txt'


    for fn in files_train:
       print(fn)
       features = load_features(folder_audio_features + fn)
       save_features(temporary_folder + 'Train_audio.csv',  features, append=True, instname=fn)
  
    print('Generating codebook file ' + codebook_audio)
    os.system('java -jar ' + jar_openxbow + ' -B ' + codebook_audio  + ' -i ' + temporary_folder + 'Train_audio.csv'  + ' ' + openxbow_options_audio  + ' ' + openxbow_options_codebook_audio)

    # Generate XBOW files for train/devel/test and audio/video
    print('Generating BoAW and BoVW feature files for Train/Devel/Test and audio/video ...')
    for fn in files_train:
        print('Generating BoAW and BoVW for file ' + fn + ' ...')
        os.system('java -jar ' + jar_openxbow + ' -b ' + codebook_audio  + ' -i ' + folder_audio_features  + fn + ' -o ' + folder_boaw + fn + ' ' + openxbow_options_audio)
           #os.system('java -jar ' + jar_openxbow + ' -b ' + codebook_visual + ' -i ' + folder_visual_features + fn + ' -o ' + folder_bovw + fn + ' ' + openxbow_options_visual)
    for fn in files_devel:
        print('Generating BoAW and BoVW for file ' + fn + ' ...')
        os.system('java -jar ' + jar_openxbow + ' -b ' + codebook_audio  + ' -i ' + folder_audio_features  + fn + ' -o ' + folder_boaw + fn + ' ' + openxbow_options_audio)
    #os.system('java -jar ' + jar_openxbow + ' -b ' + codebook_visual + ' -i ' + folder_visual_features + fn + ' -o ' + folder_bovw + fn + ' ' + openxbow_options_visual)

for dirname in os.listdir(pathin):
    print('directname', dirname)
    print('path in', os.listdir(pathin))
    print("Processing: %s/%s" % (pathin, dirname))
    
    
    bow_all_folder(pathin, pathout)
