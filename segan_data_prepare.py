import os
path = '/media/shrutikshirsagar/Data/Multimodal_journa/try'

for root, dirname, filename in os.walk(path):
    s = os.path.basename(root)
    
    
    for name in filename:
        old_name = os.path.join(root, name)
        #print(old_name)
       
        newname = name.replace('dev' , s + '_dev' )
        print(newname)
        
        new_name1 = os.path.join(root, newname)
        os.rename(old_name,new_name1)
        
