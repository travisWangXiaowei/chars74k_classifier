import os 
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg from PIL 
import Image for j in range(7):
    for i in range(10):
        rootdir = '/home/ec2-user/SageMaker/English/Img/GoodImg/Bmp/Sample0{}{}/'.format(j,i)
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                from PIL import Image
                name = os.path.join(subdir, file)
                img = Image.open(name).convert('LA')
                newdir = ('/home/ec2-user/SageMaker/Bmp/Sample0{}{}/').format(i)
                name2 = os.path.join(newdir,file)
                img.save(name2)
                img = Image.open(name2)
                new_img = img.resize((64,64))
                new_img.save(name2, "PNG", optimize=True)
