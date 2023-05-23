from django.shortcuts import render
import keras
from PIL import Image
import numpy as np
import os
from django.core.files.storage import FileSystemStorage

# Create your views here.
media='media'
model = keras.models.load_model('models/trained.h5')

def makepredictions(path):
    #we open the image

    img=Image.open(path)

    #we resize the image for model

    img_d = img.resize((255,255))

    # we check if image is RGB or not

    if len(np.array(img_d).shape)<4:
        rgb_img =Image.new("RGB",img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img=img_d


    # here we convert the image into numpy array and reshape
    rgb_img=np.array(rgb_img,dtype=np.float64)
    rgb_img=rgb_img.reshape(-1,255,255,3)

    #we make predictions here

    predictions =model.predict(rgb_img)
    a=int(np.argmax(predictions))
    if a==1:
        a="Result : glioma_tumor"
    elif a==2:
        a="Result : meningioma_tumor"
        
    elif a==3:
        a="Result : no_tumor"
    else:
        a="Result: pituitary_tumor"
    return a

def index_mri(request):
    if request.method == "POST" and request.FILES['upload']:

        if 'upload' not in request.FILES:
            err='No images Selected'
            return render(request,'mri_result.html',{'err':err})
        f = request.FILES['upload']
        if f == '':
            err='No files selected'
            return render(request,'mri_result.html',{'err':err})
        upload =request.FILES['upload']
        fss = FileSystemStorage() 
        file =fss.save(upload.name,upload)
        file_url=fss.url(file)
        predictions=makepredictions(os.path.join(media,file))
        return render(request,'mri_result.html',{'pred':predictions,'file_url':file_url})
    else:
        return render(request,'index.html')
