## Style transfer Implementation
#player with different conv layers so that you balance between the content and style of the image.
#Also you can tinker the weights of layer responses so as to add the amount of style from each layer.


import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import scipy.misc

from scipy.optimize import fmin_l_bfgs_b
from keras import metrics
from keras.models import Model
from vgg16_avg import VGG16_Avg
import keras.backend as K

img_path =   #Add your own image path here
style_image_path = # Add the style image path here
rand_img = lambda shape: np.random.uniform(-2.5, 2.5, shape)/100

def solve_image(eval_obj, niter, x):
    for i in range(niter):
        x, min_val, info = fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                         fprime=eval_obj.grads, maxfun=20)
        x = np.clip(x, -127,127)
        #print(min_val.shape)
        print('Current loss value:', min_val)
        path = #path where the styled image should store 
        scipy.misc.imsave(path, deproc(x.copy(), shp)[0])
    return x

class Evaluator(object):
    def __init__(self, f, shp): self.f, self.shp = f, shp
        
    def loss(self, x):
        loss_, self.grad_values = self.f([x.reshape(self.shp)])
        return loss_.astype(np.float64)

    def grads(self, x): return self.grad_values.flatten().astype(np.float64)

def style_loss(x, targ): return K.mean(metrics.mse(gram_matrix(x), gram_matrix(targ)))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features)) / x.get_shape().num_elements()



rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]

deproc = lambda x,s: np.clip(x.reshape(s)[:, :, :, ::-1] + rn_mean, 0, 255)


img = scipy.misc.imread(img_path)
#scipy.misc.imshow(img)

img_arr = preproc(np.expand_dims(np.array(img), 0)) ##img_arr implies pre-processed image
shp = img_arr.shape
#print(img_arr.shape)

model = VGG16_Avg(include_top=False)

style = scipy.misc.imread(style_image_path)
style = scipy.misc.imresize(style, shp[1:])
style = np.expand_dims(style, 0)[:,:,:,:3]

style_arr = preproc(style) 
shp = style_arr.shape

model = VGG16_Avg(include_top=False, input_shape=shp[1:])
outputs = {l.name: l.output for l in model.layers}


style_layers = [outputs['block{}_conv2'.format(o)] for o in range(1,6)]
content_layer = outputs['block4_conv2']

style_model = Model(model.input, style_layers)
style_targs = [K.variable(o) for o in style_model.predict(style_arr)]

content_model = Model(model.input, content_layer)
content_targ = K.variable(content_model.predict(img_arr))

style_wgts = [0.05,0.2,0.2,0.25,0.3]


loss = sum(style_loss(l1[0], l2[0])*w for l1,l2,w in zip(style_layers, style_targs, style_wgts))
loss += K.mean(metrics.mse(content_layer, content_targ))/10
grads = K.gradients(loss, model.input)
transfer_fn = K.function([model.input], [loss]+grads)

evaluator = Evaluator(transfer_fn, shp)

iterations=15
x = rand_img(shp)
x = solve_image(evaluator, iterations, x)



