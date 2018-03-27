#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import deepmodels
import json
import time
import argparse
import os.path
import subprocess
import imageutils
import utils
import dfr_caffe
import caffe
import numpy as np

with open('datasets/lfw/lfw_binary_attributes.json') as f: lfw=json.load(f)
with open('datasets/lfw/filelist.txt','r') as f: lfw_filelist=['images/'+x.strip() for x in f.readlines()]
def make_manifolds(a,s=[],t=[],N=10,X=None,visualize=False):
  '''
  a is the target attribute, s are exclusive attributes for the source,
  t are exclusive attributes for the target.
  '''
  S={k:set(v) for k,v in lfw['attribute_members'].iteritems()}
  T=lfw['attribute_gender_members']
  G=set(T[lfw['attribute_gender'][a]])
  if X is None:
    # test has correct gender, all of the source attributes and none of the target attributes
    X=[i for i in range(len(lfw_filelist)) if i in G and i not in S[a] and not any(i in S[b] for b in t) and all(i in S[b] for b in s)]
    random.seed(123)
    random.shuffle(X)
  else:
    X=[lfw_filelist.index(x) for x in X]

  def distfn(y,z):
    fy=[True if y in S[b] else False for b in sorted(S.keys())]
    fz=[True if z in S[b] else False for b in sorted(S.keys())]
    return sum(0 if yy==zz else 1 for yy,zz in zip(fy,fz))
  # source has correct gender, all of the source attributes and none of the target attributes
  # ranked by attribute distance to test image
  P=[i for i in range(len(lfw_filelist)) if i in G and i not in S[a] and not any(i in S[b] for b in t) and all(i in S[b] for b in s)]
  P=[sorted([j for j in P if j!=X[i]],key=lambda k: distfn(X[i],k)) for i in range(N)]
  # target has correct gender, none of the source attributes and all of the target attributes
  Q=[i for i in range(len(lfw_filelist)) if i in G and i in S[a] and not any(i in S[b] for b in s) and all(i in S[b] for b in t)]
  Q=[sorted([j for j in Q if j!=X[i] and j not in P[i]],key=lambda k: distfn(X[i],k)) for i in range(N)]

  return [lfw_filelist[x] for x in X],[[lfw_filelist[x] for x in y] for y in P],[[lfw_filelist[x] for x in y] for y in Q]

if __name__=='__main__':
  # configure by command-line arguments
  parser=argparse.ArgumentParser(description='Generate LFW face transformations.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--backend',type=str,default='caffe+scipy',choices=['torch','caffe+scipy'],help='reconstruction implementation')
  parser.add_argument('--device_id',type=int,default=1,help='zero-indexed CUDA device')
  parser.add_argument('--K',type=int,default=50,help='number of nearest neighbors')
  parser.add_argument('--scaling',type=str,default='beta',choices=['none','beta'],help='type of step scaling')
  parser.add_argument('--iter',type=int,default=500,help='number of reconstruction iterations')
  parser.add_argument('--postprocess',type=str,default='color',help='comma-separated list of postprocessing operations')
  parser.add_argument('--delta',type=str,default='0.2',help='comma-separated list of interpolation steps')
  config=parser.parse_args()
  postprocess=set(config.postprocess.split(','))
  print(json.dumps(config.__dict__))

  # load CUDA model
  minimum_resolution=200
  if config.backend=='torch':
    import deepmodels_torch
    model=deepmodels_torch.vgg19g_torch(device_id=config.device_id)
  elif config.backend=='caffe+scipy':
    model=deepmodels.vgg19g(device_id=config.device_id)
  else:
    raise ValueError('Unknown backend')

  # download AEGAN cropped+aligned LFW (if needed)
  if not os.path.exists('images/lfw_aegan'):
    url='https://www.dropbox.com/s/isz4ske2kheuwgr/lfw_aegan.tar.gz?dl=1'
    subprocess.check_call(['wget',url,'-O','lfw_aegan.tar.gz'])
    subprocess.check_call(['tar','xzf','lfw_aegan.tar.gz'])
    subprocess.check_call(['rm','lfw_aegan.tar.gz'])

  # read test data
  data=numpy.load('tests/dmt2-lfw-multiple-attribute-test.npz')
  pairs=list(data['pairs'][[4]]) # skip flushed face, not interesting
  X=data['X']

  # comment out the line below to generate all the test images
#   X=X[2000:]

  # Set the free parameters
  # Note: for LFW, 0.4*8.82 is approximately equivalent to beta=0.4
  K=config.K
  delta_params=[float(x.strip()) for x in config.delta.split(',')]

  t0=time.time()
  result=[]
  
 
# Distances for DFR

  distances = []
  distances_c = []
  distances_e = []
  pairs_file = '/home/rpalyam/Documents/Tutworks/test/src/pairs_ed.txt'
  lfw_aegan = '/home/rpalyam/git/deepfeatinterp/images/lfw_aegan/'

# CNN initialization for DFR  


  caffe_root = '/home/rpalyam/Downloads/caffe-master/'

  model_def = caffe_root + 'models/vgg/VGG_FACE_deploy.prototxt'
  model_weights = caffe_root + 'models/vgg/VGG_FACE.caffemodel'

  net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
  
#   original=[]
  # for each test image
  with open(pairs_file, 'r') as fl_pairs:
      lines = fl_pairs.readlines()
      for i in range(len(X)):
        content = lines[i].split()
        print(content)
        original=[]
    #     result.append([])
        xX=X[i].replace('lfw','lfw_aegan')
        o=imageutils.read(xX)
        image_dims=o.shape[:2]
        if min(image_dims)<minimum_resolution:
          s=float(minimum_resolution)/min(image_dims)
          image_dims=(int(round(image_dims[0]*s)),int(round(image_dims[1]*s)))
          o=imageutils.resize(o,image_dims)
        XF=model.mean_F([o])
        original.append(o)
        # for each transform
        for j,(a,b) in enumerate(pairs):
          _,P,Q=make_manifolds(b,[a],[],X=X[i:i+1],N=1)
          P=P[0]
          Q=Q[0]
          xP=[x.replace('lfw','lfw_aegan') for x in P]
          xQ=[x.replace('lfw','lfw_aegan') for x in Q]
          PF=model.mean_F(utils.image_feed(xP[:K],image_dims))
          QF=model.mean_F(utils.image_feed(xQ[:K],image_dims))
          if config.scaling=='beta':
            WF=(QF-PF)/((QF-PF)**2).mean()
          elif config.scaling=='none':
            WF=(QF-PF)
          max_iter=config.iter
          init=o
          # for each interpolation step
          for delta in delta_params:
            curim = xX.split('/')[3]
            filename = '/home/rpalyam/git/deepfeatinterp/sm_50/' + str(curim.split('.')[0])+'_'+str(b)+'_'+str(delta)+'.jpg'
            if not os.path.isfile(filename):
                print(xX,b,delta)
                result =[]
                result.append([])
                t2=time.time()
                Y=model.F_inverse(XF+WF*delta,max_iter=max_iter,initial_image=init)
                t3=time.time()
                print('{} minutes to reconstruct'.format((t3-t2)/60.0))
                result[-1].append(Y)
                result_temp=numpy.asarray(result)
                original_temp=numpy.asarray(original)
                if 'color' in postprocess:
                    result_temp=utils.color_match(numpy.expand_dims(original_temp,1),result_temp)
                m=imageutils.montage(result_temp)
                cur_im = xX.split('/')[3]
                imageutils.write('sm_50/'+str(cur_im.split('.')[0])+'_'+str(b)+'_'+str(delta)+'.jpg',m)
                max_iter=config.iter//2
                init=Y
            # Face scores for Verification
            if len(content) == 3:
                img2 = lfw_aegan + content[0] +'/' + content[0] +'_'+ format(int(content[2]),'04d') + '.jpg'
            elif len(content) == 4:
                img2 = lfw_aegan + content[2] +'/' +content[2] +'_'+ format(int(content[3]),'04d') + '.jpg'
            
            print(img2)
            curr_dist1, curr2, curr3 = dfr_caffe.compute_distance(net, filename, img2)
            distances = np.append(distances, curr_dist1)
            distances_c = np.append(distances_c, curr2)
            distances_e = np.append(distances_e, curr3)

  t1=time.time()
  
  np.save('distances_sm_50_0.2.npy', distances)
  np.save('dist_cosine_sm_50_0.2.npy', distances_c)
  np.save('dist_eucl_sm_50_0.2.npy', distances_e)
#   print('{} minutes ({} minutes per image).'.format((t1-t0)/60.0,(t1-t0)/60.0/result.shape[0]/result.shape[1]))
print('{} minutes total.'.format((t1-t0)/60.0))

