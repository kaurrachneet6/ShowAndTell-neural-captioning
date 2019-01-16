import torch

checkpointl = torch.load("captions_51.ckpt")
#print(checkpointl[0]['greedy'])
#print(checkpointl[0]['target'][0][0])
checkpoint = []
for i in range(len(checkpointl)):
  for j in range(len(checkpointl[i]['target'])):
      tmp = dict()
      #print(chk)
      tmp['img_ids'] = checkpointl[i]['img_ids'][j]
      #print(' '.join(checkpointl[i]['target'][j][0]))
      #break
      tmp['target'] = checkpointl[i]['target'][j]
      tmp['greedy'] = checkpointl[i]['greedy'][j]
      checkpoint.append(tmp)

targetdict = dict()
predicteddict = dict()
for chk in checkpoint:
  if chk['img_ids'] not in targetdict:
    targetdict[chk['img_ids']] = list()
  if chk['img_ids'] not in predicteddict:
    predicteddict[chk['img_ids']] = list()
  targetdict[chk['img_ids']].append(' '.join(chk['target'][0]))
  predicteddict[chk['img_ids']].append(' '.join(chk['greedy']))

for i, j in enumerate(targetdict):
  print(j)
  print(predicteddict[j])
  if i == 10:
    break

import pickle

with open('caps_flickr.pkl', 'wb') as f:
    pickle.dump([targetdict, predicteddict], f)

checkpoint = torch.load("captions_GRU.ckpt")

targetdict = dict()
predicteddict = dict()
for chk in checkpoint:
  if chk['img_ids'] not in targetdict:
    targetdict[chk['img_ids']] = list()
  if chk['img_ids'] not in predicteddict:
    predicteddict[chk['img_ids']] = list()
  targetdict[chk['img_ids']].append(' '.join(chk['target'][0][0]))
  predicteddict[chk['img_ids']].append(' '.join(chk['greedy'][0]))

for i, j in enumerate(targetdict):
  print(j)
  print(targetdict[j])
  if i == 10:
    break

import pickle

with open('caps_GRU.pkl', 'wb') as f:
    pickle.dump([targetdict, predicteddict], f)
