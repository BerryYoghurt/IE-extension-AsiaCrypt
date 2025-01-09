from PIL import Image
import torch
import clip
"""CLIP"""
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
cos = torch.nn.CosineSimilarity(dim=0)


"""computing similarity using clip as in https://medium.com/@jeremy-k/unlocking-openai-clip-part-2-image-similarity-bf0224ab5bb0
im_1, im_2: two images
"""
def clip_feature(im):
    im_preprocess = preprocess(im).unsqueeze(0).to(device)
    im_features = model.encode_image( im_preprocess )
    ret = im_features[0].cpu()
    # print(ret)
    #print(len(im_features[0]))
    return ret

def cos_sim(feature_1, feature_2, indices = None):
    if indices == None:
        similarity = cos(feature_1, feature_2).item()
    else:
        similarity = cos(feature_1[indices],feature_2[indices]).item()
    # similarity = (similarity+1)/2
    return similarity
    
    
def get_feature(pt,fn):

    im = Image.fromarray(pt,mode='RGB')
    im.save(fn,quality=90)
    feature = clip_feature(im)
    return feature
