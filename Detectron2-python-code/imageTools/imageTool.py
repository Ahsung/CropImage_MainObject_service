import torch
import math
import cv2

def limitsize(img):
    rate = img.shape[0]/img.shape[1]
    y = img.shape[0]; x = img.shape[1]
    if y > x :
        flag = True
        maxlen = y
    else :
        flag = False
        maxlen = x

    if maxlen > 800:
        if flag:
            y = 800;
            x = int(y/rate)
        else :
            x= 800
            y = int(x*rate)

    dst = cv2.resize(img, dsize=(x,y), interpolation=cv2.INTER_AREA)
    return dst



def expand16_9(ratelen,maxlen,src,dst):
    # main instance가 9비율의 y축 보다 작은경우 넓힌다.
    midlen = int((src+dst)/2);
    len = dst-src;
    if len < ratelen:
        src = midlen - int(ratelen / 2);
        dst = midlen + int(ratelen / 2)
        # 넒히는 범위가 삐져나갈경우,
        if src < 0:
            dst += abs(src);
            src = 0;
        elif dst > maxlen:
            src -= dst - maxlen;
            dst = maxlen;

    # main instance가 9비율 y축보다 클 경우 줄인다.
    elif len > ratelen:
        src = max(0, src - int(len / 17))
        dst = src + ratelen;

    return src,dst;



def rate16_9(img,y_s,y_d,x_s,x_d,ratex=16,ratey=9):
    y_s,y_d,x_s,x_d = int(y_s),int(y_d),int(x_s),int(x_d)

    maxy = img.shape[0]-1; maxx = img.shape[1]-1
    maxx_16len = int(maxy*ratex/ratey); maxy_9len = int(maxx*ratey/ratex);

    # x축 비율이 짧음
    if maxx_16len > maxx:
        # x축을 풀로 잡고, y축을 맞춤
        x_s= 0; x_d = maxx;
        y_s,y_d = expand16_9(maxy_9len,maxy,y_s,y_d)

   # y축 비율이 짧음
    elif maxy_9len > maxy:
        # y축을 풀로 잡고, x축을 맞춤
        y_s = 0; y_d = maxx;
        x_s,x_d = expand16_9(maxx_16len, maxx, x_s, x_d)
    else:
        #이미 16:9비율
        return img;

    return fitsize(img,y_s,y_d,x_s,x_d)


def get_weight(outputs,im,print_weight_info_flag = False):
    boxes = outputs["instances"].pred_boxes.tensor
    pred = outputs['instances'].pred_classes
    scores = outputs["instances"].scores

    weightlist = []
    midpos = torch.tensor([int(im.shape[0]/2),int(im.shape[1]/2)])
    linsize = im.shape[0] * im.shape[1]
    for i in range(scores.shape[0]):

        #중심에서 가까울수록 점수
        instancePos = torch.tensor([ int((boxes[i][3] + boxes[i][1]) / 2), int((boxes[i][2] + boxes[i][0])/2) ]);
        dist = torch.sum((midpos-instancePos) ** 2)
        center = int(linsize/dist+1)

        pred_Kind = math.log(pred[i]*2+1)+10
        if pred[i] == 0 : pred_Kind = math.log(pred[i]*2+1)+5

        #weight = 박스크기 * 점수^2 / 종류*2+10,,(사람 = 0) * log(center 점수)

        box_size = abs(boxes[i][2] - boxes[i][0])*abs(boxes[i][3]-boxes[i][1])

        weight = box_size*(scores[i]**1.5)*(math.log(center)+5)/(pred_Kind)

        # print weight 정보
        if print_weight_info_flag:
            print(i,":  weight:",weight, " boxsize:",box_size,"score",scores[i]**1.5,"center",math.log(center)+5,"pred",pred_Kind,"pos:",instancePos,midpos)
        weightlist.append(int(weight))

    if weightlist != []:
        idx = weightlist.index(max(weightlist))
    else : idx = -1
    weightlist = torch.tensor(weightlist)
    return idx, weightlist

#x,y축을 찾아내서 이미지 slice
def fitsize(im,y_s,y_d,x_s,x_d):
    y_s,y_d,x_s,x_d = int(y_s),int(y_d),int(x_s),int(x_d)
    return im[y_s:y_d,x_s:x_d]


#직선 축이 겹치는지 확인
def checklinear(x1,x2,y1,y2):
    #더 앞에 있는 직선을 골라냄.
    if x1 <= y1:
        front = [x1,x2]; back = [y1,y2]
    else:
        front = [y1,y2]; back = [x1,x2]

    #더 뒤에 있는 직선의 앞부분이 앞의 직선 끝보다 앞에있다면 둘은 겹침.
    if back[0] < front[1]: return True

    return False

#가까운 정도 ( 겹치거나,  이미지의 1/10 만큼 가깝다면 True )큰
def closelinear(m1, m2, add1, add2, maxline):
    if checklinear(m1, m2, add1, add2) : return True
    if (m1 > add2 and abs(m1-add2) * 10 < maxline) or (add1 > m2 and abs(add1 - m2)*10 < maxline) : return True
    else: return False


#weight가 너무 차이나면 바로 무시슬
#겹치는 Box크기가 작은 이미지의 1/5만큼 겹치면 return
# 축 하나가 lab_lin_lr 비율만큼 겹치고, 충분히 가깝다면 1 return
def getLapBox(main, add, m_w, a_w, lap_wide_lr=5, lap_lin_lr=0.7):
    #weight가 7배 이상 차이나면 바로 캔슬
    if 7*a_w <= m_w :
       return -1

    x_s = max(main[0],add[0]); x_d = min(main[2],add[2])
    y_s = max(main[1],add[1]); y_d = min(main[3],add[3])

    if not checklinear(main[0], main[2], add[0], add[2]):
        x_d = 0; x_s=0
    if not checklinear(main[1], main[3], add[1], add[3]):
        y_d = 0; y_s = 0

    main_wide = (main[2] - main[0])*(main[3]-main[1])
    add_wide = (add[2] - add[0])*(add[3]-add[1])
    overLab_wide = (x_d - x_s)*(y_d - y_s)

    shortY = min(main[3]-main[1],add[3]-add[1])
    shortX = min(main[2]-main[0],add[2]-add[0])

    longY = max(main[3] - main[1], add[3] - add[1])
    longX = max(main[2] - main[0], add[2] - add[0])


    if lap_wide_lr*overLab_wide > min(main_wide, add_wide) :
        return int(overLab_wide)
    elif (y_d-y_s) >= shortY*lap_lin_lr and closelinear(main[0], main[2], add[0], add[2], longX) :
        return 1
    elif (x_d-x_s) >= shortX*lap_lin_lr and closelinear(main[1], main[3], add[1], add[3], longY) :
        return 1

    else : return -1

#main을 기준으로 getLapBox기준에 알맞는 instance들의 index를  tensor 형태로 반환
def getconInstances(boxes,idx,weightlist,lab_wide_lr=6,lab_lin_lr=0.7):
    BOX_index = []
    for i in range(boxes.shape[0]):
        # print(i,"번째 시작")
        if getLapBox(boxes[idx],boxes[i],weightlist[idx],weightlist[i],lab_wide_lr,lab_lin_lr) != -1 :
            BOX_index.append(i)
    BOX_index = torch.tensor(BOX_index)
    return BOX_index

#
def edgeSearch(mask_line,flag):
    maskArry = torch.where(mask_line == True)[0]
    if maskArry.size() == torch.Size([0]):
        return -1
    if flag :
        src = maskArry[0]+1
    else : src = maskArry[-1]-1
    return src


def expandMask(masks, size, y_s, y_d, x_s, x_d):
    if size == 0 : return masks;
    mask = masks.clone()
    masktemp = masks.clone()
    y_s,y_d,x_s,x_d = int(y_s),int(y_d),int(x_s),int(x_d)
    for i in range(y_s,y_d):
        leftx = edgeSearch(masktemp[i],True)
        rightx = edgeSearch(masktemp[i],False)
        if leftx == -1 :
            continue
        mleft = max(leftx-size,0)
        mright = min(rightx+size,mask.shape[1])
        mask[i][mleft:leftx] = True    #왼쪽으로 쭉
        mask[i][rightx:mright] = True  #오른쪽으로 쭉

    masktemp = masks.clone()
    for i in range(x_s,x_d):
        upy = edgeSearch(masktemp.T[i],True)
        downy = edgeSearch(masktemp.T[i],False)

        if upy == -1 :
            continue
        mup = max(upy-size,0)
        mdown = min(downy+size,mask.shape[0])
        mask.T[i][mup:upy] = True     #위로 쭉
        mask.T[i][downy:mdown] = True #아래로 쭉

    return mask

def combinde_img_box(conlist_boxes):
    tmps = torch.min(conlist_boxes.T[0:2], axis=1)
    tmpd = torch.max(conlist_boxes.T[2:], axis=1)

    X_S, Y_S = tmps.values
    X_D, Y_D = tmpd.values

    return int(Y_S),int(Y_D),int(X_S),int(X_D)

def combine_img_mask(conlist_masks):
    combineMask = conlist_masks[0].clone()
    conlist_mask_index = torch.where(conlist_masks == True)
    combineMask[conlist_mask_index[1:]] = True

    return combineMask

def rmBg(im,mask,size,y_s,y_d,x_s,x_d):
    rmbgIm = torch.from_numpy(im.copy())
    emask = expandMask(mask,size, y_s, y_d, x_s, x_d)
    ebe = torch.where(emask != True)
    rmbgIm[ebe] = 224
    rmbgIm = rmbgIm.numpy()

    return rmbgIm
