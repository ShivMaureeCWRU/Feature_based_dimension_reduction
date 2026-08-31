import os
import pickle
import math

import cv2
import mediapipe as mp
import numpy as np

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)

DATA_DIR="../hagrid_100_data"

OUTPUT_DIR="../hagrid_100_representations"

os.makedirs(OUTPUT_DIR,exist_ok=True)

raw_xy=[]
raw_xyz=[]
translated_xy=[]
scaled_xy=[]
distances_xy=[]
normalized_distances_xy=[]
angles_xy=[]
hybrid=[]
labels=[]

total_images=0
processed_images=0
skipped_images=0

ANGLE_TRIPLES=[
    (0,1,2),(1,2,3),(2,3,4),
    (0,5,6),(5,6,7),(6,7,8),
    (0,9,10),(9,10,11),(10,11,12),
    (0,13,14),(13,14,15),(14,15,16),
    (0,17,18),(17,18,19),(18,19,20)
]

def pairwise_distances(points_xy):
    features=[]

    for a in range(21):
        for b in range(a+1,21):
            xa,ya=points_xy[a]
            xb,yb=points_xy[b]

            d=math.sqrt((xa-xb)**2+(ya-yb)**2)
            features.append(d)

    return features

def angle_features(points_xy):
    features=[]
    eps=1e-8

    for a,b,c in ANGLE_TRIPLES:
        pa=np.array(points_xy[a])
        pb=np.array(points_xy[b])
        pc=np.array(points_xy[c])

        u=pa-pb
        v=pc-pb

        denom=np.linalg.norm(u)*np.linalg.norm(v)

        if denom<eps:
            theta=0.0
        else:
            cos_theta=np.dot(u,v)/denom
            cos_theta=np.clip(cos_theta,-1.0,1.0)
            theta=math.acos(cos_theta)

        features.append(theta)

    return features

valid_extensions=(".jpg",".jpeg",".png",".bmp",".webp")

for dir_ in sorted(os.listdir(DATA_DIR)):
    class_dir=os.path.join(DATA_DIR,dir_)

    if not os.path.isdir(class_dir):
        continue

    for img_path in os.listdir(class_dir):
        if not img_path.lower().endswith(valid_extensions):
            continue

        total_images+=1

        img_file=os.path.join(class_dir,img_path)
        img=cv2.imread(img_file)

        if img is None:
            skipped_images+=1
            continue

        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            skipped_images+=1
            continue

        hand_landmarks=results.multi_hand_landmarks[0]

        x_vals=[]
        y_vals=[]
        z_vals=[]

        for landmark in hand_landmarks.landmark:
            x_vals.append(landmark.x)
            y_vals.append(landmark.y)
            z_vals.append(landmark.z)

        if len(x_vals)!=21 or len(y_vals)!=21 or len(z_vals)!=21:
            skipped_images+=1
            continue

        min_x=min(x_vals)
        min_y=min(y_vals)
        max_x=max(x_vals)
        max_y=max(y_vals)

        width=max_x-min_x
        height=max_y-min_y

        if width==0 or height==0:
            skipped_images+=1
            continue

        hand_box_diag=math.sqrt(width**2+height**2)

        if hand_box_diag==0:
            skipped_images+=1
            continue

        xy=[]

        for j in range(21):
            xy.append(x_vals[j])
            xy.append(y_vals[j])

        xyz=[]

        for j in range(21):
            xyz.append(x_vals[j])
            xyz.append(y_vals[j])
            xyz.append(z_vals[j])

        trans_xy=[]

        for j in range(21):
            trans_xy.append(x_vals[j]-min_x)
            trans_xy.append(y_vals[j]-min_y)

        scale_xy=[]

        for j in range(21):
            scale_xy.append((x_vals[j]-min_x)/width)
            scale_xy.append((y_vals[j]-min_y)/height)

        points_xy=list(zip(x_vals,y_vals))

        dist_xy=pairwise_distances(points_xy)

        norm_dist_xy=[]

        for d in dist_xy:
            norm_dist_xy.append(d/hand_box_diag)

        angle_xy=angle_features(points_xy)

        hybrid_features=scale_xy+norm_dist_xy+angle_xy

        raw_xy.append(xy)
        raw_xyz.append(xyz)
        translated_xy.append(trans_xy)
        scaled_xy.append(scale_xy)
        distances_xy.append(dist_xy)
        normalized_distances_xy.append(norm_dist_xy)
        angles_xy.append(angle_xy)
        hybrid.append(hybrid_features)
        labels.append(dir_)

        processed_images+=1

with open(os.path.join(OUTPUT_DIR,"raw_xy.pickle"),"wb") as f:
    pickle.dump({"data":raw_xy,"labels":labels},f)

with open(os.path.join(OUTPUT_DIR,"raw_xyz.pickle"),"wb") as f:
    pickle.dump({"data":raw_xyz,"labels":labels},f)

with open(os.path.join(OUTPUT_DIR,"translated_xy.pickle"),"wb") as f:
    pickle.dump({"data":translated_xy,"labels":labels},f)

with open(os.path.join(OUTPUT_DIR,"scaled_xy.pickle"),"wb") as f:
    pickle.dump({"data":scaled_xy,"labels":labels},f)

with open(os.path.join(OUTPUT_DIR,"distances_xy.pickle"),"wb") as f:
    pickle.dump({"data":distances_xy,"labels":labels},f)

with open(os.path.join(OUTPUT_DIR,"normalized_distances_xy.pickle"),"wb") as f:
    pickle.dump({"data":normalized_distances_xy,"labels":labels},f)

with open(os.path.join(OUTPUT_DIR,"angles_xy.pickle"),"wb") as f:
    pickle.dump({"data":angles_xy,"labels":labels,"angle_triples":ANGLE_TRIPLES},f)

with open(os.path.join(OUTPUT_DIR,"hybrid.pickle"),"wb") as f:
    pickle.dump({"data":hybrid,"labels":labels},f)

print("hagrid_100 representation datasets created successfully.")
print(f"Total images found: {total_images}")
print(f"Processed images: {processed_images}")
print(f"Skipped images: {skipped_images}")

if len(labels)>0:
    print(f"raw_xy dimension: {len(raw_xy[0])}")
    print(f"raw_xyz dimension: {len(raw_xyz[0])}")
    print(f"translated_xy dimension: {len(translated_xy[0])}")
    print(f"scaled_xy dimension: {len(scaled_xy[0])}")
    print(f"distances_xy dimension: {len(distances_xy[0])}")
    print(f"normalized_distances_xy dimension: {len(normalized_distances_xy[0])}")
    print(f"angles_xy dimension: {len(angles_xy[0])}")
    print(f"hybrid dimension: {len(hybrid[0])}")