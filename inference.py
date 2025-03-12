import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import face_alignment
from facenet_pytorch import MTCNN
import torch
import torchaudio
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw

import subprocess
import glob
from collections import deque                                                 
from skimage import transform as tf
import yaml

from look2hear.models import IIANet
from look2hear.datas.transform import get_preprocessing_pipelines
from look2hear.videomodels import ResNetVideoModel

# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform

def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped

# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):

    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:                                                
        center_y = height                                                    
    if center_y - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if center_x - width < 0:                                                 
        center_x = width                                                     
    if center_x - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if center_y + height > img.shape[0]:                                     
        center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]:                                      
        center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold:                          
        raise Exception('too much bias in width')                            
                                                                             
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

# -- RGB to GRAY
def convert_bgr2gray(data):
    return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)


def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)
    
def read_video(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()

def face2head(boxes, scale=1.5):
    new_boxes = []
    for box in boxes:
        width = box[2] - box[0]
        height= box[3] - box[1]
        width_center = (box[2] + box[0]) / 2
        height_center = (box[3] + box[1]) / 2
        square_width = int(max(width, height) * scale)
        new_box = [width_center - square_width/2, height_center - square_width/2, width_center + square_width/2, height_center + square_width/2]
        new_boxes.append(new_box)
    return new_boxes

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def detectface(video_input_path, output_path, detect_every_N_frame, scalar_face_detection, number_of_speakers):
    device = torch.device('cuda' if torch.cuda.get_device_name() else 'cpu')
    print('Running on device: {}'.format(device))
    os.makedirs(os.path.join(output_path, 'faces'), exist_ok=True)

    landmarks_dic = {}
    faces_dic = {}
    boxes_dic = {}
    
    for i in range(number_of_speakers):
        landmarks_dic[i] = []
        faces_dic[i] = []
        boxes_dic[i] = []

    mtcnn = MTCNN(keep_all=True, device=device)
    
    video = mmcv.VideoReader(video_input_path)
    print("Video statistics: ", video.width, video.height, video.resolution, video.fps)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    print('Number of frames in video: ', len(frames))
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    
    for i, frame in enumerate(frames):
        print('\rTracking frame: {}'.format(i + 1), end='')
        
        # Detect faces
        if i % detect_every_N_frame == 0:
            boxes, _ = mtcnn.detect(frame)
            boxes = boxes[:number_of_speakers]
            boxes = face2head(boxes, scalar_face_detection)
        else:
            boxes = [boxes_dic[j][-1] for j in range(number_of_speakers)]

        # Crop faces and save landmarks for each speaker
        if len(boxes) != number_of_speakers:
            boxes = [boxes_dic[j][-1] for j in range(number_of_speakers)]
        
        for j,box in enumerate(boxes):
            face = frame.crop((box[0], box[1], box[2], box[3])).resize((224,224))
            preds = fa.get_landmarks(np.array(face))
            # import pdb; pdb.set_trace()
            if i == 0:
                faces_dic[j].append(face)
                landmarks_dic[j].append(preds)
                boxes_dic[j].append(box)
            else:
                iou_scores = []
                for b_index in range(number_of_speakers):
                    last_box = boxes_dic[b_index][-1]
                    iou_score = bb_intersection_over_union(box, last_box)
                    iou_scores.append(iou_score)
                box_index = iou_scores.index(max(iou_scores))
                faces_dic[box_index].append(face)
                landmarks_dic[box_index].append(preds)
                boxes_dic[box_index].append(box)
    
    for s in range(number_of_speakers):
        frames_tracked = []
        for i, frame in enumerate(frames):
            # Draw faces
            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            draw.rectangle(boxes_dic[s][i], outline=(255, 0, 0), width=6) 
            # Add to frame list
            frames_tracked.append(frame_draw)
        dim = frames_tracked[0].size
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
        video_tracked = cv2.VideoWriter(os.path.join(output_path, 'video_tracked' + str(s+1) + '.mp4'), fourcc, 25.0, dim)
        for frame in frames_tracked:
            video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        video_tracked.release()

    # Save landmarks
    for i in range(number_of_speakers):    
        # import pdb; pdb.set_trace()
        save2npz(os.path.join(output_path, 'landmark', 'speaker' + str(i+1)+'.npz'), data=landmarks_dic[i])
        dim = face.size
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
        speaker_video = cv2.VideoWriter(os.path.join(output_path, 'faces', 'speaker' + str(i+1) + '.mp4'), fourcc, 25.0, dim)
        for frame in faces_dic[i]:
            speaker_video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        speaker_video.release()

    # Output video path
    parts = video_input_path.split('/')
    video_name = parts[-1][:-4]
    if not os.path.exists(os.path.join(output_path, 'filename_input')):
        os.mkdir(os.path.join(output_path, 'filename_input'))
    csvfile = open(os.path.join(output_path, 'filename_input', str(video_name) + '.csv'), 'w')
    for i in range(number_of_speakers):
        csvfile.write('speaker' + str(i+1)+ ',0\n')
    csvfile.close()
    return os.path.join(output_path, 'filename_input', str(video_name) + '.csv')

def crop_patch(mean_face_landmarks, video_pathname, landmarks, window_margin, start_idx, stop_idx, crop_height, crop_width, STD_SIZE=(256, 256)):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """
    
    stablePntsIDs = [33, 36, 39, 42, 45]

    frame_idx = 0
    frame_gen = read_video(video_pathname)
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == window_margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img( smoothed_landmarks[stablePntsIDs, :],
                                           mean_face_landmarks[stablePntsIDs, :],
                                           cur_frame,
                                           STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append( cut_patch( trans_frame,
                                        trans_landmarks[start_idx:stop_idx],
                                        crop_height//2,
                                        crop_width//2,))
        if frame_idx == len(landmarks)-1:
            #deal with corner case with video too short
            if len(landmarks) < window_margin:
                smoothed_landmarks = np.mean(q_landmarks, axis=0)
                cur_landmarks = q_landmarks.popleft()
                cur_frame = q_frame.popleft()

                # -- affine transformation
                trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :],
                                            mean_face_landmarks[stablePntsIDs, :],
                                            cur_frame,
                                            STD_SIZE)
                trans_landmarks = trans(cur_landmarks)
                # -- crop mouth patch
                sequence.append(cut_patch( trans_frame,
                                trans_landmarks[start_idx:stop_idx],
                                crop_height//2,
                                crop_width//2,))

            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform( trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append( cut_patch( trans_frame,
                                            trans_landmarks[start_idx:stop_idx],
                                            crop_height//2,
                                            crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None

def landmarks_interpolate(landmarks):
    
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

def crop_mouth(video_direc, landmark_direc, filename_path, save_direc, convert_gray=False, testset_only=False):
    lines = open(filename_path).read().splitlines()
    lines = list(filter(lambda x: 'test' in x, lines)) if testset_only else lines

    for filename_idx, line in enumerate(lines):

        filename, person_id = line.split(',')
        print('idx: {} \tProcessing.\t{}'.format(filename_idx, filename))

        video_pathname = os.path.join(video_direc, filename+'.mp4')
        landmarks_pathname = os.path.join(landmark_direc, filename+'.npz')
        dst_pathname = os.path.join( save_direc, filename+'.npz')

        # if os.path.exists(dst_pathname):
        #    continue

        multi_sub_landmarks = np.load(landmarks_pathname, allow_pickle=True)['data']
        landmarks = [None] * len(multi_sub_landmarks)
        for frame_idx in range(len(landmarks)):
            try:
                #landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)]['facial_landmarks'] #original for LRW
                landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)] #VOXCELEB2
            except (IndexError, TypeError):
                continue

        # -- pre-process landmarks: interpolate frames not being detected.
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        if not preprocessed_landmarks:
            continue

        # -- crop
        mean_face_landmarks = np.load('./20words_mean_face.npy')
        sequence = crop_patch(mean_face_landmarks, video_pathname, preprocessed_landmarks, 12, 48, 68, 96, 96)
        assert sequence is not None, "cannot crop from {}.".format(filename)

        # -- save
        data = convert_bgr2gray(sequence) if convert_gray else sequence[...,::-1]
        save2npz(dst_pathname, data=data)

if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    
    input_file = './test_videos/video.mp4'
    temp_output_file = './test_videos/video25fps.mp4'
    final_output_file = './test_videos/video.mp4'
    output_path = "./test_videos/video/"
    number_of_speakers = 2
    
    # subprocess.run(['ffmpeg', '-i', input_file, '-filter:v', 'fps=fps=25', temp_output_file])

    # os.rename(temp_output_file, final_output_file)

    # print(f'File has been converted and saved to {final_output_file}')
    
    # filename_path = detectface(video_input_path=final_output_file, output_path=output_path, detect_every_N_frame=8, scalar_face_detection=1.5, number_of_speakers=number_of_speakers)
    
    # # extract audio
    # subprocess.run(['ffmpeg', '-i', final_output_file, '-vn', '-ar', '16000', '-ac', '1', '-ab', '192k', '-f', 'wav', os.path.join(output_path, 'audio.wav')])
    
    # # crop mouth
    # crop_mouth(video_direc=output_path+"faces/", landmark_direc=output_path+"landmark/", filename_path=filename_path, save_direc=output_path+"mouthroi/", convert_gray=True, testset_only=False)
    
    # Load training config
    with open("checkpoints/vox2/conf.yml", "rb") as f:
        train_conf = yaml.safe_load(f)
    
    # Load model
    # print(["main_args"]["exp_dir"])
    checkpoint_path = os.path.join(train_conf["main_args"]["exp_dir"], "best_model.pth")
    audiomodel = IIANet.from_pretrain(checkpoint_path, sample_rate=train_conf["datamodule"]["data_config"]["sample_rate"], **train_conf["audionet"]["audionet_config"])
    videomodel = ResNetVideoModel(**train_conf["videonet"]["videonet_config"])
    audiomodel.cuda()
    audiomodel.eval()
    videomodel.cuda()
    videomodel.eval()
    
    with torch.no_grad():
        for i in range(number_of_speakers):
            mouth_roi = np.load(output_path+"mouthroi/speaker"+str(i+1)+".npz")["data"]
            mouth_roi = get_preprocessing_pipelines()["val"](mouth_roi)
            
            mix = torchaudio.load(output_path+"audio.wav")[0].cuda()
            
            mouth_emb = videomodel(torch.from_numpy(mouth_roi[None, None]).float().cuda())
            est_sources = audiomodel(mix[None], mouth_emb)
            
            torchaudio.save(output_path+"speaker"+str(i+1)+"_est.wav", est_sources[0].cpu(), 16000)
            
    # FFmpeg命令
    for i in range(number_of_speakers):
        command = [
            'ffmpeg',
            '-i', output_path+f"video_tracked{i+1}.mp4", 
            '-i', output_path+"speaker"+str(i+1)+"_est.wav",
            '-c:v', 'copy',         
            '-c:a', 'aac',        
            '-strict', 'experimental',
            '-map', '0:v:0',      
            '-map', '1:a:0',   
            output_path+f"s{i+1}.mp4" 
        ]
        subprocess.run(command)
