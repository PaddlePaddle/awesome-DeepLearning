""" extract frames and pcm"""
import os
import sys
import shutil


def ffmpeg_frames(mp4_addr, frame_out_folder, fps=5):
    """ffmpeg_frames"""
    if os.path.exists(frame_out_folder):
        shutil.rmtree(frame_out_folder)
    os.makedirs(frame_out_folder)
    cmd = './src/utils/ffmpeg -v 0 -i %s -r %d -q 0 %s/%s.jpg' % (mp4_addr, fps, frame_out_folder, '%08d')
    os.system(cmd)


def ffmpeg_pcm(mp4_addr, save_file_name):
    """ffmpeg_pcm"""
    cmd = './src/utils/ffmpeg -y  -i %s  -acodec pcm_s16le -f s16le -ac 1 -ar 16000 %s -v 0' \
        % (mp4_addr, save_file_name)
    os.system(cmd)


def ffmpeg_mp4(mp4_url, mp4_addr):
    """ffmpeg_mp4"""
    cmd = "wget %s -O %s -q" % (mp4_url, mp4_addr)
    print ("cmd = ", cmd)
    os.system(cmd)


def get_images(image_path):
    """get_images"""
    images = sorted(os.listdir(image_path))
    images = images
    images_path_list = [image_path + '/' + im for im in images]
    return images_path_list

