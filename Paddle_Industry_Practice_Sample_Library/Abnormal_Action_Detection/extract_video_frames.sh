IN_DATA_DIR=$1
OUT_DATA_DIR=$2
FPS = $3

echo "输入视频文件夹$1"
echo "输出视频帧文件夹$2"
echo "帧率$3"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

echo "遍历根目录"
for video in $(ls -d -U ${IN_DATA_DIR}/*); do
    echo $video 
    video_name=${video##*/} 
    name="${video_name%.*}" 

    out_video_dir=${OUT_DATA_DIR}/$name/
    echo $out_video_dir 
    mkdir -p "${out_video_dir}/"

    out_name="${out_video_dir}/%05d.jpg"

    ffmpeg -i "${video}" -r $3 -q:v 1 "${out_name}"
done