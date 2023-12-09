pwd
cd ./code
git clone https://github.com/ultralytics/yolov5
pip install -q roboflow
pip install roboflow
pip install -qr requirements.txt
cp phase1.py ./yolov5
cd yolov5
# from google.colab import drive
# drive.mount('/content/drive')
zip -r ./zipped/valid_labels.zip ./runs/detect/exp2/labels/
zip -r ./zipped/train_labels.zip ./runs/detect/exp/labels/
zip -r ./zipped/test_labels.zip ./runs/detect/exp3/labels/
zip -r ./zipped/valid_images.zip ./get_knee-3/valid/images/
zip -r ./zipped/test_images.zip ./get_knee-3/test/images/
zip -r ./zipped/train_images.zip ./get_knee-3/train/images/

python ./phase1.py
python ./detect.py --weights ./runs/train/yolov5s_results/weights/best.pt --img 460 --conf 0.4 --source ./get_knee-3/test/images --save-txt

mv /runs/detect/exp2 ../../src/p2/test_images

cd ../code
python phase2.py

mv /cropped ../src/p3/train/images

python phase3.py

