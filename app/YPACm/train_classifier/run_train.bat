@echo off

set folder="../datasets/extracted_data/"

echo Clean dir: %folder%
cd /d %folder%
if exist . (
    del /f /q /s *.* >nul 2>&1
    for /d %%p in (*) do rmdir /q /s "%%p"
)
echo Dir %folder% cleaned.


cd "../../train_classifier"
call "../.venv/Scripts/activate.bat"

set python_interpreter="python"
color 0A
echo Start file: s1_extract_yolov8pose_skeletons.py
%python_interpreter% s1_extract_yolov8pose_skeletons.py
echo File s1_extract_yolov8pose_skeletons.py finished.
color

color 01
echo Start file: s2_combine_skeletons_txt.py
%python_interpreter% s2_combine_skeletons_txt.py
echo File s2_combine_skeletons_txt.py finished.
color

color 0A
echo Start file: s3_gen_features.py
%python_interpreter% s3_gen_features.py
echo File s3_gen_features.py finished.
color

color 01
echo Start file: s4_train_classifier.py
%python_interpreter% s4_train_classifier.py
echo File s4_train_classifier.py finished.
color

echo All files started and finished.
pause