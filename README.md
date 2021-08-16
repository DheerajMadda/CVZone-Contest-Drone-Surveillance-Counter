# CVZone-Contest-Drone-Surveillance-Counter

Here I implemented a car counter according to this competition: https://www.computervision.zone/dsc/. I used the Yolo5+deepsort repo (https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch), trained on the given video as a dataset, and modified the code a bit for the purpose of the contest (track.py from line 210-233).

The modifications are the following:

- I store the detected ID-s in an array, and I append those which was not in the array to count the cars.
- Sometimes there are similar cars, so the algorithm detects them the same as previous cars, so it gives the same ID to a different car. If this happens I multiply the ID with 100 to distinguish it from the previous car and I count it as a new one.
- If cars slowly disappear from the screen at the top, sometimes the algorithm detects them as new cars, therefore after 10 frames, I only count those cars which are detected in the bottom half of the screen to ensure that I do not recount cars.

Usage:
#### python main.py --source DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4 --weights weights/best.pt --save-txt
