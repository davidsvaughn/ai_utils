cd /home/david/code/davidsvaughn/ai_utils/tracking
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
the required args are:
--frames the directory where all the raw video frames are stored and
--labels the directory where all the yolo format detection files are stored
also, the code assumes that these filenames are the same, except for txt, or jpg extension
AND the code assumes that these files stick to  the abcxyz_{frame-number}.{txt,jpg} format we've gotten used to (frame number at end, before extension)
3:44
optional args are:
--N the number of candidate frames (w/detections) to return... defaults to 5
--start and --end allow you to give it starting and ending frames within which to search..... you would get these after processing those image timestamps from the pics taken between poles, that we were talking about with Sky....
3:46
--save_path is just where the output is saved.... if not specified, script will create a folder called output next to the frame_path


pip install open3d
pip install networkx


-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------

exp23

python voat.py --frames /home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/frames --labels /home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/labels --classes /home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/classes.txt 

## spyder...
main('/home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/frames',
     '/home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/labels',
     class_file='/home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/classes.txt')

-----------------------------------------------------------------------------------------
vid1
python voat.py --frames /home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/frames --labels /home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/labels --classes /home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/classes.txt 


## spyder...
main('/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/frames',
     '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/labels',
     class_file='/home/david/code/phawk/data/generic/transmission/rgb/master/data/exp23/classes.txt')

10
9.5
9.0
8.5
8.0
7.5
7.0
6.5
6.0
5.5
5.0
4.5
4.0
[11.44518272 10.44186047]
[[2480 1522]
 [4263 2448]]
[[1783  926]]
0/926
50/926
100/926
150/926
200/926
250/926
300/926
350/926
400/926
450/926
500/926
550/926
600/926
650/926
700/926
750/926
800/926
850/926
0/6499
1000/6499
2000/6499
3000/6499
4000/6499
5000/6499
6000/6499
{'span': 50, 'conn': 0.6000000000000001, 'start_frame': 1522, 'end_frame': 2448, 'step': 5, 'frame_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/frames/', 'label_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/labels/', 'trans_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/trans/', 'verbose': 1}		n=0
Loops found: no solution.
0/926
50/926
100/926
150/926
200/926
250/926
300/926
350/926
400/926
450/926
500/926
550/926
600/926
650/926
700/926
750/926
800/926
850/926
0/6363
1000/6363
2000/6363
3000/6363
4000/6363
5000/6363
6000/6363
{'span': 50, 'conn': 0.66, 'start_frame': 1522, 'end_frame': 2448, 'step': 5, 'frame_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/frames/', 'label_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/labels/', 'trans_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/trans/', 'verbose': 1}		n=0
Loops found: no solution.
0/926
50/926
100/926
150/926
200/926
250/926
300/926
350/926
400/926
450/926
500/926
550/926
600/926
650/926
700/926
750/926
800/926
850/926
0/6297
1000/6297
2000/6297
3000/6297
4000/6297
5000/6297
6000/6297
{'span': 50, 'conn': 0.7000000000000001, 'start_frame': 1522, 'end_frame': 2448, 'step': 5, 'frame_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/frames/', 'label_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/labels/', 'trans_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/trans/', 'verbose': 1}		n=0
Loops found: no solution.
0/926
50/926
100/926
150/926
200/926
250/926
300/926
350/926
400/926
450/926
500/926
550/926
600/926
650/926
700/926
750/926
800/926
850/926
0/6462
1000/6462
2000/6462
3000/6462
4000/6462
5000/6462
6000/6462
{'span': 35, 'conn': 0.66, 'start_frame': 1522, 'end_frame': 2448, 'step': 5, 'frame_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/frames/', 'label_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/labels/', 'trans_path': '/home/david/code/phawk/data/generic/transmission/rgb/master/data/H264/vid1/trans/', 'verbose': 1}		n=0
Loops found: no solution.
0/926
50/926
100/926
150/926
200/926
250/926
300/926
350/926
400/926
450/926


-----------------------------------------------------------------------------------------

