The external python packages which are required to generate the output are 

`cv2`<br/>
`numpy`<br/>
[`ffmpeg-python`](https://github.com/kkroening/ffmpeg-python)<br/>


Run the below command to install the external packages.
```bash
pip install ffmpeg-python numpy opencv-python
```

To generate the output put the `Bill.jpg` and `Clinton.jpg` along with `morphing.py` and `utils.py` in a folder

To run the program, change the directory to the folder in which file is present 

```bash
python morphing.py --image1 Bush.jpg --image2 Clinton.jpg
```

Press `e` to append the source points and `f` to end appending of points for every image. 