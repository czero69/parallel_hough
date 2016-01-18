
#CC := g++ -O2 -g -Wall -fmessage-length=0 
CC := x86_64-apple-darwin15-g++-mp-5 -std=c++11 -std=c++0x -O3 -Wall -pedantic -fopenmp
GCC := $(CC)

INCLUDE_OPENCV := -I/opt/local/include
LIBRARY_OPENCV := -L/opt/local/lib
FLAGS_OPENCV := -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videoio -lopencv_videostab


all: bin/hough1

bin/hough%: build/EyeDescriptor.o build/main.o build/MainHough%.o
	$(CC) $(FLAGS_OPENCV) -o $@ $+ $(LIBRARY_OPENCV)

build/%.o: src/%.cpp
	$(CC) $(INCLUDE_OPENCV) $(FLAGS_OPENCV) -o $@ -c $<;


openmp%: bin/openmp%

bin/openmp%: build/openmp%.o
	$(GCC) -o $@ $+;

build/%.o: src/shortexperiments/%.cpp
	$(GCC) -o $@ -c $<;




