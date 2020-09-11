COMPILER=clang++
FLAGS=-Wno-c++11-extensions -Wall -flto -O3
FILES=flow_utility.cpp lodepng.cpp filehandling.cpp image_class.cpp


all:
	$(COMPILER) $(FLAGS) $(FILES) hornschunck.cpp main.cpp -o main

run:
	./main yos1.pgm yos2.pgm yos_truth.F 4 100 0.5 1.95 200 1.5

clean:
	rm main

test:
	$(COMPILER) $(FLAGS) $(FILES) test.cpp -o test

simple:
	$(COMPILER) $(FLAGS) $(FILES) hornschunck_simple.cpp main_simple_segmentation.cpp -o simple
