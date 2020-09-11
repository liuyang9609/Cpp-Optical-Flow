LINK.o = $(CXX) $(CXXFLAGS) $(LDFLAGS) $(TARGET_ARCH)
CXXFLAGS = -std=c++11 -g -O0 -I/anaconda3/include -Wall -fPIC
LDFLAGS= -rpath /anaconda3/lib	\
				 -L/anaconda3/lib \
				 -lopencv_core \
				 -lopencv_imgproc \
				 -lopencv_imgcodecs

SOURCES = main.cpp \
					jacobi.cpp

main: jacobi.o main.o

# link: State.o Context.o vec.o
# 	g++ -shared -std=c++11 -g -O0 -Wall -fPIC State.o Context.o -o libbutton.so

# link-static: State.o main.o
# 	ar rcs libbutton.a State.o main.o

.PHONY: clean
clean:
	rm *.o *.d

include $(subst .cpp,.d,$(SOURCES))

%.d: %.cpp
	$(CXX) -M $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
