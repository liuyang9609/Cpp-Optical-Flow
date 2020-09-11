COMPILER=clang++
FLAGS=-Wno-c++11-extensions -Wall -flto -O3
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc

hornschunck:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp shared/misc.cpp shared/tensor_computation.cpp shared/flow_computation.cpp standart/hornschunck/hornschunck.cpp -o hornschunck

brox_iso:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp shared/misc.cpp shared/tensor_computation.cpp shared/flow_computation.cpp standart/brox_iso/brox_iso.cpp -o brox_iso

brox_aniso:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp shared/misc.cpp shared/tensor_computation.cpp shared/flow_computation.cpp standart/brox_aniso/brox_aniso.cpp -o brox_aniso

hornschunck_separation:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp shared/misc.cpp shared/initial_separation.cpp shared/flow_computation.cpp shared/tensor_computation.cpp levelset/hornschunck/smooth/hornschunck_separation.cpp -o hornschunck_separation

hornschunck_separation_constant:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp shared/misc.cpp shared/initial_separation.cpp shared/flow_computation.cpp shared/tensor_computation.cpp levelset/hornschunck/constant/hornschunck_separation_constant.cpp -o hornschunck_separation_constant

brox_iso_separation:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp shared/misc.cpp shared/initial_separation.cpp shared/tensor_computation.cpp shared/flow_computation.cpp levelset/brox_iso/smooth/brox_iso_separation.cpp -o brox_iso_separation

brox_aniso_separation:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp shared/misc.cpp shared/initial_separation.cpp shared/tensor_computation.cpp shared/flow_computation.cpp levelset/brox_aniso/smooth/brox_aniso_separation.cpp -o brox_aniso_separation

brox_iso_separation_constant:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp shared/misc.cpp shared/initial_separation.cpp shared/tensor_computation.cpp shared/flow_computation.cpp levelset/brox_iso/constant/brox_iso_separation_constant.cpp -o brox_iso_separation_constant

brox_aniso_separation_constant:
	$(COMPILER) $(FLAGS) $(LIBS) main.cpp shared/misc.cpp shared/initial_separation.cpp shared/tensor_computation.cpp shared/flow_computation.cpp levelset/brox_aniso/constant/brox_aniso_separation_constant.cpp -o brox_aniso_separation_constant

scenario:
	$(COMPILER) $(FLAGS) $(LIBS) create_test_case.cpp shared/misc.cpp shared/tensor_computation.cpp shared/flow_computation.cpp -o scenario 
