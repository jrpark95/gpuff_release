NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_61

ifeq ($(OS),Windows_NT)
    OS_DETECTED := Windows
    PATH_SEP := /
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OS_DETECTED := Linux
        NVCCFLAGS += -Xcompiler -fPIC
        PATH_SEP := /
    endif
endif

OBJ_PATH = .$(PATH_SEP)objectfiles

TARGET = gpuff

SRCS = main.cu
OBJS = $(addprefix $(OBJ_PATH)$(PATH_SEP), $(SRCS:.cu=.o))

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

$(OBJ_PATH)$(PATH_SEP)%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
ifeq ($(OS_DETECTED),Windows)
	del /Q /F $(subst /,\,$(OBJS)) $(TARGET).exe $(TARGET).exp $(TARGET).lib
else
	rm -f $(OBJS) $(TARGET)
endif

$(OBJS): | $(OBJ_PATH)

$(OBJ_PATH):
	mkdir -p $(OBJ_PATH)
