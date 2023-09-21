NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_86

OBJ_PATH = ./objectfiles

TARGET = gpuff

SRCS = main.cu
OBJS = $(addprefix $(OBJ_PATH)/, $(SRCS:.cu=.o))

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

$(OBJ_PATH)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

$(OBJS): | $(OBJ_PATH)

$(OBJ_PATH):
	mkdir -p $(OBJ_PATH)
