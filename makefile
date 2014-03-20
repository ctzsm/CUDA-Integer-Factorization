all:
	nvcc factor.cu -o factor -O3 -lcudart -lcurand
