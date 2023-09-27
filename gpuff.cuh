#pragma once

#include "gpuff_struct.cuh" 

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <math.h>
#include <limits>
#include <float.h>
#include <chrono>

#ifdef _WIN32
    #include <direct.h>
#else
    #include <sys/types.h>
    #include <sys/stat.h>
#endif

#define CHECK_METDATA 0

float time_end;
float dt;
int freq_output;
int nop;
bool isRural;
bool isPG;

__constant__ float d_time_end;
__constant__ float d_dt;
__constant__ int d_freq_output;
__constant__ int d_nop;
__constant__ bool d_isRural;
__constant__ bool d_isPG;

float etas_hgt_uv[dimZ_etas-1];
float etas_hgt_w[dimZ_etas-1];

__constant__ float d_etas_hgt_uv[dimZ_etas-1];
__constant__ float d_etas_hgt_w[dimZ_etas-1];

class Gpuff
{
private:

    PresData* device_meteorological_data_pres;
    UnisData* device_meteorological_data_unis;
    EtasData* device_meteorological_data_etas;

    std::vector<Source> sources;
    std::vector<float> decayConstants;
    std::vector<float> drydepositionVelocity;
    std::vector<Concentration> concentrations;

public:

    Gpuff();
    ~Gpuff();

    float minX, minY, maxX, maxY;
    float *d_minX, *d_minY, *d_maxX, *d_maxY;

    std::chrono::high_resolution_clock::time_point _clock0, _clock1;

    __device__ __host__ struct Puffcenter{

        float x, y, z;
        float decay_const;
        float conc;
        float age;
        float virtual_distance;
        float sigma_h;
        float sigma_z;
        float drydep_vel;

        int timeidx;
        int flag;

        Puffcenter() :
            x(0.0f), y(0.0f), z(0.0f), 
            decay_const(0.0f), 
            conc(0.0f), 
            age(0.0f), 
            virtual_distance(1e-5), 
            sigma_h(1e-5), sigma_z(1e-5), 
            drydep_vel(0.0f), timeidx(0), flag(0){}

        Puffcenter(float _x, float _y, float _z, float _decayConstant, float _concentration, float _drydep_vel, int _timeidx)  : 
            x(_x), y(_y), z(_z), 
            decay_const(_decayConstant), 
            conc(_concentration), 
            virtual_distance(1e-5), 
            sigma_h(1e-5), sigma_z(1e-5), drydep_vel(_drydep_vel),
            age(0), timeidx(_timeidx), flag(0){}

    };

    std::vector<Puffcenter> puffs;
    Puffcenter* d_puffs = nullptr;

    // gpuff_func.cuh
    void print_puffs();
    void allocate_and_copy_to_device();
    void print_device_puffs_timeidx();
    void time_update();
    void time_update_val();
    void find_minmax();
    void conc_calc();
    void conc_calc_val();
    void clock_start();
    void clock_end();

    // gpuff_init.cuh
    void read_simulation_config();
    void read_etas_altitudes();
    void puff_initialization();
    void puff_initialization_val();

    // gpuff_mdata.cuh
    float Lambert2x(float LDAPS_LAT, float LDAPS_LON);
    float Lambert2y(float LDAPS_LAT, float LDAPS_LON);
    void read_meteorological_data(
        const char* filename_pres, 
        const char* filename_unis, 
        const char* filename_etas);
 
    // gpuff_plot.cuh
    int countflag();
    void swapBytes(float& value);
    void puff_output_ASCII(int timestep);
    void puff_output_binary(int timestep);
    void grid_output_binary(RectangleGrid& grid, float* h_concs);
    void grid_output_binary_val(RectangleGrid& grid, float* h_concs);
    void grid_output_csv(RectangleGrid& grid, float* h_concs);
};

#include "gpuff_kernels.cuh" 
#include "gpuff_init.cuh" 
#include "gpuff_mdata.cuh" 
#include "gpuff_func.cuh" 
#include "gpuff_plot.cuh"
