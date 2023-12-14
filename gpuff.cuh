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
#include <iomanip>

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

float vdepo[10] = {8.1e-4, 9.01e-4, 1.34e-3, 2.46e-3, 4.94e-3, 9.87e-3, 1.78e-2, 2.62e-2, 2.83e-2, 8.56e-2};
float size[9][10] = {
    {0.1,       0.1,        0.1,	    0.1,        0.1,        0.1,        0.1,        0.1,        0.1,        0.1},
    {0.0014895,	0.022675,	0.10765,	0.24585,	0.38418,	0.18402,	0.054143,	2.90E-09,	6.82E-10,	7.87E-13},
    {0.0012836,	0.019533,	0.091448,	0.23489,	0.4168,	    0.19204,	0.044002,	0,	        0,	        0},
    {0.0022139,	0.033732,	0.16302,	0.26315,	0.31207,	0.18978,	0.03604,	1.54E-08,	3.61E-09,	4.16E-12},
    {0.0014733,	0.022426,	0.10634,	0.24913,	0.38547,	0.18207,	0.053093,	9.69E-17,	2.28E-17,	2.62E-20},
    {0.0012479,	0.018991,	0.088684,	0.22864,	0.42114,	0.19771,	0.043585,	0,	        0,	        0},
    {0.0012604,	0.019181,	0.089856,	0.23155,	0.40825,	0.19483,	0.05508,	6.54E-19,	1.54E-19,	1.77E-22},
    {0.001017,	0.015482,	0.072548,	0.18129,	0.33135,	0.21486,	0.18346,	0,	        0,	        0},
    {0.0011339,	0.017258,	0.080758,	0.20548,	0.37511,	0.21113,	0.10913,	0,	        0,	        0}    
};

//float radi[4] = {1000000.0f/2.0f, 3000000.0f/2.0f, 5000000.0f/2.0f, 1.0e+10};
float radi[5] = {1609.0f, 16093.0f, 80467.0f, 804672.0f, 1.0e+10};
//float radi[5] = { 1609.0f, 16093.0f, 20467.0f, 34672.0f, 50000.0 };


#define RNUM 4
#define NNUM 9



float *d_vdepo;
//float (*d_size)[10];
float** d_size;

float *d_radi;

//float fd[3] = { 0.0f, };
//float fw[3] = { 0.0f, };


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

    std::vector<float> RCAP_windir;
    std::vector<float> RCAP_winvel;
    std::vector<int> RCAP_stab;

    float* d_RCAP_windir = nullptr;
    float* d_RCAP_winvel = nullptr;
    int* d_RCAP_stab = nullptr;

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

        float windvel;
        float windir;
        int stab;

        float head_dist;
        float tail_dist;
        int head_radidx;
        int tail_radidx;

        float tin[RNUM]  = { 0.0, };
        float tout[RNUM] = { 0.0, };
        //std::vector<float> tout;

        float fd[NNUM][RNUM] = { 0.0, };
        float fw[NNUM][RNUM] = { 0.0, };
        float fallout[NNUM][RNUM] = { 0.0, };

        //1_Xe-133  2_I-131   3_Cs-137  4_Te-132  5_Sr-89   6_Ru-106  7_La-140  8_Ce-144  9_Ba-140
        //float conc_arr[9] = { 1.0e+5, 1.0e+5, 1.0e+5, 1.0e+5, 1.0e+5, 1.0e+5, 1.0e+5, 1.0e+5, 1.0e+5 };
        float pn = 2000.0;
        float conc_arr[9] = { 4.849e+18f / pn, 2.292e+18f / pn, 1.728e+17f / pn, 3.330e+18f / pn,
            2.567e+18f / pn, 7.379e+17f / pn, 4.542e+18f / pn, 2.435e+18f / pn, 4.444e+18f / pn };
        
        Puffcenter() :
            x(0.0f), y(0.0f), z(0.0f), 
            decay_const(0.0f),
            conc(0.0f), 
            age(0.0f), 
            virtual_distance(1e-5), 
            sigma_h(1e-5), sigma_z(1e-5), 
            drydep_vel(0.0f), timeidx(0), flag(0),
            windvel(0.5f), windir(0), stab(1), head_radidx(0), tail_radidx(0),
            head_dist(0.0f), tail_dist(0.0f) {}

        Puffcenter(float _x, float _y, float _z, float _decayConstant, 
            float _concentration, float _drydep_vel, int _timeidx,
            float _windvel, float _windir, int _stab)  : 
            x(_x), y(_y), z(_z), 
            decay_const(_decayConstant), 
            conc(_concentration), 
            virtual_distance(1e-5), 
            sigma_h(1e-5), sigma_z(1e-5), drydep_vel(_drydep_vel),
            age(0), timeidx(_timeidx), flag(0),
            windvel(_windvel), windir(_windir), stab(_stab), head_radidx(0), tail_radidx(0),
            head_dist(0.0f), tail_dist(0.0f) {}

    };

    std::vector<Puffcenter> puffs;
    Puffcenter* d_puffs = nullptr;


    __device__ __host__ struct receptors_RCAP{
        float x, y, z;
        float conc;

        receptors_RCAP(float _x, float _y) :
        x(_x), y(_y), z(0.0f), conc(0.0f){}
    };

    std::vector<receptors_RCAP> receptors;
    receptors_RCAP* d_receptors;
    std::vector<float> con1, con2, con3;

    // gpuff_func.cuh
    void print_puffs();
    void allocate_and_copy_to_device();
    void print_device_puffs_timeidx();
    void time_update();
    void time_update_RCAP();
    void time_update_val();
    void find_minmax();
    void conc_calc();
    void conc_calc_val();
    void clock_start();
    void clock_end();
    void time_update_polar();
    //void calc_drydepot();
    //void calc_wetdepot();
    //void calc_fallout();

    // gpuff_init.cuh
    void read_simulation_config();
    void read_etas_altitudes();
    void puff_initialization();
    void puff_initialization_val();
    void puff_initialization_RCAP();
    void receptor_initialization_ldaps();

    // gpuff_mdata.cuh
    float Lambert2x(float LDAPS_LAT, float LDAPS_LON);
    float Lambert2y(float LDAPS_LAT, float LDAPS_LON);
    void read_meteorological_data(
        const char* filename_pres, 
        const char* filename_unis, 
        const char* filename_etas);
    void read_meteorological_data_RCAP();
 
    // gpuff_plot.cuh
    int countflag();
    void swapBytes(float& value);
    void swapBytes_int(int& value);
    void puff_output_ASCII(int timestep);
    void puff_output_binary(int timestep);
    void grid_output_binary(RectangleGrid& grid, float* h_concs);
    void grid_output_binary_val(RectangleGrid& grid, float* h_concs);
    void grid_output_csv(RectangleGrid& grid, float* h_concs);
    void receptor_output_binary_RCAP(int timestep);
};

#include "gpuff_kernels.cuh" 
#include "gpuff_init.cuh" 
#include "gpuff_mdata.cuh" 
#include "gpuff_func.cuh" 
#include "gpuff_plot.cuh"
