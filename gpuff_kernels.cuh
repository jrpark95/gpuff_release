#include "gpuff.cuh"

__device__ float Dynamic_viscosity_Sutherland(float temp){

    float mu_ref = 1.716e-5;    // Reference viscosity [Pa*s] // 1.827e-5
    float T_ref = 273.15;       // Reference temperature [K] // 291.15
    float S = 110.4;            // Sutherland temperature [K] // 120.0
    
    float mu = mu_ref * pow(temp/T_ref, 1.5)*(T_ref + S)/(temp + S);

    return mu;
}

__device__ float atomicMinFloat(float* address, float val){
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    while(val < __int_as_float(old)){
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(val));
    }
    return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* address, float val){
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    while(val > __int_as_float(old)){
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(val));
    }
    return __int_as_float(old);
}

__device__ float Sigma_h_Pasquill_Gifford(int PasquillCategory, float virtual_distance){

    float coefficient0[7] = {-1.104, -1.634, -2.054, -2.555, -2.754, -3.143, -3.143};
    float coefficient1[7] = {0.9878, 1.0350, 1.0231, 1.0423, 1.0106, 1.0418, 1.0418};
    float coefficient2[7] = {-0.0076, -0.0096, -0.0076, -0.0087, -0.0064, -0.0070, -0.0070};

    float sigma = exp(coefficient0[PasquillCategory] + 
                    coefficient1[PasquillCategory]*log(virtual_distance) + 
                    coefficient2[PasquillCategory]*log(virtual_distance)*log(virtual_distance));

    return sigma;
}

__device__ float dSh_PG(int PasquillCategory, float virtual_distance){

    float coefficient0[7] = {-1.104, -1.634, -2.054, -2.555, -2.754, -3.143, -3.143};
    float coefficient1[7] = {0.9878, 1.0350, 1.0231, 1.0423, 1.0106, 1.0418, 1.0418};
    float coefficient2[7] = {-0.0076, -0.0096, -0.0076, -0.0087, -0.0064, -0.0070, -0.0070};

    float sigma = pow(virtual_distance, coefficient1[PasquillCategory]-1)
                    *exp(coefficient0[PasquillCategory]+coefficient2[PasquillCategory]*log(virtual_distance)*log(virtual_distance))
                    *(coefficient1[PasquillCategory]+2*coefficient2[PasquillCategory]*log(virtual_distance));

    return sigma;
}

__device__ float Sigma_z_Pasquill_Gifford(int PasquillCategory, float virtual_distance){

    float coefficient0[7] = {4.679, -1.999, -2.341, -3.186, -3.783, -4.490, -4.490};
    float coefficient1[7] = {-1.172, 0.8752, 0.9477, 1.1737, 1.3010, 1.4024, 1.4024};
    float coefficient2[7] = {0.2770, 0.0136, -0.0020, -0.0316, -0.0450, -0.0540, -0.0540};

    float sigma = exp(coefficient0[PasquillCategory] + 
                    coefficient1[PasquillCategory]*log(virtual_distance) + 
                    coefficient2[PasquillCategory]*log(virtual_distance)*log(virtual_distance));

    return sigma;
}

__device__ float dSz_PG(int PasquillCategory, float virtual_distance){

    float coefficient0[7] = {4.679, -1.999, -2.341, -3.186, -3.783, -4.490, -4.490};
    float coefficient1[7] = {-1.172, 0.8752, 0.9477, 1.1737, 1.3010, 1.4024, 1.4024};
    float coefficient2[7] = {0.2770, 0.0136, -0.0020, -0.0316, -0.0450, -0.0540, -0.0540};

    float sigma = pow(virtual_distance, coefficient1[PasquillCategory]-1)
                    *exp(coefficient0[PasquillCategory]+coefficient2[PasquillCategory]*log(virtual_distance)*log(virtual_distance))
                    *(coefficient1[PasquillCategory]+2*coefficient2[PasquillCategory]*log(virtual_distance));

    return sigma;
}

__device__ float Sigma_h_Briggs_McElroy_Pooler(int PasquillCategory, float virtual_distance){

    float coefficient0_rural[7] = {0.22, 0.16, 0.11, 0.08, 0.06, 0.04, 0.04};
    float coefficient0_urban[7] = {0.32, 0.32, 0.22, 0.16, 0.11, 0.11, 0.11};
    float coefficient1_rural = 0.0001;
    float coefficient1_urban = 0.0004;

    float sigma;
    
    if(d_isRural) sigma = coefficient0_rural[PasquillCategory]*virtual_distance
                            *pow(1 + coefficient1_rural*virtual_distance, -0.5);

    else sigma = coefficient0_urban[PasquillCategory]*virtual_distance
                    *pow(1 + coefficient1_urban*virtual_distance, -0.5);

    return sigma;
}

__device__ float dSh_BMP(int PasquillCategory, float virtual_distance){

    float coefficient0_rural[7] = {0.22, 0.16, 0.11, 0.08, 0.06, 0.04, 0.04};
    float coefficient0_urban[7] = {0.32, 0.32, 0.22, 0.16, 0.11, 0.11, 0.11};
    float coefficient1_rural = 0.0001;
    float coefficient1_urban = 0.0004;

    float sigma;
    
    if(d_isRural) sigma = 0.5*coefficient0_rural[PasquillCategory]
                            *(coefficient1_rural*virtual_distance+2)
                            /pow(coefficient1_rural*virtual_distance+1,1.5);

    else sigma = 0.5*coefficient0_urban[PasquillCategory]
                    *(coefficient1_urban*virtual_distance+2)
                    /pow(coefficient1_urban*virtual_distance+1,1.5);

    return sigma;
}

__device__ float Sigma_z_Briggs_McElroy_Pooler(int PasquillCategory, float virtual_distance){

    float coefficient0_rural[7] = {0.20, 0.12, 0.08, 0.06, 0.03, 0.016, 0.016};
    float coefficient1_rural[7] = {0.0, 0.0, 0.0002, 0.0015, 0.0003, 0.0003, 0.0003};
    float coefficient2_rural[7] = {1.0, 1.0, -0.5, -0.5, -1.0, -1.0, -1.0};

    float coefficient0_urban[7] = {0.24, 0.24, 0.2, 0.14, 0.08, 0.08, 0.08};
    float coefficient1_urban[7] = {0.001, 0.001, 0.0, 0.0003, 0.00015, 0.00015, 0.00015};
    float coefficient2_urban[7] = {0.5, 0.5, 1.0, -0.5, -0.5, -0.5, -0.5};


    float sigma;
    
    if(d_isRural) sigma = coefficient0_rural[PasquillCategory]*virtual_distance*
                            pow(1 + coefficient1_rural[PasquillCategory]*virtual_distance, coefficient2_rural[PasquillCategory]);

    else sigma = coefficient0_urban[PasquillCategory]*virtual_distance*
                    pow(1 + coefficient1_urban[PasquillCategory]*virtual_distance, coefficient2_urban[PasquillCategory]);

    return sigma;
}

__device__ float dSz_BMP(int PasquillCategory, float virtual_distance){

    float coefficient0_rural[7] = {0.20, 0.12, 0.08, 0.06, 0.03, 0.016, 0.016};
    float coefficient1_rural[7] = {0.0, 0.0, 0.0002, 0.0015, 0.0003, 0.0003, 0.0003};
    float coefficient2_rural[7] = {1.0, 1.0, -0.5, -0.5, -1.0, -1.0, -1.0};

    float coefficient0_urban[7] = {0.24, 0.24, 0.2, 0.14, 0.08, 0.08, 0.08};
    float coefficient1_urban[7] = {0.001, 0.001, 0.0, 0.0003, 0.00015, 0.00015, 0.00015};
    float coefficient2_urban[7] = {0.5, 0.5, 1.0, -0.5, -0.5, -0.5, -0.5};

    float sigma;
    
    if(d_isRural) sigma = pow(coefficient1_rural[PasquillCategory]*virtual_distance+1, coefficient2_rural[PasquillCategory]-1)*
                            (coefficient0_rural[PasquillCategory]*coefficient1_rural[PasquillCategory]*
                            (coefficient2_rural[PasquillCategory]+1)*virtual_distance+coefficient0_rural[PasquillCategory]);

    else sigma = pow(coefficient1_urban[PasquillCategory]*virtual_distance+1, coefficient2_urban[PasquillCategory]-1)*
                    (coefficient0_urban[PasquillCategory]*coefficient1_urban[PasquillCategory]*
                    (coefficient2_urban[PasquillCategory]+1)*virtual_distance+coefficient0_urban[PasquillCategory]);

    return sigma;
}

__device__ float NewtonRaphson_h(int PasquillCategory, float targetSigma, float init){

    float x = init, fx, dfx;
    int loops = 0;

    while(1){
        if(d_isPG){
            fx = Sigma_h_Pasquill_Gifford(PasquillCategory, x) - targetSigma;
            dfx = dSh_PG(PasquillCategory, x);
        }
        else{
            fx = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, x) - targetSigma;
            dfx = dSh_BMP(PasquillCategory, x);
        }
        x = x - fx / dfx;
        if (fabs(fx) < 1e-4){
            //printf("%d ", loops);
            break;
        }
        //printf("%f, %f, %f\n", Sigma_h_Pasquill_Gifford(PasquillCategory, x), targetSigma, fx);
        loops++;
    }

    return x;
}

__device__ float NewtonRaphson_z(int PasquillCategory, float targetSigma, float init){

    float x = init, fx, dfx;
    int loops = 0;

    while(1){
        if(d_isPG){
            fx = Sigma_z_Pasquill_Gifford(PasquillCategory, x) - targetSigma;
            dfx = dSz_PG(PasquillCategory, x);
        }
        else{
            fx = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, x) - targetSigma;
            dfx = dSz_BMP(PasquillCategory, x);
        }
        x = x - fx / dfx;
        if (fabs(fx) < 1e-4){
            //printf("%d ", loops);
            break;
        }
        //printf("%f, %f, %f\n", Sigma_z_Pasquill_Gifford(PasquillCategory, x), targetSigma, fx);
        loops++;
    }

    return x;
}


__global__ void checkValueKernel(){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Value of d_nop inside kernel: %d\n", d_nop);
    }
}

__global__ void print_timeidx(Gpuff::Puffcenter* d_puffs)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("%d\n", d_nop);
    // printf("%f\n", d_dt);
    // printf("%d\n", d_freq_output);
    // printf("%f\n", d_time_end);

    if (tid < d_nop){
        printf("Timeidx of puff %d: %f\n", tid, d_puffs[tid].y/1500.0);
    }


}

__global__ void update_puff_flags(
    Gpuff::Puffcenter* d_puffs, float activationRatio) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];

    if (idx < int(d_nop * activationRatio)){
        p.flag = 1;
    }
    //if(idx==0)printf("actnum=%d\n", int(d_nop * activationRatio));
}

__global__ void move_puffs_by_wind(
    Gpuff::Puffcenter* d_puffs, 
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    int xidx = int(p.x/1500.0);
    int yidx = int(p.y/1500.0);
    int zidx_uv = 1;
    int zidx_w = 1;

    for(int i=0; i<dimZ_etas-1; i++){
        if(p.z<d_etas_hgt_uv[i]){
            zidx_uv=i+1;
            break;
        }
    }

    for(int i=0; i<dimZ_etas-1; i++){
        if(p.z<d_etas_hgt_w[i]){
            zidx_w=i+1;
            break;
        }
    }

    if(zidx_uv<0) {
        printf("Invalid zidx_uv error.\n");
        zidx_uv = 1;
    }

    if(zidx_w<0) {
        printf("Invalid zidx_w error.\n");
        zidx_w = 1;
    }

    float x0 = p.x/1500.0-xidx;
    float y0 = p.y/1500.0-yidx;

    float x1 = 1-x0;
    float y1 = 1-y0;

    float xwind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].UGRD
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].UGRD
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].UGRD
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].UGRD;

    float ywind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].VGRD
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].VGRD
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].VGRD
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].VGRD;

    float zwind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_w].DZDT
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_w].DZDT
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_w].DZDT
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_w].DZDT;

    p.x += xwind*d_dt;
    p.y += ywind*d_dt;
    p.z += zwind*d_dt;

    if(p.z<2.0) p.z=2.0;
}

__global__ void dry_deposition(
    Gpuff::Puffcenter* d_puffs, 
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    int xidx = int(p.x/1500.0);
    int yidx = int(p.y/1500.0);

    float x0 = p.x/1500.0-xidx;
    float y0 = p.y/1500.0-yidx;

    float x1 = 1-x0;
    float y1 = 1-y0;

    float mixing_height = x1*y1*device_meteorological_data_unis[xidx*(dimY) + yidx].HPBL
                            +x0*y1*device_meteorological_data_unis[(xidx+1)*(dimY) + yidx].HPBL
                            +x1*y0*device_meteorological_data_unis[xidx*(dimY) + (yidx+1)].HPBL
                            +x0*y0*device_meteorological_data_unis[(xidx+1)*(dimY) + (yidx+1)].HPBL;

    p.conc *= exp(-p.drydep_vel*d_dt/mixing_height);

}

__global__ void wet_scavenging(
    Gpuff::Puffcenter* d_puffs, 
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    int xidx = int(p.x/1500.0);
    int yidx = int(p.y/1500.0);
    int zidx_pres = 1;

    for(int i=0; i<dimZ_pres-1; i++){
        if(p.z<device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + yidx*(dimZ_pres) + i].HGT){
            zidx_pres=i+1;
            break;
        }
    }

    if(zidx_pres<0) {
        printf("Invalid zidx_pres error.\n");
        zidx_pres = 1;
    }

    float x0 = p.x/1500.0-xidx;
    float y0 = p.y/1500.0-yidx;

    float x1 = 1-x0;
    float y1 = 1-y0;

    float Relh = x1*y1*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + yidx*(dimZ_pres) + zidx_pres].RH
                +x0*y1*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + yidx*(dimZ_pres) + zidx_pres].RH
                +x1*y0*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres) + zidx_pres].RH
                +x0*y0*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres) + zidx_pres].RH;

    float Lambda = 3.5e-5*(Relh-0.8)/(1.0-0.8);

    if(Relh>0.8) p.conc *= exp(-Lambda*d_dt);

}

__global__ void radioactive_decay(
    Gpuff::Puffcenter* d_puffs, 
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    p.conc *= exp(-p.decay_const*d_dt);

}

__global__ void puff_dispersion_update(
    Gpuff::Puffcenter* d_puffs, 
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    int xidx = int(p.x/1500.0);
    int yidx = int(p.y/1500.0);
    int zidx_uv = 1;
    int zidx_w = 1;

    for(int i=0; i<dimZ_etas-1; i++){
        if(p.z<d_etas_hgt_uv[i]){
            zidx_uv=i+1;
            break;
        }
    }

    for(int i=0; i<dimZ_etas-1; i++){
        if(p.z<d_etas_hgt_w[i]){
            zidx_w=i+1;
            break;
        }
    }

    if(zidx_uv<0) {
        printf("Invalid zidx_uv error.\n");
        zidx_uv = 1;
    }

    if(zidx_w<0) {
        printf("Invalid zidx_w error.\n");
        zidx_w = 1;
    }

    float x0 = p.x/1500.0-xidx;
    float y0 = p.y/1500.0-yidx;

    float x1 = 1-x0;
    float y1 = 1-y0;

    float xwind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].UGRD
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].UGRD
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].UGRD
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].UGRD;

    float ywind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].VGRD
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].VGRD
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].VGRD
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].VGRD;

    float zwind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_w].DZDT
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_w].DZDT
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_w].DZDT
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_w].DZDT;


    float vel = sqrt(xwind*xwind + ywind*ywind + zwind*zwind);

    //printf("zwind: %f, vel: %f ", zwind, vel);

    float t0 = x1*y1*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].TMP
                +x0*y1*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].TMP
                +x1*y0*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].TMP
                +x0*y0*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].TMP;

    float tu = x1*y1*device_meteorological_data_unis[xidx*(dimY) + yidx].T1P5
                +x0*y1*device_meteorological_data_unis[(xidx+1)*(dimY) + yidx].T1P5
                +x1*y0*device_meteorological_data_unis[xidx*(dimY) + (yidx+1)].T1P5
                +x0*y0*device_meteorological_data_unis[(xidx+1)*(dimY) + (yidx+1)].T1P5;

    float gph0 = x1*y1*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].HGT
                +x0*y1*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].HGT
                +x1*y0*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].HGT
                +x0*y0*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].HGT;

    float dtp100m = 100.0*(t0-tu)/(gph0-1.5);

    int PasquillCategory = 0;

    if(dtp100m < -1.9) PasquillCategory = 0;        // A: Extremely unstable
    else if(dtp100m < -1.7) PasquillCategory = 1;   // B: Moderately unstable
    else if(dtp100m < -1.5) PasquillCategory = 2;   // C: Slightly unstable
    else if(dtp100m < -0.5) PasquillCategory = 3;   // D: Neutral
    else if(dtp100m < 1.5) PasquillCategory = 4;    // E: Slightly stable
    else if(dtp100m < 4.0) PasquillCategory = 5;    // F: Moderately stable
    else PasquillCategory = 6;                      // G: Extremely stable

    float new_virtual_distance_h = NewtonRaphson_h(PasquillCategory, p.sigma_h, p.virtual_distance) + vel*d_dt;
    float new_virtual_distance_z = NewtonRaphson_z(PasquillCategory, p.sigma_z, p.virtual_distance) + vel*d_dt;

    if(d_isPG){
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
        p.sigma_z = Sigma_z_Pasquill_Gifford(PasquillCategory, new_virtual_distance_z);
    }
    else{
        p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
        p.sigma_z = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_z);
    }

    p.virtual_distance = new_virtual_distance_h;

}

__global__ void findMinMax(
    Gpuff::Puffcenter* d_puffs, 
    float* d_minX, float* d_minY, 
    float* d_maxX, float* d_maxY)
{
    
    extern __shared__ float sharedData[];
    float* s_minX = sharedData;
    float* s_minY = &sharedData[blockDim.x];
    float* s_maxX = &sharedData[2 * blockDim.x];
    float* s_maxY = &sharedData[3 * blockDim.x];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    s_minX[tid] = (index < d_nop) ? d_puffs[index].x : FLT_MAX;
    s_minY[tid] = (index < d_nop) ? d_puffs[index].y : FLT_MAX;
    s_maxX[tid] = (index < d_nop) ? d_puffs[index].x : -FLT_MAX;
    s_maxY[tid] = (index < d_nop) ? d_puffs[index].y : -FLT_MAX;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            s_minX[tid] = fminf(s_minX[tid], s_minX[tid + s]);
            s_minY[tid] = fminf(s_minY[tid], s_minY[tid + s]);
            s_maxX[tid] = fmaxf(s_maxX[tid], s_maxX[tid + s]);
            s_maxY[tid] = fmaxf(s_maxY[tid], s_maxY[tid + s]);
        }
        __syncthreads();
    }

    if(tid == 0){
        atomicMinFloat(d_minX, s_minX[0]);
        atomicMinFloat(d_minY, s_minY[0]);
        atomicMaxFloat(d_maxX, s_maxX[0]);
        atomicMaxFloat(d_maxY, s_maxY[0]);
    }
}

// __global__ void accumulateConc(
//     Gpuff::Puffcenter* puffs, 
//     RectangleGrid::GridPoint* d_grid, 
//     float* concs, 
//     int ngrid)
// {
//     int puffIdx = blockIdx.y * blockDim.y + threadIdx.y;
//     int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (puffIdx >= d_nop || gridIdx >= ngrid) return;

//     Gpuff::Puffcenter& p = puffs[puffIdx];
//     RectangleGrid::GridPoint& g = d_grid[gridIdx];

//     printf("%f, %f\n", p.x, g.y);

//     if (puff.flag){
//         float dx = g.x - p.x;
//         float dy = g.y - p.y;
//         float distSq = dx * dx + dy * dy;

//         if (distSq != 0.0f){
//             float contribution = 1.0f / distSq;
//             atomicAdd(&concs[gridIdx], contribution);
//         }
//     }
// }

__global__ void accumulate_conc(
    Gpuff::Puffcenter* puffs, 
    RectangleGrid::GridPoint* d_grid, 
    float* concs, 
    int ngrid)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridIdx = globalIdx % ngrid;
    int puffIdx = globalIdx / ngrid;

    if(puffIdx >= d_nop) return;

    Gpuff::Puffcenter& p = puffs[puffIdx];
    RectangleGrid::GridPoint& g = d_grid[gridIdx];

    if(p.flag){
        float dx = g.x - p.x;
        float dy = g.y - p.y;
        float dz = g.z - p.z;
        float dzv = g.z + p.z;

        if(p.sigma_h != 0.0f && p.sigma_z != 0.0f){
            float contribution = p.conc/(pow(2*PI,1.5)*p.sigma_h*p.sigma_h*p.sigma_z)
                                *exp(-0.5*abs(dx*dx/p.sigma_h/p.sigma_h))
                                *exp(-0.5*abs(dy*dy/p.sigma_h/p.sigma_h))
                                *(exp(-0.5*abs(dz*dz/p.sigma_z/p.sigma_z))
                                +exp(-0.5*abs(dzv*dzv/p.sigma_z/p.sigma_z)));

            atomicAdd(&concs[gridIdx], contribution);
        }
    }
}


__global__ void move_puffs_by_wind_val(Gpuff::Puffcenter* d_puffs)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    float xwind = 1.0f;
    float ywind = 0.0f;
    float zwind = 0.0f;

    p.x += xwind*d_dt;
    p.y += ywind*d_dt;
    p.z += zwind*d_dt;

    if(p.z<0.0) p.z=-p.z;
}

__global__ void dry_deposition_val(Gpuff::Puffcenter* d_puffs) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    p.conc *= exp(-p.drydep_vel*d_dt/1000.0);

}

__global__ void wet_scavenging_val(Gpuff::Puffcenter* d_puffs) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    float Lambda = 3.5e-5*(1.0-0.8)/(1.0-0.8);

    p.conc *= exp(-Lambda*d_dt);

}

__global__ void radioactive_decay_val(Gpuff::Puffcenter* d_puffs) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    p.conc *= exp(-p.decay_const*d_dt);

}

__global__ void puff_dispersion_update_val(Gpuff::Puffcenter* d_puffs) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    float xwind = 1.0f;
    float ywind = 0.0f;
    float zwind = 0.0f;

    float vel = sqrt(xwind*xwind + ywind*ywind + zwind+zwind);
    int PasquillCategory = 5;

    float new_virtual_distance_h = NewtonRaphson_h(PasquillCategory, p.sigma_h, p.virtual_distance) + vel*d_dt;
    float new_virtual_distance_z = NewtonRaphson_z(PasquillCategory, p.sigma_z, p.virtual_distance) + vel*d_dt;

    if(d_isPG){
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
        p.sigma_z = Sigma_z_Pasquill_Gifford(PasquillCategory, new_virtual_distance_z);
    }
    else{
        p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
        p.sigma_z = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_z);
    }

    p.virtual_distance = new_virtual_distance_h;

}

__global__ void accumulate_conc_val(
    Gpuff::Puffcenter* puffs, 
    RectangleGrid::GridPoint* d_grid, 
    float* concs, 
    int ngrid)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridIdx = globalIdx % ngrid;
    int puffIdx = globalIdx / ngrid;

    if(puffIdx >= d_nop) return;

    Gpuff::Puffcenter& p = puffs[puffIdx];
    RectangleGrid::GridPoint& g = d_grid[gridIdx];

    if(p.flag){
        float dx = g.x - p.x;
        float dy = g.y - p.y;
        float dz = g.z - p.z;
        float dzv = g.z + p.z;

        if(p.sigma_h != 0.0f && p.sigma_z != 0.0f){
            float contribution = p.conc/(pow(2*PI,1.5)*p.sigma_h*p.sigma_h*p.sigma_z)
                                *exp(-0.5*abs(dx*dx/p.sigma_h/p.sigma_h))
                                *exp(-0.5*abs(dy*dy/p.sigma_h/p.sigma_h))
                                *(exp(-0.5*abs(dz*dz/p.sigma_z/p.sigma_z))
                                +exp(-0.5*abs(dzv*dzv/p.sigma_z/p.sigma_z)));

            atomicAdd(&concs[gridIdx], contribution);
        }

    }
}