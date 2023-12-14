#include "gpuff.cuh"

int main()
{
    Gpuff gpuff;

    cudaMalloc((void **)&d_vdepo, 10 * sizeof(float));
    //cudaMalloc((void **)&d_size, 9 * 10 * sizeof(float));
    cudaMalloc((void **)&d_radi, (RNUM+1) * sizeof(float));

    cudaMemcpy(d_vdepo, vdepo, 10 * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_size, size, 9 * 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_radi, radi, (RNUM+1) * sizeof(float), cudaMemcpyHostToDevice);


    cudaMalloc(&d_size, 9 * sizeof(float*));
    for (int i = 0; i < 9; i++) {
        float* d_row;
        cudaMalloc(&d_row, 10 * sizeof(float));
        cudaMemcpy(d_row, size[i], 10 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_size[i], &d_row, sizeof(float*), cudaMemcpyHostToDevice);
    }

    gpuff.read_simulation_config();
    gpuff.read_meteorological_data_RCAP();
    gpuff.puff_initialization_RCAP();
    gpuff.allocate_and_copy_to_device();
    gpuff.time_update_RCAP();

    cudaFree(d_vdepo);
    cudaFree(d_size);

    // gpuff.read_simulation_config();
    // gpuff.puff_initialization();
    // gpuff.receptor_initialization_ldaps();
    // gpuff.read_etas_altitudes();
    // gpuff.read_meteorological_data("pres.bin", "unis.bin", "etas.bin");
    // gpuff.allocate_and_copy_to_device();
    // gpuff.time_update_polar();

    return 0;
}