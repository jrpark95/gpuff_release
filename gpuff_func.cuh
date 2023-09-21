#include "gpuff.cuh"

Gpuff::Gpuff() 
    : device_meteorological_data_pres(nullptr), 
      device_meteorological_data_unis(nullptr), 
      device_meteorological_data_etas(nullptr){}

Gpuff::~Gpuff()
{
    if (device_meteorological_data_pres){
        cudaFree(device_meteorological_data_pres);
    }
    if (device_meteorological_data_unis){
        cudaFree(device_meteorological_data_unis);
    }
    if (device_meteorological_data_etas){
        cudaFree(device_meteorological_data_etas);
    }
    if (d_puffs){
        cudaFree(d_puffs);
    }
}


// void Gpuff::print_puffs() const {
//     std::ofstream outfile("output.txt");

//     if (!outfile.is_open()){
//         std::cerr << "Failed to open output.txt for writing!" << std::endl;
//         return;
//     }

//     outfile << "puffs Info:" << std::endl;
//     for (const auto& p : puffs){
//         outfile << "---------------------------------\n";
//         outfile << "x: " << p.x << ", y: " << p.y << ", z: " << p.z << "\n";
//         outfile << "Decay Constant: " << p.decay_const << "\n";
//         outfile << "Concentration: " << p.conc << "\n";
//         outfile << "Time Index: " << p.timeidx << "\n";
//         outfile << "Flag: " << p.flag << "\n";
//     }

//     outfile.close();
// }

void Gpuff::clock_start(){
    _clock0 = std::chrono::high_resolution_clock::now();
}

void Gpuff::clock_end(){
    _clock1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(_clock1 - _clock0);
    std::cout << "Elapsed time: " << duration.count()/1.0e6 << " seconds" << std::endl;
}

void Gpuff::print_puffs(){

    std::ofstream outfile("output.txt");

    outfile << std::left << std::setw(12) << "X" 
           << std::setw(12) << "Y" 
           << std::setw(12) << "Z" 
           << std::setw(17) << "decay_const" 
           << std::setw(17) << "source_conc"
           << std::setw(10) << "timeidx"
           << std::setw(10) << "flag" 
           << std::endl;
    outfile << std::string(110, '-') << std::endl;

    for(const auto& p : puffs){
        outfile << std::left << std::fixed << std::setprecision(2)
                << std::setw(12) << p.x 
                << std::setw(12) << p.y 
                << std::setw(12) << p.z;

        outfile << std::scientific
                << std::setw(17) << p.decay_const 
                << std::setw(17) << p.conc
                << std::setw(10) << p.timeidx 
                << std::setw(10) << p.flag 
                << std::endl;
    }
    outfile.close();

}


void Gpuff::allocate_and_copy_to_device(){

    cudaError_t err = cudaMalloc((void**)&d_puffs, puffs.size() * sizeof(Puffcenter));

    if (err != cudaSuccess){
        std::cerr << "Failed to allocate device memory for puffs: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_puffs, puffs.data(), puffs.size() * sizeof(Puffcenter), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cerr << "Failed to copy puffs from host to device: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Gpuff::print_device_puffs_timeidx(){

    const int threadsPerBlock = 256; 
    const int blocks = (puffs.size() + threadsPerBlock - 1) / threadsPerBlock;

    print_timeidx<<<blocks, threadsPerBlock>>>(d_puffs);

    cudaDeviceSynchronize();

}

void Gpuff::time_update(){

    cudaError_t err = cudaGetLastError();
    
    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (puffs.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;
    

    while(currentTime <= time_end){

        activationRatio = currentTime / time_end;

        update_puff_flags<<<blocks, threadsPerBlock>>>
            (d_puffs, activationRatio);
        cudaDeviceSynchronize();

        move_puffs_by_wind<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        dry_deposition<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        wet_scavenging<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        radioactive_decay<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        puff_dispersion_update<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("CUDA error: %s, timestep = %d\n", cudaGetErrorString(err), timestep);

        currentTime += dt;
        timestep++;

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", currentTime);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));

            //puff_output_ASCII(timestep);
            puff_output_binary(timestep);
        }

    }

    // printf("-------------------------------------------------\n");
    // printf("Time : %f\n\tsec", currentTime);
    // printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));
    // printf("size = %d\n", puffs.size());
    // puff_output_ASCII(timestep);

}

void Gpuff::time_update_val(){

    cudaError_t err = cudaGetLastError();
    
    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (puffs.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;
    

    while(currentTime <= time_end){

        activationRatio = currentTime / time_end;

        update_puff_flags<<<blocks, threadsPerBlock>>>
            (d_puffs, activationRatio);
        cudaDeviceSynchronize();

        move_puffs_by_wind_val<<<blocks, threadsPerBlock>>>(d_puffs);
        cudaDeviceSynchronize();

        dry_deposition_val<<<blocks, threadsPerBlock>>>(d_puffs);
        cudaDeviceSynchronize();

        wet_scavenging_val<<<blocks, threadsPerBlock>>>(d_puffs);
        cudaDeviceSynchronize();

        radioactive_decay_val<<<blocks, threadsPerBlock>>>(d_puffs);
        cudaDeviceSynchronize();

        puff_dispersion_update_val<<<blocks, threadsPerBlock>>>(d_puffs);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("CUDA error: %s, timestep = %d\n", cudaGetErrorString(err), timestep);

        currentTime += dt;
        timestep++;

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", currentTime);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));

            //puff_output_ASCII(timestep);
            puff_output_binary(timestep);
        }

    }

}


void Gpuff::find_minmax(){

    cudaMalloc(&d_minX, sizeof(float));
    cudaMalloc(&d_minY, sizeof(float));
    cudaMalloc(&d_maxX, sizeof(float));
    cudaMalloc(&d_maxY, sizeof(float));

    float initial_min = FLT_MAX;
    float initial_max = -FLT_MAX;

    cudaMemcpy(d_minX, &initial_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minY, &initial_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxX, &initial_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxY, &initial_max, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (puffs.size() + threadsPerBlock - 1) / threadsPerBlock;

    findMinMax<<<blocks, threadsPerBlock, 4 * threadsPerBlock * sizeof(float)>>>(d_puffs, d_minX, d_minY, d_maxX, d_maxY);

    cudaMemcpy(&minX, d_minX, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&minY, d_minY, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxX, d_maxX, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxY, d_maxY, sizeof(float), cudaMemcpyDeviceToHost);

    //std::cout << "Min X: " << minX << ", Max X: " << maxX << ", Min Y: " << minY << ", Max Y: " << maxY << std::endl;

    cudaFree(d_minX);
    cudaFree(d_minY);
    cudaFree(d_maxX);
    cudaFree(d_maxY);

}

// void Gpuff::conc_calc() {

//     RectangleGrid rect(minX, minY, maxX, maxY);

//     int ngrid = rect.rows * rect.cols;
    
//     RectangleGrid::GridPoint* d_grid;
//     float* d_concs;

//     cudaMalloc(&d_grid, ngrid * sizeof(RectangleGrid::GridPoint));
//     cudaMalloc(&d_concs, ngrid * sizeof(float));

//     float* h_concs = new float[ngrid];

//     cudaMemcpy(d_grid, rect.grid, ngrid * sizeof(RectangleGrid::GridPoint), cudaMemcpyHostToDevice);

//     int blockSize = 128;
//     dim3 threadsPerBlock(blockSize, blockSize);
//     dim3 numBlocks((ngrid + blockSize - 1) / blockSize, (nop + blockSize - 1) / blockSize);
    
//     accumulateConc<<<numBlocks, threadsPerBlock>>>
//         (d_puffs, d_grid, d_concs, ngrid);

//     cudaMemcpy(h_concs, d_concs, ngrid * sizeof(float), cudaMemcpyDeviceToHost);

//     grid_output_binary(rect, h_concs);

//     delete[] h_concs;
//     cudaFree(d_grid);
//     cudaFree(d_concs);

// }

void Gpuff::conc_calc(){

    RectangleGrid rect(minX, minY, maxX, maxY);

    int ngrid = rect.rows * rect.cols;
    
    RectangleGrid::GridPoint* d_grid;
    float* d_concs;

    cudaMalloc(&d_grid, ngrid * sizeof(RectangleGrid::GridPoint));
    cudaMalloc(&d_concs, ngrid * sizeof(float));

    float* h_concs = new float[ngrid];

    cudaMemcpy(d_grid, rect.grid, ngrid * sizeof(RectangleGrid::GridPoint), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int totalThreads = ngrid * nop;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    
    accumulate_conc<<<numBlocks, blockSize>>>
        (d_puffs, d_grid, d_concs, ngrid);

    cudaMemcpy(h_concs, d_concs, ngrid * sizeof(float), cudaMemcpyDeviceToHost);

    grid_output_binary(rect, h_concs);
    grid_output_csv(rect, h_concs);

    delete[] h_concs;
    cudaFree(d_grid);
    cudaFree(d_concs);

}

void Gpuff::conc_calc_val(){

    RectangleGrid rect(minX, minY, maxX, maxY);

    int ngrid = rect.rows * rect.cols * rect.zdim;
    
    RectangleGrid::GridPoint* d_grid;
    float* d_concs;

    cudaMalloc(&d_grid, ngrid * sizeof(RectangleGrid::GridPoint));
    cudaMalloc(&d_concs, ngrid * sizeof(float));

    float* h_concs = new float[ngrid];

    cudaMemcpy(d_grid, rect.grid, ngrid * sizeof(RectangleGrid::GridPoint), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int totalThreads = ngrid * nop;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    
    accumulate_conc_val<<<numBlocks, blockSize>>>
        (d_puffs, d_grid, d_concs, ngrid);

    cudaMemcpy(h_concs, d_concs, ngrid * sizeof(float), cudaMemcpyDeviceToHost);

    grid_output_binary_val(rect, h_concs);
    grid_output_csv(rect, h_concs);

    delete[] h_concs;
    cudaFree(d_grid);
    cudaFree(d_concs);

}