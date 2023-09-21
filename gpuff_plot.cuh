#include "gpuff.cuh"

int Gpuff::countflag(){
    int count = 0;
    for(int i = 0; i < nop; ++i) if(puffs[i].flag == 1) count++;
    return count;
}

void Gpuff::puff_output_ASCII(int timestep){

    cudaMemcpy(puffs.data(), d_puffs, nop * sizeof(Puffcenter), cudaMemcpyDeviceToHost);

    int part_num = countflag();

    std::ostringstream filenameStream;

    std::string path = "./output";

#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0777);
#endif

    filenameStream << "./output/puff_" << std::setfill('0') 
    << std::setw(5) << timestep << "stp.vtk";

    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename);

    if (!vtkFile.is_open()){
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0" << std::endl;
    vtkFile << "puff data" << std::endl;
    vtkFile << "ASCII" << std::endl;
    vtkFile << "DATASET POLYDATA" << std::endl;

    vtkFile << "POINTS " << part_num << " float" << std::endl;
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        vtkFile << puffs[i].x << " " << puffs[i].y << " " << puffs[i].z << std::endl;
    }

    // vtkFile << "POINTS " << nop << " float" << std::endl;
    // for (int i = 0; i < nop; ++i){
    //     //if(!puffs[i].flag) continue;
    //     vtkFile << puffs[i].x << " " << puffs[i].y << " " << i << " " << puffs[i].flag << std::endl;
    // }

    vtkFile.close();
}


void Gpuff::swapBytes(float& value){
    char* valuePtr = reinterpret_cast<char*>(&value);
    std::swap(valuePtr[0], valuePtr[3]);
    std::swap(valuePtr[1], valuePtr[2]);
}

void Gpuff::puff_output_binary(int timestep){

    cudaMemcpy(puffs.data(), d_puffs, nop * sizeof(Puffcenter), cudaMemcpyDeviceToHost);

    int part_num = countflag();

    std::ostringstream filenameStream;

    std::string path = "./output";

    #ifdef _WIN32
        _mkdir(path.c_str());
    #else
        mkdir(path.c_str(), 0777);
    #endif

    // filenameStream << "./output/puff_" << std::setfill('0') 
    //                << std::setw(5) << timestep << "stp.vtk";

    //filenameStream << "./output/puff_" << timestep << "stp.vtk";

    filenameStream << "./output/puff_" << std::setfill('0') 
                   << std::setw(5) << timestep << ".vtk";

    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()){
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "puff data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    vtkFile << "POINTS " << part_num << " float\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float x = puffs[i].x;
        float y = puffs[i].y;
        float z = puffs[i].z;

        swapBytes(x);
        swapBytes(y);
        swapBytes(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }

    vtkFile << "POINT_DATA " << part_num << "\n";
    vtkFile << "SCALARS sigma_h float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float vval = puffs[i].sigma_h;
        swapBytes(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS sigma_z float 1\n"; 
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float vval = puffs[i].sigma_z;
        swapBytes(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS virtual_dist float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float vval = puffs[i].virtual_distance;
        swapBytes(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS Q float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float vval = puffs[i].conc;
        swapBytes(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }


    vtkFile.close();
}


void Gpuff::grid_output_binary(RectangleGrid& rect, float* h_concs){

    std::string path = "./grids";

    #ifdef _WIN32
        _mkdir(path.c_str());
    #else
        mkdir(path.c_str(), 0777);
    #endif

    std::string filename = "./grids/grid.vtk";

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()){
    std::cerr << "Cannot open file for writing: " << filename << std::endl;
    return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "Grid data generated by RectangleGrid\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET STRUCTURED_GRID\n";
    vtkFile << "DIMENSIONS " << rect.cols << " " << rect.rows << " " << 1 << "\n";
    vtkFile << "POINTS " << rect.rows * rect.cols << " float\n";

    for (int i = 0; i < rect.rows; ++i) {
        for (int j = 0; j < rect.cols; ++j) {

            int index = i * rect.cols + j;

            float x = rect.grid[index].x;
            float y = rect.grid[index].y;
            float z = 0.0f;

            swapBytes(x);
            swapBytes(y);
            swapBytes(z);

            vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
            vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
            vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));

        }
    }

        vtkFile << "\nPOINT_DATA " << rect.rows * rect.cols << "\n";
        vtkFile << "SCALARS concentration float 1\n";
        vtkFile << "LOOKUP_TABLE default\n";
    
        for (int i = 0; i < rect.rows; ++i) {
            for (int j = 0; j < rect.cols; ++j) {
                float conc = h_concs[i * rect.cols + j];
                swapBytes(conc);
                vtkFile.write(reinterpret_cast<char*>(&conc), sizeof(float));
            }
        }
    

    vtkFile.close();
}

void Gpuff::grid_output_binary_val(RectangleGrid& rect, float* h_concs){

    std::string path = "./grids";

    #ifdef _WIN32
        _mkdir(path.c_str());
    #else
        mkdir(path.c_str(), 0777);
    #endif

    for(int zidx = 0; zidx < 22; ++zidx){

        std::stringstream ss;
        ss << "./grids/grid" << std::setw(3) << std::setfill('0') << zidx << ".vtk";
        std::string filename = ss.str();
    
        std::ofstream vtkFile(filename, std::ios::binary);
    
        if (!vtkFile.is_open()) {
            std::cerr << "Cannot open file for writing: " << filename << std::endl;
            continue; 
        }

        vtkFile << "# vtk DataFile Version 4.0\n";
        vtkFile << "Grid data generated by RectangleGrid\n";
        vtkFile << "BINARY\n";
        vtkFile << "DATASET STRUCTURED_GRID\n";
        vtkFile << "DIMENSIONS " << rect.cols << " " << rect.rows << " " << 1 << "\n";
        vtkFile << "POINTS " << rect.rows * rect.cols << " float\n";

        for (int i = 0; i < rect.rows; ++i) {
            for (int j = 0; j < rect.cols; ++j) {

                int index = i * rect.cols * 21 + j * 21 + zidx;

                float x = rect.grid[index].x;
                float y = rect.grid[index].y;
                float z = 0.0f;

                swapBytes(x);
                swapBytes(y);
                swapBytes(z);

                vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
                vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
                vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));

            }
        }

            vtkFile << "\nPOINT_DATA " << rect.rows * rect.cols << "\n";
            vtkFile << "SCALARS concentration float 1\n";
            vtkFile << "LOOKUP_TABLE default\n";
        
            for (int i = 0; i < rect.rows; ++i) {
                for (int j = 0; j < rect.cols; ++j) {
                    float conc = h_concs[i * rect.cols * 21 + j * 21 + zidx];
                    swapBytes(conc);
                    vtkFile.write(reinterpret_cast<char*>(&conc), sizeof(float));
                }
            }
        

        vtkFile.close();
    }
}

void Gpuff::grid_output_csv(RectangleGrid& rect, float* h_concs){

    std::string path = "./grids_csv";

    #ifdef _WIN32
        _mkdir(path.c_str());
    #else
        mkdir(path.c_str(), 0777);
    #endif

    std::stringstream ss;
    ss << "./grids_csv/grid.csv";
    std::string filename = ss.str();

    std::ofstream csvFile(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
    }

    for (int i = 0; i < rect.rows; ++i) {
        for (int j = 0; j < rect.cols; ++j){

            int index = i * rect.cols + j;
            float conc = h_concs[index];

            // Separate values by comma, but not at the end of the line
            csvFile << conc;
            if (j < rect.cols - 1) {
                csvFile << ",";
            }
        }
        csvFile << "\n"; // New line at the end of each row
    }

    csvFile.close();

}