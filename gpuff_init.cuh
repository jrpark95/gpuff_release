#include "gpuff.cuh"

void Gpuff::read_simulation_config(){

    FILE* file;
    FILE* sourceFile;

    #ifdef _WIN32
        file = fopen(".\\input\\setting.txt", "r");
        sourceFile = fopen(".\\input\\source.txt", "r");
    #else
        file = fopen("./input/setting.txt", "r");
        sourceFile = fopen("./input/source.txt", "r");
    #endif

    if (!file){
        std::cerr << "Failed to open setting.txt" << std::endl;
        exit(1);
    }

    char buffer[256];
    int tempValue;

    while (fgets(buffer, sizeof(buffer), file)){

        if (buffer[0] == '#') continue;

        if (strstr(buffer, "Time_end(s):")){
            sscanf(buffer, "Time_end(s): %f", &time_end);
        } else if (strstr(buffer, "dt(s):")){
            sscanf(buffer, "dt(s): %f", &dt);
        } else if (strstr(buffer, "Plot_output_freq:")){
            sscanf(buffer, "Plot_output_freq: %d", &freq_output);
        } else if (strstr(buffer, "Total_number_of_puff:")){
            sscanf(buffer, "Total_number_of_puff: %d", &nop);
        } else if (strstr(buffer, "Rural/Urban:")){
            sscanf(buffer, "Rural/Urban: %d", &tempValue);
            isRural = tempValue;
        } else if (strstr(buffer, "Pasquill-Gifford/Briggs-McElroy-Pooler:")){
            sscanf(buffer, "Pasquill-Gifford/Briggs-McElroy-Pooler: %d", &tempValue);
            isPG = tempValue;
        }

    }

    fclose(file);

    if (!sourceFile){
        std::cerr << "Failed to open source.txt" << std::endl;
        exit(1);
    }

    while (fgets(buffer, sizeof(buffer), sourceFile)){
        if (buffer[0] == '#') continue;
    
        // SOURCE coordinates
        if (strstr(buffer, "[SOURCE]")) {
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[SOURCE_TERM]")) {
                if (buffer[0] == '#') continue;
    
                Source src;
                sscanf(buffer, "%f %f %f", &src.lat, &src.lon, &src.height);
                sources.push_back(src);
            }
            sources.pop_back();
        }
    
        // SOURCE_TERM values
        if (strstr(buffer, "[SOURCE_TERM]")){
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[RELEASE_CASES]")) {
                if (buffer[0] == '#') continue;
    
                int srcnum;
                float decay, depvel;
                sscanf(buffer, "%d %f %f", &srcnum, &decay, &depvel);
                decayConstants.push_back(decay);
                drydepositionVelocity.push_back(depvel);
            }
            decayConstants.pop_back();
            drydepositionVelocity.pop_back();
        }
    
        // RELEASE_CASES
        if (strstr(buffer, "[RELEASE_CASES]")){
            while (fgets(buffer, sizeof(buffer), sourceFile)) {
                if (buffer[0] == '#') continue;
    
                Concentration conc;
                sscanf(buffer, "%d %d %lf", &conc.location, &conc.sourceterm, &conc.value);
                concentrations.push_back(conc);
            }
        }
    }
    
    fclose(sourceFile);

    nop = floor(nop/(sources.size()*decayConstants.size()))*sources.size()*decayConstants.size();

    cudaError_t err;

    err = cudaMemcpyToSymbol(d_time_end, &time_end, sizeof(float));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_dt, &dt, sizeof(float));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_freq_output, &freq_output, sizeof(int));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_nop, &nop, sizeof(int));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_isRural, &isRural, sizeof(bool));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_isPG, &isPG, sizeof(bool));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));

    // for(const auto& source : sources){
    //     std::cout << source.lat << ", " << source.lon << ", " << source.height << std::endl;
    // }
    // for(float decay : decayConstants){
    //     std::cout << decay << std::endl; 
    // }
    // for(float depvel : drydepositionVelocity){
    //     std::cout << depvel << std::endl; 
    // }
    // for(const auto& conc : concentrations){
    //     std::cout << conc.location << ", " << conc.sourceterm << ", " << conc.value << std::endl;
    // }

    // std::cout << "isRural = " << isRural << std::endl;
    // std::cout << "isPG = " << isPG << std::endl;

}

void Gpuff::read_etas_altitudes(){


    #ifdef _WIN32
        FILE* file = fopen(".\\input\\hgt_uv.txt", "r");
    #else
        FILE* file = fopen("./input/hgt_uv.txt", "r");
    #endif


    if (!file){
        std::cerr << "Failed to open setting.txt" << std::endl;
        exit(1);
    }

    char line[1000];
    int count = 0;
    int idx = 0;

    while(fgets(line, sizeof(line), file)){

        fgets(line, sizeof(line), file);

        char* token = strtok(line, ":");
        count = 0;
        char* val = nullptr;

        while(token){
            if (count == 4){
                val = token;
                break;
            }
            token = strtok(nullptr, ":");
            count++;
        }
        etas_hgt_uv[idx++] = atoi(strtok(val, " "));

    }

    cudaError_t err = cudaMemcpyToSymbol(d_etas_hgt_uv, etas_hgt_uv, sizeof(float) * (dimZ_etas-1));
    if (err != cudaSuccess) std::cerr << "Failed to copy data to constant memory: " << cudaGetErrorString(err) << std::endl;

    #ifdef _WIN32
        file = fopen(".\\input\\hgt_w.txt", "r");
    #else
        file = fopen("./input/hgt_w.txt", "r");
    #endif

    if (!file){
        std::cerr << "Failed to open setting.txt" << std::endl;
        exit(1);
    }

    count = 0;
    idx = 0;

    while(fgets(line, sizeof(line), file)){

        char* token = strtok(line, ":");
        count = 0;
        char* val = nullptr;

        while(token){
            if (count == 4){
                val = token;
                break;
            }
            token = strtok(nullptr, ":");
            count++;
        }
        etas_hgt_w[idx++] = atof(strtok(val, " "));
        //printf("hgt_w[%d] = %f\n", idx-1, etas_hgt_w[idx-1]);

    }

    err = cudaMemcpyToSymbol(d_etas_hgt_w, etas_hgt_w, sizeof(float) * (dimZ_etas-1));
    if (err != cudaSuccess) std::cerr << "Failed to copy data to constant memory: " << cudaGetErrorString(err) << std::endl;

}

void Gpuff::puff_initialization_val(){
    int puffsPerConc = nop / concentrations.size();

    for (const auto& conc : concentrations){
        for (int i = 0; i < puffsPerConc; ++i){
            
            float x = sources[conc.location - 1].lat;
            float y = sources[conc.location - 1].lon;
            float z = sources[conc.location - 1].height;

            puffs.push_back(Puffcenter(x, y, z, decayConstants[conc.sourceterm - 1], conc.value*time_end/nop, drydepositionVelocity[conc.sourceterm - 1], i + 1));
        }
    }

    // Sort the puffs by timeidx
    std::sort(puffs.begin(), puffs.end(), [](const Puffcenter& a, const Puffcenter& b){
        return a.timeidx < b.timeidx;
    });
}

void Gpuff::puff_initialization(){
    int puffsPerConc = nop / concentrations.size();

    for (const auto& conc : concentrations){
        for (int i = 0; i < puffsPerConc; ++i){
            
            float x = Lambert2x(
                sources[conc.location - 1].lat, 
                sources[conc.location - 1].lon);
            float y = Lambert2y(
                sources[conc.location - 1].lat, 
                sources[conc.location - 1].lon);
            float z = sources[conc.location - 1].height;

            puffs.push_back(Puffcenter(x, y, z, decayConstants[conc.sourceterm - 1], conc.value, drydepositionVelocity[conc.sourceterm - 1], i + 1));
        }
    }

    // Sort the puffs by timeidx
    std::sort(puffs.begin(), puffs.end(), [](const Puffcenter& a, const Puffcenter& b){
        return a.timeidx < b.timeidx;
    });
}

