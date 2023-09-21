#pragma once

#define LDAPS_E 132.36
#define LDAPS_W 121.06
#define LDAPS_N 43.13
#define LDAPS_S 32.20
#define PI 3.141592

#define dimX 602
#define dimY 781
#define dimZ_pres 24
#define dimZ_etas 71

struct PresData {
    float DZDT;     // No.1 [DZDT] Vertical velocity (m/s)
    float UGRD;     // No.2 [UGRD] U-component of wind (m/s)
    float VGRD;     // No.3 [VGRD] V-component of wind (m/s)
    float HGT;      // No.4 [HGT] Geopotential height (m)
    float TMP;      // No.5 [TMP] Temperature (K)
    float RH;       // No.7 [RH] Relative Humidity (%)
};

struct EtasData {
    float UGRD;     // No.1 [UGRD] U-component of wind (m/s)
    float VGRD;     // No.2 [VGRD] V-component of wind (m/s)
    float DZDT;     // No.6 [DZDT] Vertical velocity (m/s)
    float DEN;      // No.7 [DEN] Density of the air (kg/m)
};

struct UnisData {
    float HPBLA;    // No.12 [HPBLA] Boundary Layer Depth after B. LAYER (m)
    float T1P5;     // No.21 [TMP] Temperature at 1.5m above ground (K)
    float SHFLT;    // No.39 [SHFLT] Surface Sensible Heat Flux on Tiles (W/m^2)
    float HTBM;     // No.43 [HTBM] Turbulent mixing height after B. Layer (m)
    float HPBL;     // No.131 [HPBL] Planetary Boundary Layer Height (m)
    float SFCR;     // No.132 [SFCR] Surface Roughness (m)
};

struct Source {
    float lat;
    float lon;
    float height;
};

struct Concentration {
    int location;
    int sourceterm;
    double value;
};

class RectangleGrid {
private:

public:

    float minX, minY, maxX, maxY;
    float intervalX, intervalY, intervalZ;
    int rows, cols, zdim;

    struct GridPoint{
        float x;
        float y;
        float z;
        float conc;
    };

    GridPoint* grid;

    RectangleGrid(float _minX, float _minY, float _maxX, float _maxY){

        float width = _maxX - _minX;
        float height = _maxY - _minY;

        minX = _minX - width * 0.5;
        maxX = _maxX + width * 0.5;
        minY = _minY - height * 0.5;
        maxY = _maxY + height * 0.5;

        rows = std::sqrt(3000 * (height / width));
        cols = std::sqrt(3000 * (width / height));

        intervalX = (maxX - minX) / (cols - 1);
        intervalY = (maxY - minY) / (rows - 1);
        intervalZ = 10.0f;

        grid = new GridPoint[rows * cols];
        for(int i = 0; i < rows; ++i){
            for(int j = 0; j < cols; ++j){
                int index = i * cols + j;
                grid[index].x = minX + j * intervalX;
                grid[index].y = minY + i * intervalY;
                grid[index].z = 20.0;
            }
        }

    } 

    ~RectangleGrid(){
        delete[] grid;
    }
};