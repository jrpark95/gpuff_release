#include "gpuff.cuh"

int main()
{
    Gpuff gpuff;

    gpuff.read_simulation_config();

    gpuff.puff_initialization();

    gpuff.read_etas_altitudes();

    gpuff.read_meteorological_data("pres.bin", "unis.bin", "etas.bin");

    gpuff.allocate_and_copy_to_device();

    gpuff.clock_start();
    gpuff.time_update();
    gpuff.clock_end();//
    
    gpuff.find_minmax();
    
    gpuff.conc_calc();

    return 0;
}