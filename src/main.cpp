#include "driver.h"

#include <iostream>

////////////////////////////////////////////////////////////
// 

int main( int argc, char * argv[  ] )
{
    input args;
    if( argc != 2 )
        throw std::runtime_error( "Invalid args number" );
    args.set_file( argv[ 1 ] );
    args.    read(           );
    
    using   spec_t = species::base_t  <  4  >;
    using   latt_t = lattice::base_t  < int >;
    using   proc_t = process::base_t         ;
    driver::base_t < proc_t,  latt_t, spec_t > driver;

    driver.init  ( args );
    driver.action(      );
    driver.free  (      );
    return 0;
}
