#pragma once

#include "species.h"
#include "process.h"
#include "utilities/input.h"

#include <fstream>
#include <iostream>

namespace driver
{
////////////////////////////////////////////////////////////
// Process kernel

template < class  proc_T, class latt_T, class spec_T >
__global__ void kernel
( const proc_T proc, const latt_T latt, const spec_T spec )
{
    return proc( latt, spec );
}

////////////////////////////////////////////////////////////
// Output mode

enum omode_t : int { last_step = 0, quasi_log = 1 };

template < class I > I base( I i )
{
    I res( 1 );
    for( i /= 10; i > 0; i /= 10 )
        res *= 10;
    return res;
}

////////////////////////////////////////////////////////////
// Driver

template < class proc_T, class latt_T, class spec_T >
struct base_t
{
    bool                  osd;
    int                  n_th;
    int                  n_bl;
    size_t             n_iter;
    
    latt_T             latt_d;
    latt_T             latt_h;
    spec_T               spec;
    proc_T               proc;
    size_t        output_next;
    omode_t       output_mode;
    std::string output_prefix;
    std::string output_suffix;

    int               n_check;
    std::array< float_t, 2 > occupy_term;

    __host__ void init( const input & args )
    {
        cudaSetDevice
        ( args.get< int > ( "mesh", "idx_gpu", 0 ) );
        
        type::idx_t    n_cell ( 128, 128 );
        args( "mesh", "n_cell", n_cell.x );
        latt_h.setup(  n_cell,      true );
        latt_d.setup(  n_cell,     false );

        n_th   = args.get< int >( "mesh", "n_thread", 64 );
        n_bl   = ( latt_h. n_tot(  ) + n_th - 1 ) / n_th;
        n_iter = args.get< size_t >( "mesh", "n_iter", 1 );
        
        spec  .init( args, latt_h );
        // latt_h.p_dt[ 0 ] = spec.dt_new;
        latt_h.p_dt[ 1 ] = FLT_MAX;

        latt_d.copy_from( latt_h );
        cudaMemcpy( latt_d.p_dt, latt_h.p_dt, 2 * sizeof
                  ( type::float_t ),   cudaMemcpyDefault );
        
        osd = args.get< bool > ( "mesh", "osd", 0 );
        output_mode   = ( omode_t ) args.get< int >
                      ( "mesh", "output",      0  );
        output_prefix = args.get< std::string >
                      ( "mesh", "prefix",  "test" );
        output_suffix = args.get< std::string >
                      ( "mesh", "prefix",  ".dat" );
        output_next = ( output_mode == last_step ?
                        n_iter -  1 :  0 );        
        n_check       = args.get< type::float_t >
                      ( "adsorption", "occupy_check",  10 );
        occupy_term = { 1e-3, 0.99 };
        args( "adsorption", "occupy_term", occupy_term );
        return;
    };

    __host__ void free(  )
    {
        latt_d   .free(  );
        latt_h   .free(  );
        spec     .free(  );
        return;
     };
    
    __host__ void action(  )
    {
        const int s_sh = n_th *
            ( sizeof( type::float_t ) + sizeof( int ) );
        for( size_t i = 0; i <= n_iter;   ++ i )
        {
            spec.i_iter = ( 1 + i ) % latt_d.n_tot(  );
            kernel  <<<  n_bl,   n_th,    s_sh >>>
                    (    proc, latt_d,    spec   );
            latt_d.set_dt ( i, n_check,   latt_h );
            type::float_t f_occupy  = ( * latt_h.p_occupy );
            f_occupy               /=     latt_h.n_tot (  );
            if( f_occupy < occupy_term[ 0 ] ||
                f_occupy > occupy_term[ 1 ]  )
            {
                output_next = i  ;
                n_iter      = i  ;
            }
            output_shell    ( i );
        }
        return;
    };

    __host__ void output_shell( const size_t & iter )
    {
        if( iter != output_next )
            return;
        std::string file_name( output_prefix );
        if( output_mode == last_step )
            file_name   += output_suffix;
        else
        {
            char buf[ 11 ];
            std::sprintf( buf, "_%0.1e", ( float ) iter );
            file_name   += buf + output_suffix;
            output_next += base( iter );
        }
        return output( file_name, iter );
    };

    __host__ void output
    ( const std::string & file_name, const size_t & iter )
    {
        latt_h.copy_from( latt_d        );
        const  auto &  n( latt_h.n_cell );

        if( osd )
            std::cout << "Step " << iter << ", dt = "
                      << latt_h.p_dt[ 1 - iter % 2 ]
                      << " " << ( * latt_h.p_occupy )
                      << std::endl;

        std::ofstream fout( file_name.c_str(  ) );
        fout << "# " << n[ 0 ] << ' ' << n[ 1 ] << '\n';
        type::idx_t i;
        for( i[ 0 ] = 0; i[ 0 ] < n[ 0 ]; ++ i[ 0 ] )
        {
            for( i[ 1 ] = 0; i[ 1 ] < n[ 1 ]; ++ i[ 1 ] )
                fout << ( * latt_h.at( i ) ) << ' ';
            fout << '\n';
        }
        fout.close(  );
        return;
    };
};

};

