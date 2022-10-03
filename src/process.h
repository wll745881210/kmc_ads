#pragma once

#include "lattice.h"
#include "type.h"
#include "utilities/reduction.h"

namespace process
{
////////////////////////////////////////////////////////////
//

struct base_t
{
    template < class latt_T, class spec_T >
    __device__ __forceinline__ void operator(  )
    ( const latt_T & latt, const spec_T & spec ) const
    {
        // #warning "TEST: space for CRTP"
        const auto & self( * this );
        lattice::nbr_t < typename latt_T::stat_t >  dat;
        if( ! self.dispatch( dat.idx, latt ) )
            return;

        dat .load( latt );
        self.proc( dat, latt, spec );
        return;
    };

    template < class latt_T > __device__ __forceinline__
    bool dispatch( type::idx_t & i, const latt_T & l ) const
    {
        const int n_max = l.n_cell[ 0 ] * l.n_cell[ 1 ];
        const int i_tot = threadIdx.x
                        +  blockIdx.x * blockDim.x;
        if( i_tot >= n_max )
            return false;

        i[ 1 ] = i_tot / l.n_cell[ 0 ];
        i[ 0 ] = i_tot - l.n_cell[ 0 ] * i[ 1 ];
        return true;
    };
    
    template < class nbr_T, class latt_T, class spec_T >
    __device__ __forceinline__ 
    void proc( nbr_T & dat, const latt_T & latt ,
                            const spec_T & spec ) const
    {
        extern __shared__ type::float_t dt_local[  ];
        int * occupy_local = ( int * )
                             ( dt_local + blockDim.x );
                
        spec.rnd_load(             ) ;
        spec. impinge( dat,   latt ) ;
        occupy_local [ threadIdx.x ] =
        spec. hopping( dat,   latt ) ;
        spec.rnd_save(             ) ;
        
        dt_local[ threadIdx.x ] = spec.dt_new * spec.cfl;
        __syncthreads(  );
        if( threadIdx.x  == 0 )
        {
            for( int i = 1; i < blockDim.x; ++ i )
            {
                dt_local    [ 0 ]  = fminf( dt_local[ 0 ]  ,
                                            dt_local[ i ] );
                occupy_local[ 0 ] +=    occupy_local[ i ]  ;
            }
            utils::atomic_red < type::float_t,     fminf >
                         ( latt.p_dt_new,  dt_local[ 0 ] );
            atomicAdd( latt.p_occupy,  occupy_local[ 0 ] );
        }
        return;
    };
};

};
