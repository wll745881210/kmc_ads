#pragma once

#include "type.h"

#include <cfloat>

namespace lattice
{
////////////////////////////////////////////////////////////
// Lattice for adsorption

template< class stat_T >
struct base_t
{
    using stat_t =         stat_T  ;
    using this_t = base_t< stat_T >;

    stat_T *            data;
    bool                host;
    int           * p_occupy;
    type::  idx_t     n_cell;
    type::float_t *     p_dt;
    type::float_t * p_dt_old;
    type::float_t * p_dt_new;

    __host__  base_t(  ) : p_dt( nullptr ), host( true ), 
                           data( nullptr )
    {
        n_cell[ 0 ] = 0;
        n_cell[ 1 ] = 0;
    };

    __device__ __host__ __forceinline__
    stat_T * at( const type::idx_t & idx ) const
    {
        int i[ 2 ];
        for( int a = 0; a < 2; ++ a )
        {
            i[ a ]  = idx[ a ];
            i[ a ] += i  [ a ] < 0 ? n_cell[ a ] : 0;
            i[ a ] -= i  [ a ] >=    n_cell[ a ] ?
                                     n_cell[ a ] : 0;
        }
        return const_cast< stat_T * >
             ( data + i[ 0 ] + n_cell[ 0 ] * i[ 1 ] );
    };

    __host__ __device__ int n_tot(  ) const
    {
        return n_cell[ 0 ] * n_cell[ 1 ];
    };

    __host__ __device__ size_t s_tot(  ) const
    {
        return n_tot(  ) * sizeof( stat_T );
    };

    __host__ virtual void setup
    ( const type::idx_t & n_cell, const bool & host )
    {
        this->n_cell = n_cell;
        this->  host =   host;
        if( host )
        {
            cudaMallocHost( & data,     s_tot (        ) );
            cudaMallocHost( & p_dt, 2 * sizeof( * p_dt ) );
            cudaMallocHost( & p_occupy, sizeof(    int ) );
        }
        else
        {
            cudaMalloc    ( & data,     s_tot (        ) );
            cudaMalloc    ( & p_dt, 2 * sizeof( * p_dt ) );
            cudaMalloc    ( & p_occupy, sizeof(    int ) );
        }
        p_dt_old = p_dt;
        p_dt_new = p_dt + 1;
        this->null(  );
        return;
    };

    __host__ virtual void set_dt
    ( const size_t & i_iter, const int & n_check,
      this_t & dest )
    {
        auto * q = p_dt_new;
        p_dt_new = p_dt_old;
        p_dt_old = q;
        if( i_iter % 10 == 0 )
            cudaMemcpy( dest.p_occupy, p_occupy, sizeof
                      ( int ), cudaMemcpyDefault );
        cudaMemset( p_dt_new, 126, sizeof( * q ) );
        cudaMemset( p_occupy, 0,   sizeof( int ) );
        
        return;
    };

    __host__ virtual void null(  )
    {
        if( host )
        {
            memset    ( data, 0,     s_tot (        ) );
            memset    ( p_dt, 0, 2 * sizeof( * p_dt ) );
            memset    ( p_occupy, 0, sizeof(    int ) );
        }
        else
        {
            cudaMemset( data, 0,     s_tot (        ) );
            cudaMemset( p_dt, 0, 2 * sizeof( * p_dt ) );
            cudaMemset( p_occupy, 0, sizeof(    int ) );
        }
        return;
    };

    __host__ virtual void free(  )
    {
        if( data == nullptr )
            return;
        if( host )
        {
            cudaFreeHost( data     );
            cudaFreeHost( p_dt     );
            cudaFreeHost( p_occupy );            
        }
        else
        {
            cudaFree    ( data     );
            cudaFree    ( p_dt     );
            cudaFree    ( p_occupy );
        }
        data     = nullptr;
        p_dt     = nullptr;
        p_occupy = nullptr;
        return;
    };

    __host__ virtual void copy_from( const this_t & src )
    {
        cudaMemcpy( data, src.data, s_tot(  ),
                    cudaMemcpyDefault );
        cudaMemcpy( p_dt, src.p_dt, 2 * sizeof( * p_dt ),
                    cudaMemcpyDefault );
        return;
    };
};

////////////////////////////////////////////////////////////
// Neighbors

template< class stat_T >
struct nbr_t
{
    using stat_t = stat_T;
    type::idx_t       idx;
    stat_T       dat[ 5 ];

    template < class latt_T > __device__ __forceinline__
    void load( const latt_T & l )
    {
        type::idx_t  i_t;
        short count( 0 );
        for( int i = 0; i < 3; ++ i )
        {
            i_t[ 0 ] = idx[ 0 ] + i - 1 ;
            for( int j = 0; j < 3; ++ j )
            {
                if( j != 1 && i != 1 )
                    continue;
                i_t[ 1 ]      = idx [ 1 ]   + j - 1;
                dat[ count ++ ] = ( * l.at( i_t ) );
            }
        }
        return;
    };

    __device__ __forceinline__ stat_T & center(  )
    {
        return dat[ 2 ];
    };

    template< class fun_T > __device__ __forceinline__
    void map( const fun_T & f ) const
    {
        type::idx_t di;
        short count( 0 );        
        for( di[ 0 ] = -1; di[ 0 ] <= 1; ++ di[ 0 ] )
            for( di[ 1 ] = -1; di[ 1 ] <= 1; ++ di[ 1 ] )
            {
                if( di[ 0 ] != 0 && di[ 1 ] != 0 )
                    continue;
                if( di[ 0 ] == 0 && di[ 1 ] == 0 )
                {
                    ++ count;
                    continue;
                }
                f( dat[ count ++ ], di );
            }
        return;
    };
    
};

};

