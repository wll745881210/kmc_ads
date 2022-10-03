#pragma once

#include "type.h"
#include "utilities/input.h"

#include <curand_kernel.h>
#include <curand.h>
#include <cstdio>
#include <cfloat>

namespace species
{
////////////////////////////////////////////////////////////
//

using rand_t = curandState;

__global__ void setup_rng
( rand_t * r_data, const int n_tot, const int seed = 4321 )
{
    const auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if( i < n_tot )
        curand_init( seed, i, 0, r_data + i );
    return;
}

////////////////////////////////////////////////////////////
//

template < int N >
struct base_t
{
    int                     n_spe;
    int                     n_rnd;
    rand_t       *       rnd_data;
    mutable rand_t      rnd_local;
    mutable type::float_t  dt_new;     

    
    type::float_t             cfl;
    type::float_t          cfl_in;
    type::float_t            beta;    
    type::float_t         zeta_in;
    type::float_t x_gas_cumu[ N ];

    int                    i_iter;
    int                  n_nb_max; // For energy calculation
    type::float_t  nu_hop_sp[ N ];
    type::float_t  nu_dad_sp[ N ];
    type::float_t   e_ads_sp[ N ];
    type::float_t   e_hop_sp[ N ];    
    type::float_t  e_ngbr_sp[ N ][ N ];
    type::float_t     e_corr[ N ];

    __host__ void init_rnd
    ( const int & n_rnd, const int & n_th )
    {
        this->n_rnd = n_rnd;
        cudaMalloc( & rnd_data, n_rnd * sizeof( rand_t ) );

        const int n_bl = ( n_rnd + n_th - 1 ) / n_th;
        setup_rng <<< n_bl, n_th >>> ( rnd_data, n_rnd );
        return;
    };

    template< class latt_T > __host__
    void init( const input & args, latt_T & latt )
    {
        std :: vector  < std::string > species  ;
        args( "adsorption", "species", species );
        n_spe = species.size(  );
        if( n_spe > N )
            throw std::runtime_error( "Incompatible n_sp" );

        n_nb_max = args.get< int >
                 ( "adsorption", "nb_max", 4 );

        std::vector< type::float_t > tmp( species.size( ) );
        type::float_t x_cumu( 0 );
        dt_new = FLT_MAX;
        for( int i = 0; i < n_spe;  ++ i )
        {
            const auto & sp = species[ i ];
            x_cumu += args.get< float > ( sp, "abund", 0. );
            x_gas_cumu[ i ] = x_cumu;
            e_ads_sp  [ i ] = args.get   < float >
                            ( sp,  "e_ads",    0 );
            e_hop_sp  [ i ] = args.get   < float >
                            ( sp,  "e_hop",    0 );
            nu_hop_sp [ i ] = args.get   < float >
                            ( sp, "nu_hop", 1e12 );
            nu_dad_sp [ i ] = args .get  < float >
                            ( sp, "nu_ads", 1e12 );

            args( sp, "e_ngbr", tmp );
            for( int j = 0; j < n_spe; ++ j )
                e_ngbr_sp[ i ][ j ] = tmp[ j ];
            e_corr[ i ] = args.get< type::float_t >
                        ( sp, "e_corr", 0.f );

            dt_new = std::min( dt_new, 1 / nu_hop_sp[ i ] );
            dt_new = std::min( dt_new, 1 / nu_dad_sp[ i ] );
        }
        for( int i = 0; i < n_spe;  ++ i )
            x_gas_cumu[ i ] /= x_cumu;

        cfl     = args.get <  float >
                ( "adsorption", "cfl",      1.e0 );
        cfl_in  = args.get <  float >
                ( "adsorption", "cfl",      1.e1 );
        beta    = 1 / args.get <  float >
                ( "adsorption", "beta_inv", 1e32 );
        zeta_in = args.get <  float >
                ( "adsorption", "zeta_in",  1e12 );
        dt_new  = std::min( dt_new, 1  / zeta_in );

        init_rnd( latt.n_tot(  ), 64 );
        set_core( args,         latt );
        return;
    };

    template< class latt_T > __host__
    void set_core( const input & args, const latt_T & latt )
    {
        const auto & pre = args.get_prefixes(  );
        if( pre.find( "cond_nucleus" ) == pre.end(  ) )
            return;
        
        std :: vector  < std::string > sp_all ;
        args( "adsorption", "species", sp_all );

        std::string sp_cond = args.get< std::string >
            ( "cond_nucleus", "species", "" );
        int  i_sp( 0 );
        for( i_sp = 0; i_sp < n_spe; ++ i_sp )
            if( sp_cond == sp_all[ i_sp ] )
                break;

        type::idx_t loc( 0, 0 ), i( 0, 0 );
        args( "cond_nucleus", "location", loc.x );
        const  auto rad = args.get< int >
            ( "cond_nucleus", "radius",       0 );

        srand( time( 0 ) );
        const auto f_rand = [ & ] (  )
        {
            return ( type::float_t )( rand(  ) ) / RAND_MAX;
        };
            
        const auto n( latt.n_cell );
        for( i[ 0 ] = loc[ 0 ] - rad;
             i[ 0 ] < loc[ 0 ] + rad; ++ i[ 0 ] )
            for( i[ 1 ] = loc[ 1 ] - rad;
                 i[ 1 ] < loc[ 1 ] + rad; ++ i[ 1 ] )
                    ( * latt.at( i ) ) = ( i_sp + 1 );
        return;
    };

    __host__ void free(  )
    {
        if( rnd_data != nullptr )
            cudaFree ( rnd_data );
        rnd_data      = nullptr ;
        return;
    };

    __device__ __forceinline__ int  rnd_dispatch(  ) const
    {
        return threadIdx.x + blockIdx.x * blockDim.x;
    };
    __device__ __forceinline__ void rnd_load(  ) const
    {
        const auto  i = rnd_dispatch(  );
        if( i < n_rnd )
            rnd_local = rnd_data[ i ];
    };
    __device__ __forceinline__ void rnd_save(  ) const
    {
        const auto  i = rnd_dispatch(  );
        if( i < n_rnd )
            rnd_data[ i ] = rnd_local;
    };
    
    __device__ __forceinline__ auto f_rnd(  ) const
    {
        return curand_uniform( & rnd_local );
    };

    template    < class nbr_T, class latt_T > __device__
    bool impinge( nbr_T & dat, const latt_T & l ) const
    {
        
        dt_new = cfl_in / ( zeta_in  * l.n_tot(  ) );
        
        if( dat.center(  ) > 0 )
            return false;
        if( f_rnd(  ) < 1 - expf
          ( -zeta_in * ( * l.p_dt_old ) ) )
        {
            const   auto x_rnd = f_rnd   (   );
            for( int i = 0;  i < n_spe; ++ i )
                if(  x_rnd < x_gas_cumu[ i ] )
                {
                    if( i_iter != 0 )
                        dt_new  = 1 / nu_dad_sp[ i ]
                            * expf( beta * e_ads_sp[ i ] );
                    atomicCAS( l.at( dat.idx ), 0, i + 1 );
                    dat.center(  ) = i + 1;
                    break;
                }
        }
        return true;
    };

    template    < class nbr_T, class latt_T > __device__
    int  hopping( nbr_T & dat, const latt_T & l ) const
    {
        const   auto & c( dat.center(  ) );
        if( c == 0 )
            return 0;
        
        type ::float_t  e_ngbr  ( 0 );
        int n_hop( 0 ), nb_count( 0 );
        dat.map( [ & ] ( const auto & s, const auto & di )
        {
            n_hop    += ( s == 0 );
            nb_count += ( s >  0 );
        }   );
              
        dat.map( [ & ] ( const auto & s, const auto & di )
        {
            if( s > 0 && di[ 0 ] * di[ 1 ] == 0 )
                e_ngbr += e_ngbr_sp[ c - 1 ][ s - 1 ]
                       * ( nb_count <= n_nb_max );
        }   );
        if( nb_count > n_nb_max )
            e_ngbr *= ( decltype( e_ngbr ) )
                      ( n_nb_max ) / nb_count;
        const auto    dt  = ( * l.p_dt_old ) ;
        e_ngbr           -= e_corr  [ c - 1 ];
        const auto e_ads  = e_ads_sp[ c - 1 ] + e_ngbr;
        const auto e_hop  = e_hop_sp[ c - 1 ] + e_ngbr;
        
        const auto zeta_hop  = nu_hop_sp[ c - 1 ]
                             * expf( - beta     * e_hop );
        const auto zeta_dad  = nu_dad_sp[ c - 1 ]
                             * expf( - beta     * e_ads );

        if( i_iter != 0 )
            dt_new = 1 / fmaxf( zeta_dad,
                                zeta_hop * ( n_hop > 0 ) );
        if( f_rnd(  ) < expf
                 ( -( zeta_hop + zeta_dad ) * dt ) )
            return 1;

        int res( 1 );
        if ( f_rnd( ) < zeta_dad / ( zeta_hop + zeta_dad ) )
        {
             atomicCAS( l.at( dat.idx ), c, 0 );
             dt_new = cfl_in / ( zeta_in * l.n_tot(  ) );
             res    = 0;
        }
        else
        {
            int count( n_hop * f_rnd(  ) );
            dat.map
            ( [ & ] ( const auto & s, const auto & di )
            {
                if( s != 0 )
                    return ;
                if( count == 0 )
                {
                    auto idx = dat.idx;
                    idx     +=      di;
                    if( s == atomicCAS( l.at( idx ), s, c ))
                        atomicCAS ( l.at( dat.idx ), c, 0 );
                }
                -- count ;
            }   );
        }
        return res;
    };
};
    
};
