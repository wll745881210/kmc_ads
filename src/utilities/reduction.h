#pragma once

namespace utils
{
////////////////////////////////////////////////////////////
// Atomic reduction; pass the operation via template arg.

template < class F, F ( * f ) ( F, F ) > __device__
std::enable_if_t< std::is_same< F,  float >::value, void >
atomic_red( F * des, const F x )
{
    int * d_i  = ( int * )  des;
    int   old  = ( * d_i );
    int   assumed( 0 );
    do
    {
        assumed = old;
        old     = atomicCAS
                ( d_i, assumed, __float_as_int
                ( f( x, __int_as_float( assumed ) ) ) );
    }   while( assumed != old );
}

template < class F, F ( * f ) ( F, F ) > __device__
std::enable_if_t< std::is_same< F, double >::value, void >
atomic_red( F * des, const F x )
{
    typedef unsigned long long int __ul;
    __ul * d_i( ( __ul * ) des );
    __ul   old( * d_i );
    __ul   a  (   0   );
    do
    {
        a = old;
        const auto & b = f( x,__longlong_as_double( a ) );
        const auto & c =__ul( __double_as_longlong( b ) );
        old = atomicCAS( d_i, a, c );
    }   while( a != old );
    return;
}

////////////////////////////////////////////////////////////
// Block-wise reduction operation

template< typename F, F ( * f )( F, F ) >
__device__ void block_reduce( F * x, int n )
{
    __syncthreads(  );
    const auto & i = threadIdx.x;
    while( n > 1 )
    {
        const int & dn = ( n + 1 )   >> 1;
        const int & di =  dn + i ;
        if( di < n )
            x[ i ] = f( x[ i ], x[ di ] );
        __syncthreads(  );
        n = dn;
    }
    return;
}

template< class V, class fun_T > __device__
void block_reduce( V & x, const fun_T & f, int n )
{
    __syncthreads(  );
    const auto & i = threadIdx.x;
    while( n > 1 )
    {
        const int & dn = ( n + 1 )   >> 1;
        const int & di =  dn + i ;
        if( di < n )
            x[ i ] = f( x[ i ], x[ di ] );
        __syncthreads(  );
        n = dn;
    }
    return;
}

};                              // namespace utils
