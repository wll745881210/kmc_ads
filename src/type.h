#pragma once

namespace type
{
////////////////////////////////////////////////////////////
//

using byte_t   =    std::byte;
using float_t  =        float;
using float2_t =       double;

struct idx_t
{
    int x[ 2 ];

    __host__ __device__ idx_t(  ) = default;

    __host__ idx_t( const int & n_i, const int & n_j )
    {
        x[ 0 ] = n_i;
        x[ 1 ] = n_j;
    };

    __forceinline__ __host__ __device__
    auto & operator[  ] ( const int & i )
    {
        return x[ i ];
    };

    __forceinline__ __host__ __device__
    const auto & operator[  ] ( const int & i ) const
    {
        return x[ i ];
    };

    __forceinline__ __host__ __device__
    void operator += ( const idx_t & i )
    {
        x[ 0 ] += i.x[ 0 ];
        x[ 1 ] += i.x[ 1 ];
    };
};

};
