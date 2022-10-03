#include "input.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

////////////////////////////////////////////////////////////
//

namespace str_op
{
inline std::string trim
( std::string s, const std::string delim = " \f\n\r\t\v" )
{
    s = s.erase( s.find_last_not_of( delim ) + 1 );
    return s.erase( 0, s.find_first_not_of( delim ) );
}
};

////////////////////////////////////////////////////////////
// Initializer

input::input(  ) : input_file_name( "par.par" ),
                   is_silent      (     true  )
{
    return;
}

input::input( const std::string & file_name,
              const bool        &    silent )
{
    set_file( file_name, silent );
    return;
}

input::~input(  )
{
    return;
}

void input::set_file( const std::string & file_name,
                      const bool        &    silent )
{
    this->input_file_name = file_name;
    this->is_silent       =    silent;
    return;
}

////////////////////////////////////////////////////////////
// Read from file

std::string input::extract_prefix( const std::string & src )
{
    auto i_e =  src.find    ( ']' );
    if(  i_e == std::string::npos )
    {
        std::cerr << "Incorrect input section.\n";
        throw std::runtime_error( "input.cpp" );
    }
    return src.substr( 1, i_e - 1 );
}

void input::read(  )
{
    std::fstream fin( input_file_name.c_str(  ),
                      std::ios::in | std::ios::binary );
    if( ! fin )
    {
        std::cerr << "Unable to open input file: "
                  << input_file_name << std::endl;
        throw std::runtime_error( "input.cpp" );
    }

    std::string item_temp, line_temp, value_temp;
    std::stringstream ss;
    std::string prefix( "" );

    prefix_all.clear(  );
    while( ! fin.eof(  ) )
    {
        getline ( fin, line_temp );
        ss.clear(                );
        ss.str  (      line_temp );
        std::getline   ( ss, line_temp, '#' );
        line_temp = str_op::trim( line_temp );
        if( line_temp.empty(  ) )
            continue;
        if( line_temp[ 0 ] == '[' )
        {
            prefix = extract_prefix ( line_temp );
            prefix_all.insert       (    prefix );
            continue;
        }
        ss.clear(           );
        ss.str  ( line_temp );
        std::getline( ss,      item_temp,   '=' );
        std::getline( ss,     value_temp,   '=' );
        item_temp   = str_op::trim (  item_temp );
        item_map[ key_expand( prefix, item_temp ) ]
                    = str_op::trim ( value_temp );
    }
    fin.close(  );
    return;
}

////////////////////////////////////////////////////////////
// Data access

std::string input::key_expand
( const std::string & pre,  const std::string & key ) const
{
    return ( pre.empty(  ) ? "" : pre + "|" ) + key;
}

void input::msg_not_found
( const std::string & pre, const std::string & key ) const
{
    if( ! is_silent )
        std::cerr << "Entry \"[" + pre + "]: "
            + key + "\" not found; using default values.\n";
    throw std::runtime_error( "input::get" );
    return;
}

bool input::found( const std::string & pre,
                   const std::string & key ) const
{
    return item_map.find( key_expand( pre, key ) ) !=
           item_map. end(  );
}

const std::map<std::string, std::string> &
input::get_item_map(  ) const
{
    return item_map;
}

const std::set<std::string> & input::get_prefixes(  ) const
{
    return prefix_all;
}
