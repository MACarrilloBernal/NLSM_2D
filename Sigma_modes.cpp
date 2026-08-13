/*******************************************************************************

Sigma_modes.cpp

This algorithm computes and stores the lowest momentum modes of the non-linear
O(3) sigma-model.

References:

[1] Luscher and Wolff 1990
    Nucl.Phys.B 339 (1990) 222-252

10•31•24                                                      MA Carrillo-Bernal

*******************************************************************************/

/***** HEADERS ******/

#define _USE_MATH_DEFINES ; // for calling Pi

#include <iostream>   // cin, cout
#include <array>      // array<TYPE,SIZE>
#include <fstream>    // ifstream, ofstream, open, close
#include <string>     // string, getline
#include <sstream>    // stringstream
#include <iomanip>    // setprecision
#include <cmath>      // atan, acos, cos, sin, M_PI

using namespace std;

/*
*******************************************************************************/

/***** LATTICE VOLUME AND NUMBER OF CONFIGURATIONS ******/

// GLOBAL VARIABLES

const int TIME        = 256;     // Time extension of the lattice
const int LEN         = 128;      // Space extension of the lattice
const int ORDER       = 3;       // sigma O(n) order
const int CONF        = 1000;  // Number of configurations to be generated

const double INV_L    = 1./LEN;             // Inverse of the length
const double UNIT     = 2.0 * M_PI * INV_L;
const int    MAX_MODE = 8;

array<array<array<double,ORDER>,LEN>,TIME> field;

/*
*******************************************************************************/

//////
//
// METHOD:  Build_spin
// Takes:   Polar and azimuthal  angles
// Returns: A spin vector of magnitude 1
// Uses:    sin, cos

array<double,ORDER> Build_spin( array<double,ORDER-1> angles ){
  array<double,ORDER> spin;
  spin[0] = sin( angles[0] ) * cos( angles[1] );
  spin[1] = sin( angles[0] ) * sin( angles[1] );
  spin[2] = cos( angles[0] );
  return spin;
}

// This 'Build_spin' method reads the polar and azimuthal  angles to generate the
// components of a spin vector of magnitude 1.
//
//////

//////
//
// METHOD:  Re_one_particle
// Takes:   time, spin component
// Returns: One particle functional at a given lattice site (real part)
// Uses:    cos

double Re_one_particle( int t, int component, int n ) {
  double Re_FourierT = 0.0;
  for( int x = 0; x < LEN; x++ ) {
    Re_FourierT += field[t][x][component] * cos( x * n * UNIT );
  }
  return Re_FourierT;
}

// This 'Re_one_particle' method computes the real part of the discrete Fourier
// Transform of the field at a given time 't'.
//
//////

//////
//
// METHOD:  Im_one_particle
// Takes:   time, spin component
// Returns: One particle functional at a given lattice site (imaginary part)
// Uses:    sin

double Im_one_particle( int t, int component, int n ) {
  double Im_FourierT = 0.0;
  for( int x = 0; x < LEN; x++ ) {
    Im_FourierT += field[t][x][component] * sin( - x * n * UNIT );
  }
  return Im_FourierT;
}

// This 'Im_one_particle' method computes the imag part of the discrete Fourier
// Transform of the field at a given time 't'.
//
//////

int main() {

  string re_name = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/0_Configurations/re_modes",
         im_name = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/0_Configurations/im_modes",
         type    = ".dat";

  // Create file for saving the jackknife ensemble of correlation functions.
  ofstream write_re_modes( re_name+type );
  ofstream write_im_modes( im_name+type );

  // 02 - Go over the saved cofigurations and set the lattice variable 'field',
  //      then compute the one-particle functionals and the correlation function
  //      at all times.

  ifstream input_field;              // Will store the input file.
  input_field.open( "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/0_Configurations/field_new2.dat" ); // Open the file with the configurations.
  string line_field;                 // Will store one of the configurations.

  for ( int m = 0; m < CONF; m++) {

    getline( input_field, line_field );  // Read the m-th configuration.
    stringstream ss_field( line_field ); // Parse the m-th configuration.
    string row_field;                    // Will store one time row of the m-th
                                         // configuration.

    for ( int t = 0; t < TIME; t++) {

      getline( ss_field, row_field, ';' );    // Read the t-th time row of the
                                              // m-th configuration.
      stringstream ss_row_field( row_field ); // Parse the t-th time row.
      string col_field;                       // Will store one site in the t-th
                                              // time row of the m-th
                                              // configuration.

      for ( int x = 0; x < LEN; x++) {

        getline( ss_row_field, col_field, ' ' ); // Read the x-th site at the
                                                 // t-th time row of the m-th
                                                 // configuration.
        stringstream ss_col_field( col_field );  // Parse the x-th site.
        string component;                        // Will store one component of
                                                 // the field at (t,x).
        array<double,ORDER-1> angles;            // Will store the angle
                                                 // variables of the spin at the
                                                 // given site.

        for ( int k = 0; k < ORDER-1; k++) {

          // k=0 corresponds to the polar angle.
          // k=1 corresponds to the azimuthal angle.

          getline( ss_col_field, component, ',' ); // Read the k-th angle.
          angles[k] = stod( component );           // Save the k-th angle.
        }
        field[t][x] = Build_spin( angles );  // Set the spin at (t,x) of the
                                             // m-th configuration.
      }
    } // At this point, one field configuration has been stored in 'field'.

    for ( int t = 0; t < TIME; t++ ) {
      for (size_t n = 0; n < MAX_MODE; n++) {
        for ( int k = 0; k < ORDER; k++ ) {
          write_re_modes <<fixed<<setprecision(18)<< Re_one_particle( t%TIME, k, n );
          write_im_modes <<fixed<<setprecision(18)<< Im_one_particle( t%TIME, k, n );

          if ( k < ORDER-1 ) {
            write_re_modes << ","; write_im_modes << ",";
          }
        }
        write_re_modes << " "; write_im_modes << " ";
      }
      write_re_modes << ";"; write_im_modes << ";";
    }
    write_re_modes << "\n"; write_im_modes << "\n";
  } // At this points, the correlation function has been sampled for all the
    // considered configurations of the field and for all correlation times.

  write_re_modes.close(); write_im_modes.close();

  return 0;
}
