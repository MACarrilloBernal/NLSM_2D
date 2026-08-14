/*******************************************************************************

Sigma_corrs_from_modes.cpp

This algorithm computes and stores one-particle 2-point Euclidean correlation
functions of the non-linear O(3) sigma-model.

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

const int TIME       = 256;     // Time extension of the lattice
const int LEN        = 128;      // Space extension of the lattice
const int ORDER      = 3;       // sigma O(n) order
const int CONF       = 500000;  // Number of configurations to be generated
const int MAX_MODE   = 7;

const int TAVG =256;
const int TLIM = 96;

const int P =  0;


/***** JACKKNIFE PARAMETERS ******/


const int JK_BINS = 1000;
const int BIN_LEN = CONF / JK_BINS;
const double JK_DENM = 1. / ( BIN_LEN );

array<array<array<double,ORDER>,MAX_MODE>,TIME> re_FT_field;
array<array<array<double,ORDER>,MAX_MODE>,TIME> im_FT_field;

/*
*******************************************************************************/

int main() {
  //
  // 01 - Since a large number of configurations may have been generated, we
  //      require to allocate memory for the sampled values of the correlation
  //      functions, this is done in order to avoid a stack overflow. Two
  //      variables are required, one for the real and another for the imaginary
  //      parts.

  double * re_corrs = new double[TIME];
  for ( int t = 0; t < TLIM; t++ ){
    re_corrs[t] = 0.0;
  }

  double * im_corrs = new double[TIME];
  for ( int t = 0; t < TLIM; t++ ){
    im_corrs[t] = 0.0;
  }

  string re_name = "./2pt/1P/re_one_corr_p",
         im_name = "./2pt/1P/im_one_corr_p",
         a       = to_string( P ),
         type    = "_TAVG"+to_string( TAVG )+".bn";

  // Create file for saving the jackknife ensemble of correlation functions.
  ofstream write_re_corr( re_name+a+type );
  ofstream write_im_corr( im_name+a+type );

  //
  ifstream input_re_modes;    // Will store the input file 're_modes.dat'
  input_re_modes.open( "./re_modes.dat" );
  string line_re_modes;

  ifstream input_im_modes;    // Will store the input file 're_modes.dat'
  input_im_modes.open( "./im_modes.dat" );
  string line_im_modes;

  for ( int m = 0; m < CONF; m++) {
    //
    getline( input_re_modes, line_re_modes );
    stringstream ss_re_modes( line_re_modes );
    string row_re_modes;

    getline( input_im_modes, line_im_modes );
    stringstream ss_im_modes( line_im_modes );
    string row_im_modes;

    for ( int t = 0; t < TIME; t++) {
      //
      getline( ss_re_modes, row_re_modes, ';' );

      stringstream ss_row_re_modes( row_re_modes );
      string col_re_modes;
      //
      getline( ss_im_modes, row_im_modes, ';' );

      stringstream ss_row_im_modes( row_im_modes );
      string col_im_modes;
      //
      for ( int n = 0; n < MAX_MODE; n++) {
        //
        getline( ss_row_re_modes, col_re_modes, ' ' );

        stringstream ss_col_re_modes( col_re_modes );
        string re_component;

        array<double,ORDER> re_mode;
        //
        getline( ss_row_im_modes, col_im_modes, ' ' );

        stringstream ss_col_im_modes( col_im_modes );
        string im_component;

        array<double,ORDER> im_mode;

        for ( int k = 0; k < ORDER; k++) {
          //
          getline( ss_col_re_modes, re_component, ',' );
          re_mode[k] = stod( re_component );
          //
          getline( ss_col_im_modes, im_component, ',' );
          im_mode[k] = stod( im_component );
          //
        }
        //
        re_FT_field[t][n] = re_mode;
        im_FT_field[t][n] = im_mode;
        //
      }
      //
    }
    //

    //
    for ( int t = 0; t < TLIM; t++) {
      //
      for ( int k = 0; k < ORDER; k++) {
        //
        for ( int t0 = 0; t0 < TAVG; t0++) {
          //
          re_corrs[t] += ( re_FT_field[(t+t0)%TIME][P][k] * re_FT_field[t0][P][k]
                         + im_FT_field[(t+t0)%TIME][P][k] * im_FT_field[t0][P][k] );
          //
          im_corrs[t] += ( im_FT_field[(t+t0)%TIME][P][k] * re_FT_field[t0][P][k]
                         - re_FT_field[(t+t0)%TIME][P][k] * im_FT_field[t0][P][k] );
          //
        }
        //
      }
      //
    }
    //
    if ( !( (m+1)%BIN_LEN ) ) {
      //
      for ( int t = 0; t < TLIM; t++) {
        //
        double bn_reC_2pt = re_corrs[t] / double( BIN_LEN*TAVG ),
               bn_imC_2pt = im_corrs[t] / double( BIN_LEN*TAVG );

        write_re_corr <<fixed<<setprecision(18)<< bn_reC_2pt <<" ";
        write_im_corr <<fixed<<setprecision(18)<< bn_imC_2pt <<" ";
        re_corrs[t] = 0.0; im_corrs[t] = 0.0;
        //
      }
      //
      write_re_corr << "\n"; write_im_corr << "\n";
      //
    }
    //
  }
  //
  write_re_corr.close(); write_im_corr.close();
  //
  return 0;
}
