/*******************************************************************************

Sigma_2P_corrs_from_modes.cpp

This algorithm computes and stores two-particle 2-point Euclidean correlation
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
const int TMAX = 42;

const int N_TOTAL   =  1;
const int N_REL_OUT =  1;
const int N_REL_IN  =  5;

/***** JACKKNIFE PARAMETERS ******/


const int JK_BINS = 1000;
const int BIN_LEN = CONF / JK_BINS;
const double JK_DENM = 1. / ( BIN_LEN );

array<array<array<double,ORDER>,MAX_MODE>,TIME> re_FT_field;
array<array<array<double,ORDER>,MAX_MODE>,TIME> im_FT_field;


/*
*******************************************************************************/

//////
//
// METHOD:  Two_particle
// Takes:   time, spin components, momenta
// Returns: Two particle functional at a given lattice site
// Uses:    PENDING

array<double,2> Two_particle( int t, int a, int b, int n_rel ) {

  double re_operator, im_operator;

  int sign_n_a = 1-2*signbit( n_rel ),
      sign_n_b = 1-2*signbit( N_TOTAL-n_rel );

  int n_a = abs( n_rel ),
      n_b = abs( N_TOTAL-n_rel );

  double Re_Sa = re_FT_field[t][n_a][a],
         Im_Sa = sign_n_a * im_FT_field[t][n_a][a],
         Re_Sb = re_FT_field[t][n_b][b],
         Im_Sb = sign_n_b * im_FT_field[t][n_b][b];

  re_operator = Re_Sa * Re_Sb - Im_Sa * Im_Sb;
  im_operator = Re_Sa * Im_Sb + Im_Sa * Re_Sb;

  return { re_operator, im_operator };
}

// This 'Two_particle' method computes the two-particle
// functional using the corresponding one-particle modes.
//
//////

//////
//
// METHOD:  Re_two_particle_iso_1_all
// Takes:   time, momenta
// Returns: Two particle functional with isospin 1 (real part)
// Uses:    Re_two_particle, Im_two_particle

array<double,2> Two_particle_iso1_a( int t, int n_rel, int a ) {

  array<double,2> operator_bc, O_bc_aux, O_cb_aux;

  int b = ( a + 1 ) % ORDER, c = ( a + 2 ) % ORDER;

  O_bc_aux = Two_particle( t, b, c, n_rel );
  O_cb_aux = Two_particle( t, c, b, n_rel );

  operator_bc[0] = ( O_bc_aux[0] - O_cb_aux[0] );
  operator_bc[1] = ( O_bc_aux[1] - O_cb_aux[1] );

  return operator_bc;
}

// This 'Re_two_particle_iso_1_all' method computes the real part of the
// two-particle functional with isospin I=1 exploiting all different
// combinations.
//
//////

int main() {
  //
  // 01 - Since a large number of configurations may have been generated, we
  //      require to allocate memory for the sampled values of the correlation
  //      functions, this is done in order to avoid a stack overflow. Two
  //      variables are required, one for the real and another for the imaginary
  //      parts.

  double * re_corrs = new double[TMAX];
  for ( int t = 0; t < TMAX; t++ ){
    re_corrs[t] = 0.0;
  }

  double * im_corrs = new double[TMAX];
  for ( int t = 0; t < TMAX; t++ ){
    im_corrs[t] = 0.0;
  }

  string re_name = "./2pt/2P/I1/N"+to_string( N_TOTAL )+"/re_two_corr_i",
         im_name = "./2pt/2P/I1/N"+to_string( N_TOTAL )+"/im_two_corr_i",
         aux     = to_string( N_REL_OUT )+"_j"+to_string( N_REL_IN )
                   +"_TAVG"+to_string( TAVG ),
         type    = "_new.bn";

  // Create file for saving the jackknife ensemble of correlation functions.
  ofstream write_re_corr( re_name+aux+type );
  ofstream write_im_corr( im_name+aux+type );

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

    array<double,2> O_bc_out, O_bc_in;

    for ( int t = 0; t < TMAX; t++) {
      //
      for ( int t0 = 0; t0 < TAVG; t0++) {
        //
        for ( int a = 0; a < ORDER; a++) {

          //
          O_bc_out = Two_particle_iso1_a( (t+t0)%TIME, N_REL_OUT, a ),
          O_bc_in  = Two_particle_iso1_a( t0, N_REL_IN, a );

          re_corrs[t] += ( O_bc_out[0] * O_bc_in[0] + O_bc_out[1] * O_bc_in[1] );
          im_corrs[t] += ( O_bc_out[1] * O_bc_in[0] - O_bc_out[0] * O_bc_in[1] );
          //
        }
        //
      }
      //
    }

    //
    if ( !( (m+1)%BIN_LEN ) ) {
      //
      for ( int t = 0; t < TMAX; t++) {
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
  delete[] re_corrs; delete[] im_corrs;
  //
  return 0;
}
