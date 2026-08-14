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
const int LEN        = 128;     // Space extension of the lattice
const int ORDER      = 3;       // sigma O(n) order
const int CONF       = 500000;  // Number of configurations to be generated
const int MAX_MODE   = 7;
const double BETA    = 1.54; 

const int TAVG =256;
const int TMAX = 11;

const int N_IN       = 3; 
const int N_IN_SIGN  = 1-2*signbit( N_IN ); 

const int N_OUT      = 3;
const int N_OUT_SIGN = 1-2*signbit( N_OUT ); 

const int N_J        = abs( N_IN - N_OUT );
const int N_J_SIGN   = 1-2*signbit( N_IN - N_OUT );

const int IN   = 0;
const int OUT  = 1; 


/***** JACKKNIFE PARAMETERS ******/


const int JK_BINS = 1000;
const int BIN_LEN = CONF / JK_BINS;
const double JK_DENM = 1. / ( BIN_LEN );

array<array<array<double,ORDER>,MAX_MODE>,TIME> re_FT_field;
array<array<array<double,ORDER>,MAX_MODE>,TIME> im_FT_field;

array<array<array<double,ORDER>,MAX_MODE>,TIME> re_FT_J;
array<array<array<double,ORDER>,MAX_MODE>,TIME> im_FT_J;


/*
*******************************************************************************/

int main() {

  //
  // 01 - Since a large number of configurations may have been generated, we
  //      require to allocate memory for the sampled values of the correlation
  //      functions, this is done in order to avoid a stack overflow. Two
  //      variables are required, one for the real and another for the imaginary
  //      parts.

  double * re_corrs = new double[2*TMAX-1];
  for ( int t = 0; t < 2*TMAX-1; t++ ){
    re_corrs[t] = 0.0;
  }

  double * im_corrs = new double[2*TMAX-1];
  for ( int t = 0; t < 2*TMAX-1; t++ ){
    im_corrs[t] = 0.0;
  }

  string re_name = "./3pt/1P/re_1P1_corr",
         im_name = "./3pt/1P/im_1P1_corr",
         aux     = "_N_OUT_"+to_string( N_OUT )
                   +"_N_IN_" +to_string( N_IN )
                   +"_TMAX"+to_string( TMAX )+"_TAVG"+to_string( TAVG ),
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

  ///////////////////////////////////////////////////////////////////////
  //// NEW    

  ifstream input_re_J_modes;    // Will store the input file 're_modes.dat'
  input_re_J_modes.open( "./re_J_modes.dat" );
  string line_re_J_modes;

  ifstream input_im_J_modes;    // Will store the input file 'im_modes.dat'
  input_im_J_modes.open( "./im_J_modes.dat" );
  string line_im_J_modes;          
  
  //// NEW                      
  ///////////////////////////////////////////////////////////////////////

  for ( int m = 0; m < CONF; m++) {
    //
    getline( input_re_modes, line_re_modes );
    stringstream ss_re_modes( line_re_modes );
    string row_re_modes;

    getline( input_im_modes, line_im_modes );
    stringstream ss_im_modes( line_im_modes );
    string row_im_modes;

    /////////////////////////////////////////////////////////////////////
    //// NEW 

    getline( input_re_J_modes, line_re_J_modes );
    stringstream ss_re_J_modes( line_re_J_modes );
    string row_re_J_modes;

    getline( input_im_J_modes, line_im_J_modes );
    stringstream ss_im_J_modes( line_im_J_modes );
    string row_im_J_modes;

    //// NEW 
    /////////////////////////////////////////////////////////////////////

    for ( int t = 0; t < TIME; t++) {
      //
      getline( ss_re_modes, row_re_modes, ';' );

      stringstream ss_row_re_modes( row_re_modes );
      string col_re_modes;
      //
      getline( ss_im_modes, row_im_modes, ';' );

      stringstream ss_row_im_modes( row_im_modes );
      string col_im_modes;

      ///////////////////////////////////////////////////////////////////
      //// NEW 

      getline( ss_re_J_modes, row_re_J_modes, ';' );

      stringstream ss_row_re_J_modes( row_re_J_modes );
      string col_re_J_modes;
      //
      getline( ss_im_J_modes, row_im_J_modes, ';' );

      stringstream ss_row_im_J_modes( row_im_J_modes );
      string col_im_J_modes;

      //// NEW 
      ///////////////////////////////////////////////////////////////////

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

        /////////////////////////////////////////////////////////////////
        //// NEW
        getline( ss_row_re_J_modes, col_re_J_modes, ' ' );

        stringstream ss_col_re_J_modes( col_re_J_modes );
        string re_J_component;

        array<double,ORDER> re_J_mode;
        //
        getline( ss_row_im_J_modes, col_im_J_modes, ' ' );

        stringstream ss_col_im_J_modes( col_im_J_modes );
        string im_J_component;

        array<double,ORDER> im_J_mode;

        //// NEW
        /////////////////////////////////////////////////////////////////

        for ( int k = 0; k < ORDER; k++) {
          //
          getline( ss_col_re_modes, re_component, ',' );
          re_mode[k] = stod( re_component );
          //
          getline( ss_col_im_modes, im_component, ',' );
          im_mode[k] = stod( im_component );
          //

          ///////////////////////////////////////////////////////////////
          //// NEW
          getline( ss_col_re_J_modes, re_J_component, ',' );
          re_J_mode[k] = stod( re_J_component );
          //
          getline( ss_col_im_J_modes, im_J_component, ',' );
          im_J_mode[k] = stod( im_J_component );
          //// NEW
          ///////////////////////////////////////////////////////////////
        }
        //
        re_FT_field[t][n] = re_mode;
        im_FT_field[t][n] = im_mode;
        //

        /////////////////////////////////////////////////////////////////
        //// NEW
        re_FT_J[t][n] = re_J_mode;
        im_FT_J[t][n] = im_J_mode;
        //// NEW
        /////////////////////////////////////////////////////////////////
      }
      //
    }
    //

    
    array<array<double,ORDER>,2> re_operators, im_operators;
    

    for ( int t0 = 0+TMAX+1; t0 < TAVG+TMAX+1; t0++ ) {
    //for ( int t0 = 0; t0 < TAVG; t0++ ) {
      //
      for ( int a = 0; a < ORDER; a++ ) {
        //

        re_operators[OUT][a] = re_FT_field[(TMAX+t0)%TIME][abs(N_OUT)][a]; 
        im_operators[OUT][a] = N_OUT_SIGN * im_FT_field[(TMAX+t0)%TIME][abs(N_OUT)][a]; 

        re_operators[IN][a]  = re_FT_field[(TIME-TMAX+t0)%TIME][abs(N_IN)][a]; 
        im_operators[IN][a]  = - N_IN_SIGN * im_FT_field[(TIME-TMAX+t0)%TIME][abs(N_IN)][a]; 

        //
      }
      //
      for ( int t = -TMAX+1; t < TMAX; t++) {  // UNCOMMENT 
      //for ( int t = 1; t < 2*TMAX; t++) {  // DELETE!!!
        //
        double reC_3pt = 0.0, imC_3pt = 0.0; 

        for ( int a = 0; a < ORDER; a++) {

          //// Positive permutations (a,b,c) = (0,1,2), (1,2,0), (2,0,1). 
          int b = (a+1)%ORDER, c = (a+2)%ORDER;
          
          //double re_j0_bc = re_FT_J[ (t+t0)%TIME ][N_J][a], // DELETE!!! 
          //       im_j0_bc = N_J_SIGN * im_FT_J[ (t+t0)%TIME ][N_J][a], // DELETE!!! 
          double re_j0_bc = re_FT_J[ (TIME+t+t0)%TIME ][N_J][a], // UNCOMMENT 
                 im_j0_bc = N_J_SIGN * im_FT_J[ (TIME+t+t0)%TIME ][N_J][a], // UNCOMMENT 
                 re_SbSc  = re_operators[OUT][b] * re_operators[IN][c]
                            - im_operators[OUT][b] * im_operators[IN][c],
                 im_SbSc  = im_operators[OUT][b] * re_operators[IN][c]
                            + re_operators[OUT][b] * im_operators[IN][c];
          //
          reC_3pt += re_SbSc * re_j0_bc - im_SbSc * im_j0_bc; 
          imC_3pt += re_SbSc * im_j0_bc + im_SbSc * re_j0_bc;

          //// Negative permutations (a,b,c) = (0,2,1), (1,0,2), (2,1,0). 
          //// j0_bc has a relative sign with respect to the positive 
          //// permutations. 
          //c = (a+1)%ORDER; b = (a+2)%ORDER;
          
          ////re_j0_bc = -re_FT_J[ (t+t0)%TIME ][N_J][a];  // DELETE!!!
          ////im_j0_bc = -N_J_SIGN * im_FT_J[ (t+t0)%TIME ][N_J][a]; // DELETE!!!
          //re_j0_bc = -re_FT_J[ (TIME+t+t0)%TIME ][N_J][a];  // UNCOMMENT
          //im_j0_bc = -N_J_SIGN * im_FT_J[ (TIME+t+t0)%TIME ][N_J][a]; // UNCOMMENT
          //re_SbSc  = re_operators[OUT][b] * re_operators[IN][c]
          //           - im_operators[OUT][b] * im_operators[IN][c];
          //im_SbSc  = im_operators[OUT][b] * re_operators[IN][c]
          //           + re_operators[OUT][b] * im_operators[IN][c];
          //
          //reC_3pt += re_SbSc * re_j0_bc - im_SbSc * im_j0_bc; 
          //imC_3pt += re_SbSc * im_j0_bc + im_SbSc * re_j0_bc;

          //if (m==0 && t0==TMAX+1 && t==0)
          //{
          //  cout << b << ", " << c << ", J=" << re_j0_bc << endl;
          //}
          //
        }

        //
        re_corrs[ TMAX+t-1 ] += reC_3pt; // UNCOMMENT 
        im_corrs[ TMAX+t-1 ] += imC_3pt; // UNCOMMENT 
        //re_corrs[ t-1 ] += reC_3pt; // DELETE!!
        //im_corrs[ t-1 ] += imC_3pt; // DELETE!!
        //
      } // Closes loop over time for the current. 
      //
    } // Closes loop over time insertion. 


    //
    if ( !( (m+1)%BIN_LEN ) ) {
      //
      for ( int t = 0; t < 2*TMAX-1; t++) {
        //
        double bn_reC_3pt = re_corrs[t] / double( BIN_LEN*TAVG ),
               bn_imC_3pt = im_corrs[t] / double( BIN_LEN*TAVG );

        write_re_corr <<fixed<<setprecision(18)<< bn_reC_3pt <<" ";
        write_im_corr <<fixed<<setprecision(18)<< bn_imC_3pt <<" ";
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
