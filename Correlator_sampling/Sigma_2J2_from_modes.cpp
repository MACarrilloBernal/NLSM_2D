/*******************************************************************************

Sigma_2J2_from_modes.cpp

Analysis algorithm for the two-particle matrix elements of the non-linear O(3)
sigma-model in the iso-vector channel.

References:

[1] Luscher and Wolff 1990
    Nucl.Phys.B 339 (1990) 222-252

2•05•25                                                       MA Carrillo-Bernal

*******************************************************************************/

/***** HEADERS ******/

#define _USE_MATH_DEFINES ; // for calling Pi

#include <iostream>   // cin, cout
#include <array>      // array<TYPE,SIZE>
#include <set>        // set<TYPE>
#include <vector>     // vector<TYPE>
#include <fstream>    // ifstream, ofstream, open, close
#include <string>     // string, getline
#include <sstream>    // stringstream
#include <iomanip>    // setprecision
#include <cmath>      // atan, acos, cos, sin, M_PI

using namespace std;

/******************************************************************************/

/***** LATTICE VOLUME AND NUMBER OF CONFIGURATIONS ******/

// GLOBAL CONSTANTS

const int TIME       =  256;      // Time extension of the lattice
const int LEN        =  128;      // Space extension of the lattice
const int ORDER      =    3;      // sigma O(n) order
const int CONF       = 100;    // Number of configurations to be generated
const int MAX_MODE   =    7; 

const double INV_L    = 1.0/LEN;             // Inverse of the length
const double UNIT     = 2.0 * M_PI * INV_L;
 
const int STATE_IN         = 0;
const int N_TOTAL_IN       = 0;
const int N_TOTAL_IN_SIGN  = 1-2*signbit( N_TOTAL_IN );
const int SKIP_IN          = 2;

const int STATE_OUT        = 0;
const int N_TOTAL_OUT      = 0;
const int N_TOTAL_OUT_SIGN = 1-2*signbit( N_TOTAL_OUT );
const int SKIP_OUT         = 1;

const int N_J      = abs(N_TOTAL_IN - N_TOTAL_OUT);
const int N_J_SIGN = 1-2*signbit( N_TOTAL_IN - N_TOTAL_OUT );

const int TMAX        =    4;
const int TAVG        =  256;

const int IN          =   0;
const int OUT         =   1;


const int SIZE = 4; // NEW SIZE OF THE BASIS

/***** BINNING PARAMETERS ******/

const int NOFBINS     = 10;
const int BIN_LEN     = CONF / NOFBINS;
const double AVG_DENM = 1.0 / ( BIN_LEN * TAVG );

array<array<array<double,ORDER>,LEN>,TIME> field;

/***** GLOBAL VARIABLES ******/

array<array<array<double,ORDER>,MAX_MODE>,TIME> re_FT_field;
array<array<array<double,ORDER>,MAX_MODE>,TIME> im_FT_field;

array<array<array<double,ORDER>,MAX_MODE>,TIME> re_FT_J;
array<array<array<double,ORDER>,MAX_MODE>,TIME> im_FT_J;


/******************************************************************************/

//////
//
// METHOD:  Two_particle
// Takes:   time, spin components, momenta
// Returns: Two particle functional at a given lattice site
// Uses:    PENDING

array<double,2> Two_particle( int t, int a, int b, int n_rel, int N_TOTAL ) {

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

array<double,2> Two_particle_iso1_a( int t, int n_rel, int N_TOTAL, int a ) {

  array<double,2> operator_bc, O_bc_aux, O_cb_aux;

  int b = ( a + 1 ) % ORDER, c = ( a + 2 ) % ORDER;

  O_bc_aux = Two_particle( t, b, c, n_rel, N_TOTAL );
  O_cb_aux = Two_particle( t, c, b, n_rel, N_TOTAL );

  operator_bc[0] = ( O_bc_aux[0] - O_cb_aux[0] );
  operator_bc[1] = ( O_bc_aux[1] - O_cb_aux[1] );

  return operator_bc;
}

// This 'Re_two_particle_iso_1_all' method computes the real part of the
// two-particle functional with isospin I=1 exploiting all different
// combinations.
//
//////

/******************************************************************************/

int main() {

  ///////////////////////////////
  //// MOVE THESE TO A FILE  :( 
  
  // STATE=0, N=0, re: { 0.9948366334887462, 0.07180032250757065, 0.04934509253736619, 0.05205446468827866 }
  // STATE=0, N=0, im: { 0.0000, 0.0000, 0.0000, 0.0000 }
  //
  // STATE=1, N=0, re: { -0.04790465081342017, 0.9925372869498695, 0.09713180987706949, 0.056037482102290405 }
  // STATE=1, N=0, im: { 0.0000, 0.0000, 0.0000, 0.0000 }
  //
  // STATE=2, N=0, re: { -0.01760975565755451, -0.06424138308260273, 0.9942515316502314, 0.08382558172652199 }
  // STATE=2, N=0, im: { 0.0000, 0.0000, 0.0000, 0.0000 }
  
  // STATE=0, N=1, re: { 0.998069610186438, 0.0473948019050169, 0.0347315132677358, 0.017872764535657965 }
  // STATE=0, N=1, im: { 0.0, 0.008915768205632425, -0.0014229921217819878, 0.0018232893184000013 }
  //
  // STATE=1, N=1, re: { -0.033727765086541926, 0.9935173608446186, 0.08535397613642576, 0.06646828059897995 }
  // STATE=1, N=1, im: { -0.005521456702860439, 0.0, -0.0054614867830784525, -0.0046468203630489055 }
  //
  // STATE=2, N=1, re: { -0.012352574595499025, -0.05782761442553825, 0.9937950765690159, 0.09041979382930856 }
  // STATE=2, N=1, im: { -0.0006904788598089764, -0.002287704966727964, 0.0, 0.026313435355498762 }
  
  // STATE=0, N=2, re: { 0.996090728982676, 0.057907056350858625, 0.04902755919545753, 0.024392131176143462 }
  // STATE=0, N=2, im: { 0.0, -0.036663214507518825, -0.01026957847776186, -0.001008372700454157 }
  //
  // STATE=1, N=2, re: { -0.04934344807046051, 0.9925884776958016, 0.08916869560036768, 0.05959320922454863 }
  // STATE=1, N=2, im: { -0.014614613144268606, 0.0, 0.024627000973697327, 0.0031594023524868157 }
  //
  // STATE=2, N=2, re: { -0.019897829942753877, -0.0630071653012266, 0.9923592487984976, 0.09433075330079259 }
  // STATE=2, N=2, im: { -0.0012688873096977206, 0.0033477432270087332, 0.0, 0.044106297887884095 }
  
  // STATE=0, N=3, re: { 0.9979750198478156, 0.02345361453609617, 0.012953562290429986, 0.015011811092418595 }  
  // STATE=0, N=3, im: { 0.0, 0.05531236913861418, 0.005594774961046946, -0.003156724104087579 }  
  //
  // STATE=1, N=3, re: { -0.034160103851008, 0.9901371270097683, 0.052785611872890714, 0.03336649537213064 }
  // STATE=1, N=3, im: { -0.11371747391930631, 0.0, 0.04008879937476986, -0.002884381441618907 }
  //
  // STATE=2, N=3, re: { -0.01394939507257657, -0.05795207925952377, 0.9921792930825236, 0.10777332346745006 }
  // STATE=2, N=3, im: { -0.004030068706095253, 0.01004238445963179, 0.0, -0.017159144997689873 }

  array<double,SIZE> re_weights_out = { 0.9948366334887462, 0.07180032250757065, 0.04934509253736619, 0.05205446468827866 },
                     im_weights_out = { 0.0000, 0.0000, 0.0000, 0.0000 },
                     re_weights_in  = { 0.9948366334887462, 0.07180032250757065, 0.04934509253736619, 0.05205446468827866 },
                     im_weights_in  = { 0.0000, 0.0000, 0.0000, 0.0000 };
  
  //// MOVE THESE TO A FILE :( 
  ///////////////////////////////

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

  string re_name = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/0_Configurations/TEST_100/re_2J2_corr",
         im_name = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/0_Configurations/TEST_100/im_2J2_corr",
         aux     = "_OUT_E"+to_string( STATE_OUT )+"_N"+to_string( N_TOTAL_OUT )
                   +"_IN_E"+to_string( STATE_IN  )+"_N"+to_string( N_TOTAL_IN  )
                   +"_TMAX"+to_string( TMAX )+"_TAVG"+to_string( TAVG ),
         type    = ".bn";

  // Create file for saving the jackknife ensemble of correlation functions.
  ofstream write_re_corr( re_name+aux+type );
  ofstream write_im_corr( im_name+aux+type );

  //
  ifstream input_re_modes;    // Will store the input file 're_modes.dat'
  input_re_modes.open( "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/0_Configurations/re_modes.dat" );
  string line_re_modes;

  ifstream input_im_modes;    // Will store the input file 're_modes.dat'
  input_im_modes.open( "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/0_Configurations/im_modes.dat" );
  string line_im_modes;

  ///////////////////////////////////////////////////////////////////////
  //// NEW    

  ifstream input_re_J_modes;    // Will store the input file 're_modes.dat'
  input_re_J_modes.open( "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/0_Configurations/re_J_modes.dat" );
  string line_re_J_modes;

  ifstream input_im_J_modes;    // Will store the input file 'im_modes.dat'
  input_im_J_modes.open( "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/0_Configurations/im_J_modes.dat" );
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
    

    for ( int t0 = 0; t0 < TAVG; t0++ ) {
    //for ( int t0 = 0; t0 < TAVG; t0++ ) {
      //
      for ( int a = 0; a < ORDER; a++ ) {
        //

        re_operators[OUT][a] = 0.0; 
        im_operators[OUT][a] = 0.0;

        re_operators[IN][a]  = 0.0; 
        im_operators[IN][a]  = 0.0; 

        for ( int v = 0; v < SIZE; v++ ) {
          //

          array<double,2> O_bc_out = Two_particle_iso1_a( (TMAX+t0)%TIME, v+SKIP_OUT, N_TOTAL_OUT, a ), 
                          O_bc_in  = Two_particle_iso1_a( (TIME-TMAX+t0)%TIME, v+SKIP_IN, N_TOTAL_IN, a );

          re_operators[OUT][a] += re_weights_out[v] * O_bc_out[0] 
                                  + im_weights_out[v] * O_bc_out[1] ; 
          im_operators[OUT][a] += N_TOTAL_OUT_SIGN 
                                  * ( re_weights_out[v] * O_bc_out[1]
                                      - im_weights_out[v] * O_bc_out[0] ); 

          re_operators[IN][a]  += re_weights_in[v] * O_bc_in[0]
                                  + im_weights_in[v] * O_bc_in[1] ; 
          im_operators[IN][a]  -= N_TOTAL_IN_SIGN 
                                  * ( re_weights_in[v] * O_bc_in[1] 
                                      - im_weights_in[v] * O_bc_in[0] ); 
          //
        }

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

          //if (m==0 && t0==TMAX+1 && t==0)
          //{
          //  cout << b << ", " << c << ", J=" << re_j0_bc << ", m=" << m << endl;  // DELETE
          //  cout << b << ", " << c << ", R=" << re_SbSc << ", m=" << m << endl;  // DELETE
          //  cout << b << ", " << c << ", C3pt=" << reC_3pt << ", m=" << m << endl;  // DELETE
          //}
          

          //// Negative permutations (a,b,c) = (0,2,1), (1,0,2), (2,1,0). 
          //// j0_bc has a relative sign with respect to the positive 
          //// permutations. 
          //c = (a+1)%ORDER; b = (a+2)%ORDER;  // COMMENT
          
          ////re_j0_bc = -re_FT_J[ (t+t0)%TIME ][N_J][a];  // DELETE!!!
          ////im_j0_bc = -N_J_SIGN * im_FT_J[ (t+t0)%TIME ][N_J][a]; // DELETE!!!
          //re_j0_bc = -re_FT_J[ (TIME+t+t0)%TIME ][N_J][a];  // COMMENT
          //im_j0_bc = -N_J_SIGN * im_FT_J[ (TIME+t+t0)%TIME ][N_J][a]; // COMMENT
          //re_SbSc  = re_operators[OUT][b] * re_operators[IN][c]       // COMMENT
          //           - im_operators[OUT][b] * im_operators[IN][c];    // COMMENT  
          //im_SbSc  = im_operators[OUT][b] * re_operators[IN][c]       // COMMENT 
          //           + re_operators[OUT][b] * im_operators[IN][c];    // COMMENT 
          
          //reC_3pt += re_SbSc * re_j0_bc - im_SbSc * im_j0_bc;         // COMMENT  
          //imC_3pt += re_SbSc * im_j0_bc + im_SbSc * re_j0_bc;         // COMMENT 

          //if (m==0 && t0==TMAX+1 && t==0)
          //{
          //  cout << b << ", " << c << ", J=" << re_j0_bc << endl;
          //}
          //
        }

        //if (t==0 & m==0)
        //{
        //  cout << reC_3pt << endl;
        //}
        
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
