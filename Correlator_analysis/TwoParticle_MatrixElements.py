import numpy as np 
import matplotlib.pyplot as plt 

import sys 
sys.path.append( "/Users/markbook/Cobra" ) 

import jackknife as jk 

from iminuit import Minuit  

import OOP_TEST_v4 as model 

LEN  = 128.0 
UNIT = 2.0 * np.pi / LEN  
TCAP = 12 

########
########
###
### FOR ONE-PARTICLE 
###

## Create an object representing the NLSM 
field = model.NLSM()

## Create an object to contain the one-particle properties of the NLSM 
one_particle_2pt = model.one_2pt( field ) 

NOF_STATES = 7  ## Number of one-particle modes 

## Paths to one-particle correlation data
folder = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/2pt/1P/" 
paths  = [ folder+'re_one_corr_p'+str( n )+'_TAVG256.bn' for n in range( NOF_STATES ) ]  

## Read correlation data and assign the as attributes of the one-particle object 
one_particle_2pt.read_corrs( paths ) 

## "Accepted" fit-ranges based on Chi-square minimization 
fit_ranges = [ ( 13, 34 ),   ## Lp/(2pi) = 0 
               ( 16, 34 ),   ## Lp/(2pi) = 1 
               ( 13, 32 ),   ## Lp/(2pi) = 2 
               ( 11, 31 ),   ## Lp/(2pi) = 3 
               ( 15, 31 ),   ## Lp/(2pi) = 4 
               ( 12, 26 ),   ## Lp/(2pi) = 5 
               ( 10, 20 )    ## Lp/(2pi) = 6 
             ][:NOF_STATES] 

## "Accpeted" guess values for energies and modulus-squared overlaps 
guess_E = [ 0.073295, 0.08835, 0.1223, 0.1645, 0.2092, 0.2556, 0.3037 ] 
guess_A = [   1182.5,   985.6,  707.5,  530.5,  530.5,  413.7,  340.5 ]  

GUESS = np.array( [ guess_E, guess_A ] ).T 

## Fit energies of the one-particle modes 
one_particle_2pt.fit_spectrum( fit_ranges, GUESS, CHI2dof=True ) 

## Declare "MASS" variable using the correlation data 
MASS = np.mean( field.MASS )
M2   = MASS * MASS 

###
### FOR ONE-PARTICLE 
###
########
########

########
########
###
### FOR TWO-PARTICLE MATRIX ELEMENTS
###

## Create an object to contain the two-particle matrix-elements of the NLSM 
two_particle_3pt = model.two_3pt( field, TCAP=TCAP ) 

#### TWO-PARTICLE INTERPOLATORS CORRELATION ANALYSIS (BEGIN) #### 

## Paths to two-particle interpolators correlation data 
folder  = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/2pt/2P/Interp/" 
re_name = "re_2P_corr_Interps_E{0}_N{1}_TMAX42_TAVG256_NEW.bn" 
## E and N indicate energy level and boost, respectively 

## "Accepted" fit-ranges based on Chi-square minimization 
## for "ti"  
##            N0  N1  N2  N3 
ti_list = [ [  9, 11,  9, 11 ],   ## E0 
            [  7,  9,  9,  9 ],   ## E1 
            [  6,  9,  8,  8 ] ]  ## E2
## for "tf"
##            N0  N1  N2  N3      
tf_list = [ [ 37, 38, 35, 35 ],   ## E0 
            [ 30, 37, 32, 33 ],   ## E1 
            [ 20, 26, 20, 26 ] ]  ## E2 

## "Accpeted" guess values for energies and modulus-squared overlaps 
##                 N0                      N1                        N2                        N3 
pars_GUESS = [ [ [ 648072.24, 0.1736806 ], [ 778252.41, 0.1608590 ], [ 563528.40, 0.1941359 ], [ 464465.90, 0.2104521 ] ],   ## E0
               [ [ 339258.48, 0.2397667 ], [ 469465.36, 0.2067230 ], [ 352124.91, 0.2480103 ], [ 423998.50, 0.2350172 ] ],   ## E1 
               [ [ 187555.31, 0.3217390 ], [ 252390.70, 0.2808026 ], [ 197232.69, 0.3246418 ], [ 275889.88, 0.2919575 ] ] ]  ## E2 

## Build a list of energy-momentum tuples (E,N) as the data is fitted 
list_of_EP = []
for STATE_aux in range( 2 ): 
    for BOOST_aux in range( 4 ): 
        if STATE_aux < 2: ## Consider ground and 1st excited states only (for now)
            list_of_EP.append( (STATE_aux,BOOST_aux) )
            ti, tf = ti_list[STATE_aux][BOOST_aux], tf_list[STATE_aux][BOOST_aux] 
            re_path = folder+re_name.format( STATE_aux, BOOST_aux )
            two_particle_3pt.fit_C2pt_interps( STATE_aux, BOOST_aux, re_path, ti, tf, GUESS=pars_GUESS[STATE_aux][BOOST_aux] )  


## P.S. It would suffice to fit only the states (i.e. 2) of interest  

#### TWO-PARTICLE INTERPOLATORS CORRELATION ANALYSIS (END) #### 

#### TWO-PARTICLE 3-POINT CORRELATION ANALYSIS (BEGIN) #### 

## The current version opts for analysing a single matrix element 
## as indicated by a tuple indicating the energy-momentum of the final 
## and initial states ((Ef,Nf),(Ei,Ni)) 

OUT, IN = 0, 1
label_ME = ( ( 1, 3 ), ( 1, 0 ) )  

label_OUT, label_IN = label_ME[OUT], label_ME[IN] 

## Choice of fitting ranges based on minimum Chi2 per d.o.f. value 
## this excludes the cases when ti==-3 or tf==3 

fit_ranges = [ 
[ (-9,9), (-4,4), (-8,7), (-7,6), (-4,8), (-9,4), (-4,4), (-4,4) ], 
[ (-5,4), (-7,5), (-4,5), (-5,9), (-9,5), (-4,4), (-7,9), (-9,9) ], 
[ (-4,8), (-9,5), (-6,9), (-9,4), (-4,8), (-5,8), (-8,9), (-9,8) ], 
[ (-6,6), (-6,5), (-7,5), (-9,4), (-4,8), (-4,4), (-7,4), (-9,7) ], 
[ (-6,7), (-7,4), (-5,4), (-7,5), (-4,5), (-4,6), (-5,4), (-4,4) ], 
[ (-5,8), (-4,4), (-8,8), (-5,4), (-5,9), (-4,4), (-4,7), (-8,4) ], 
[ (-4,8), (-6,4), (-9,6), (-5,8), (-4,4), (-4,5), (-9,9), (-4,4) ], 
[ (-9,9), (-7,6), (-7,4), (-4,4), (-9,6), (-9,6), (-5,4), (-5,5) ] 
 ] 

folder = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/3pt/2P/I1/TMAX"+str(TCAP)+"/" 

## Loop over all possible matrix elements 
for STATE_OUT, BOOST_OUT in list_of_EP: 
    for STATE_IN, BOOST_IN in list_of_EP: 
        label = ( ( STATE_OUT, BOOST_OUT ), ( STATE_IN, BOOST_IN ) ) 

        ## For now, consider only the case given by "label_ME" 
        if label != label_ME: continue 

        ## Read files with correlation data 
        re_name = "re_2J2_corr_OUT_E{0}_N{1}_IN_E{2}_N{3}_TMAX{4}_TAVG256.bn".format( STATE_OUT, BOOST_OUT, STATE_IN, BOOST_IN, TCAP ) 
        im_name = "im_2J2_corr_OUT_E{0}_N{1}_IN_E{2}_N{3}_TMAX{4}_TAVG256.bn".format( STATE_OUT, BOOST_OUT, STATE_IN, BOOST_IN, TCAP ) 
        two_particle_3pt.read_corrs( label[OUT], label[IN], folder+re_name, folder+im_name )

        ## Assign correlation data to the 3-point object 
        index_ME = two_particle_3pt.C3pt_labels.index( label ) 
        two_particle_3pt.compute_ratio_from_pars(      label ) 
        two_particle_3pt.compute_3pt_phase(            label ) 

#print( two_particle_3pt.C3pt_labels ) 

#### TWO-PARTICLE 3-POINT CORRELATION ANALYSIS (END)   #### 

#### TWO-PARTICLE 3-POINT CORRELATION FIT (BEGIN) #### 

index_ME  = two_particle_3pt.C3pt_labels.index( label_ME  ) 
index_OUT = two_particle_3pt.C2pt_labels.index( label_OUT ) 
index_IN  = two_particle_3pt.C2pt_labels.index( label_IN  ) 

E_OUT_jk = two_particle_3pt.C2pt_pars[ index_OUT ][1] 
E_IN_jk  = two_particle_3pt.C2pt_pars[ index_IN  ][1] 

Q2_jk = np.square( (label_OUT[1]-label_IN[1])*UNIT ) - np.square( E_OUT_jk - E_IN_jk )

print( index_ME, index_OUT, index_IN )
print()

#quit()

re_ratio_jk = np.array( two_particle_3pt.re_ratios[ index_ME ] )
im_ratio_jk = two_particle_3pt.im_ratios[ index_ME ] 

def ratio_hypth( C, t ): 
    return C * np.ones( len( t ) ) 

print( np.shape( re_ratio_jk ) ) 

ti_range = np.arange( -TCAP+1+2, -2, 1 ) 
tf_range = np.arange(  3,  TCAP-1-1, 1 ) 

re_GUESS = 1.00 
im_GUESS = 0.05 

print( "tf=", tf_range )

re_ratios = [] 
im_ratios = [] 
re_ratios_errs = [] 
im_ratios_errs = [] 
re_chi2dof = [] 
im_chi2dof = [] 



for ti in [fit_ranges[ index_OUT ][ index_IN ][0]]: 
    re_ratio_fit_aux = [] 
    im_ratio_fit_aux = [] 
    re_ratio_err_aux = [] 
    im_ratio_err_aux = [] 
    re_chi2_dof_aux  = [] 
    im_chi2_dof_aux  = [] 
    for tf in [fit_ranges[ index_OUT ][ index_IN ][1]]: 
        ni, nf = ti+TCAP-1, tf+TCAP-1 
        t_fit  = np.arange( ti, tf+1, 1 ) 

        dof = len( t_fit )-1

        re_cov_matrix = jk.jk_cov_matrix( re_ratio_jk[ni:nf+1] ) 
        InvCov        = np.matrix( np.linalg.inv( re_cov_matrix ) )  

        re_ratio_fit_jk = [] 
        re_chi2_dof_jk  = [] 

        for jk_copy in re_ratio_jk.T: 

            data = jk_copy[ ni : nf+1 ]

            def Chi2( C ): 
                fit   = ratio_hypth( C, t_fit ) 
                diffs = np.matrix( data - fit ).T 
                return ( diffs.T @ InvCov @ diffs )[0,0] 
        
            Chi2_fit = Minuit( Chi2, C=re_GUESS ) 
            Chi2_fit.errordef = Minuit.LEAST_SQUARES 
            Chi2_fit.strategy = 2 
            Chi2_fit.errors["C"] = 0.0002 
            Chi2_fit.migrad() 

            re_ratio_fit_jk.append( Chi2_fit.values[0] ) 
            re_chi2_dof_jk.append( Chi2_fit.fval / dof )

        re_ratio_fit_aux.append( np.mean( re_ratio_fit_jk ) ) 
        re_chi2_dof_aux.append( np.mean( re_chi2_dof_jk ) )  

        re_ratio_err_aux.append( jk.jk_std( np.array( re_ratio_fit_jk ) ) ) 

        if label_ME[0][1] == 0 and label_ME[1][1] == 0: 
            pass
        else:

            im_cov_matrix = jk.jk_cov_matrix( im_ratio_jk[ni:nf+1] ) 
            InvCov        = np.matrix( np.linalg.inv( im_cov_matrix ) ) 
    
            im_ratio_fit_jk = [] 
            im_chi2_dof_jk  = [] 
    
            for jk_copy in im_ratio_jk.T:
                
                data = jk_copy[ ni : nf+1 ] 
    
                def Chi2( C ): 
                    fit   = ratio_hypth( C, t_fit ) 
                    diffs = np.matrix( data - fit ).T 
                    return ( diffs.T @ InvCov @ diffs )[0,0] 
                
                Chi2_fit = Minuit( Chi2, C=im_GUESS ) 
                Chi2_fit.errordef = Minuit.LEAST_SQUARES 
                Chi2_fit.strategy = 2 
                Chi2_fit.errors["C"] = 0.0002 
                Chi2_fit.migrad() 
    
                im_ratio_fit_jk.append( Chi2_fit.values[0] ) 
                im_chi2_dof_jk.append( Chi2_fit.fval / dof ) 
    
            im_ratio_fit_aux.append( np.mean( im_ratio_fit_jk ) ) 
            im_ratio_err_aux.append( jk.jk_std( np.array(im_ratio_fit_jk) ) ) 
            im_chi2_dof_aux.append( np.mean( im_chi2_dof_jk ) ) 

        if ( ti, tf ) == fit_ranges[ index_OUT ][ index_IN ]: 

            print() 
            print( label_OUT, label_IN, ( ti, tf ) ) 
            print() 

            save_folder = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/2_Analysis/3pt/2P/TMAX"+str(TCAP)+"/"

            file = open( save_folder+"Ratio_OUT_E{0}P{1}_IN_E{2}P{3}_TMAX{4}.jk".format( label_OUT[0], label_OUT[1], label_IN[0], label_IN[1], TCAP ), "w" )  

            if label_ME[0][1] == 0 and label_ME[1][1] == 0: 
                midstep = np.array( [ Q2_jk, re_ratio_fit_jk, np.zeros(len(re_ratio_fit_jk)), E_OUT_jk, E_IN_jk ] ).T 
            else:
                midstep = np.array( [ Q2_jk, re_ratio_fit_jk, im_ratio_fit_jk, E_OUT_jk, E_IN_jk ] ).T 

            for line in midstep:

                file.write( "{0:.16f} {1:.16f} {2:.16f} {3:.16f} {4} {5:.16f} {6}\n".format( 
                            line[0], line[1], line[2], line[3], label_OUT[1], line[4], label_IN[1] ) ) 
                ##          Q2       re_ratio im_ratio E_OUT    d_OUT         E_IN     d_IN 

            file.close() 

            print()

            quit()


    #aux_str = [ "{0:.2f}".format( x ) for x in re_chi2_dof_aux ] 
    aux_str = [ "{0:.2f}".format( x ) for x in im_chi2_dof_aux ] 
    #aux_str = [ "{0:.2f}".format( x ) for x in re_ratio_fit_aux ] 
    #aux_str = [ "{0:.2f}".format( x ) for x in re_ratio_fit_aux ] 

    re_chi2dof.append( re_chi2_dof_aux )
    re_ratios.append( re_ratio_fit_aux )
    re_ratios_errs.append( re_ratio_err_aux )

    if label_ME[0][1] == 0 and label_ME[1][1] == 0 : 
        pass
    else:
        im_chi2dof.append( im_chi2_dof_aux ) 
        im_ratios.append( im_ratio_fit_aux ) 
        im_ratios_errs.append( im_ratio_err_aux ) 

    print( aux_str, "ti=", ti ) 

print()

#quit()

folder = "/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/2_Analysis/3pt/2P/TMAX"+str(TCAP)+"/" 

file = open( folder+"re_fit_all_ranges_E"+str(label_IN[0])+"P"+str(label_IN[1])+"_to_E"+str(label_OUT[0])+"P"+str(label_OUT[1])+"_TMAX"+str(TCAP)+".dat", "w" ) 

file.write( "---Chi Square per d.o.f---\n\n" )

file.write( "tf= \n" )
for tf in tf_range: 
    file.write( "{0} ".format(tf) )
file.write( "\n" )

counter = -1 
for line in re_chi2dof: 
    counter += 1
    for value in line: 
        file.write( "{0:.2f} ".format( value ) ) 
    file.write( " ti="+str(ti_range[counter])+"\n" ) 

file.write( "\n\n---Ratio mean (real part)---\n\n" ) 

file.write( "tf= \n" )
for tf in tf_range: 
    file.write( "{0} ".format(tf) )
file.write( "\n" )

counter = -1 
for line in re_ratios: 
    counter += 1
    for value in line: 
        file.write( "{0:.4f} ".format( value ) ) 
    file.write( " ti="+str(ti_range[counter])+"\n" ) 

file.write( "\n\n---Ratio err (real part)---\n\n" ) 

file.write( "tf= \n" )
for tf in tf_range: 
    file.write( "{0} ".format(tf) )
file.write( "\n" )

counter = -1 
for line in re_ratios_errs: 
    counter += 1
    for value in line: 
        file.write( "{0:.4f} ".format( value ) ) 
    file.write( " ti="+str(ti_range[counter])+"\n" ) 

file.close()


###
if label_ME[0][1] == 0 and label_ME[1][1] == 0:
    pass 
else:

    file = open( folder+"im_fit_all_ranges_E"+str(label_IN[0])+"P"+str(label_IN[1])+"_to_E"+str(label_OUT[0])+"P"+str(label_OUT[1])+"_TMAX"+str(TCAP)+".dat", "w" ) 
    
    file.write( "---Chi Square per d.o.f---\n\n" )
    
    file.write( "tf= \n" )
    for tf in tf_range: 
        file.write( "{0} ".format(tf) )
    file.write( "\n" )
    
    counter = -1 
    for line in im_chi2dof: 
        counter += 1
        for value in line: 
            file.write( "{0:.2f} ".format( value ) ) 
        file.write( " ti="+str(ti_range[counter])+"\n" ) 
    
    file.write( "\n\n---Ratio mean (imaginary part)---\n\n" ) 
    
    file.write( "tf= \n" )
    for tf in tf_range: 
        file.write( "{0} ".format(tf) )
    file.write( "\n" )
    
    counter = -1 
    for line in im_ratios: 
        counter += 1
        for value in line: 
            file.write( "{0:.4f} ".format( value ) ) 
        file.write( " ti="+str(ti_range[counter])+"\n" ) 
    
    file.write( "\n\n---Ratio err (imaginary part)---\n\n" ) 
    
    file.write( "tf= \n" )
    for tf in tf_range: 
        file.write( "{0} ".format(tf) )
    file.write( "\n" )
    
    counter = -1 
    for line in im_ratios_errs: 
        counter += 1
        for value in line: 
            file.write( "{0:.4f} ".format( value ) ) 
        file.write( " ti="+str(ti_range[counter])+"\n" ) 
    
    file.close()
    
#### TWO-PARTICLE 3-POINT CORRELATION FIT (END)   #### 
    
###
### FOR TWO-PARTICLE MATRIX ELEMENTS
###
########
########



