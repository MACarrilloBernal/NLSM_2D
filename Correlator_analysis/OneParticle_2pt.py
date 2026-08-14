import numpy as np 

import sys
sys.path.append("/Users/markbook/Cobra")
import jackknife as jk

import OOP_TEST_v4 as OOP 

STATE = 3 

folder = '/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/1_Correlations/wahab/2pt/1P/'

paths = [ folder+'re_one_corr_p'+str( n )+'_TAVG256.bn' for n in range( STATE+1 ) ]

field = OOP.NLSM() 

one_particle_2pt = OOP.one_2pt( field ) 

one_particle_2pt.read_corrs( paths ) 

''' CONSERVATIVE '''
fit_ranges = [ ( 13, 25 ), 
               ( 16, 27 ), 
               ( 15, 21 ), 
               ( 11, 26 ), 
               ( 12, 27 ), 
               ( 12, 24 ),
               ( 12, 21 )
             ]

''' BOLD '''
#fit_ranges = [ ( 13, 34 ), 
#               ( 16, 34 ), 
#               ( 13, 32 ), 
#               ( 11, 31 ), 
#               ( 15, 29 ), 
#               ( 12, 33 ),
#               ( 10, 31 )
#             ]

guess_E = [ 0.073295, 0.08827, 0.1223, 0.1645, 0.2092, 0.2559, 0.3037 ]#[STATE] 
guess_A = [   1182.5,   983.9,  707.5,  530.5,  412.9,  341.9,  289.8 ]#[STATE] 

CHECK_CHI2dof = 1

if CHECK_CHI2dof :
    ti_range = [ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 ]
    #ti_range = [ 13, 14, 15, 16, 17, 18 ] 
    #ti_range = [ 16, 17, 18, 19, 20, 21, 22 ] 
    #tf_range = [ 21, 22, 23, 24, 25, 26, 27 ]
    tf_range = [ 27, 28, 29, 30, 31, 32, 33 ] 
    #tf_range = [ 33, 34, 35, 36, 37, 38, 39 ] 
    #tf_range = [ 39, 40, 41, 42, 43, 44, 45 ] 

    tf_range = [ 24, 25, 26, 27, 28, 29, 30 ]

    guess_E = [ 0.073295, 0.08827, 0.1223, 0.1645, 0.2092, 0.2556, 0.3037 ][STATE] 
    guess_A = [   1182.5,   983.9,  707.5,  530.5,  412.9,  413.7,  340.5 ][STATE] 
    
    print( tf_range ) 
    
    all_Es = [] 
    all_As = [] 
    all_Cs = [] 
    
    for ti in ti_range: 
    
        E_aux = [] 
        A_aux = [] 
        C_aux = []
    
        for tf in tf_range: 
    
            aux = one_particle_2pt.make_fit( ti, tf, STATE, GUESS=[guess_E,guess_A], CHI2dof=True ) 
    
            E_aux.append( np.mean( aux[0] ) ) 
            A_aux.append( np.mean( aux[1] ) ) 
            C_aux.append( np.mean( aux[2] ) ) 
    
        all_Es.append( E_aux ) 
        all_As.append( A_aux )
        all_Cs.append( C_aux ) 
    
        print( C_aux, "ti:", ti )
    
    print()
    print( "E:" )
    for k, _ in enumerate( all_Es ): 
        print( _, "ti:", ti_range[k] )
    print()
    print( "|Z|2:" )
    for k, _ in enumerate( all_As ): 
        print( _, "ti:", ti_range[k] )

    quit()
    
        
one_particle_2pt.fit_spectrum( fit_ranges, GUESS=np.array( [ guess_E, guess_A ] ).T, CHI2dof=True ) 

for _ in one_particle_2pt.spetrum : 
    print( np.mean( _ ), "+/-", jk.jk_std( _ ) )

for _ in one_particle_2pt.overlaps : 
    print( np.mean( _ ), "+/-", jk.jk_std( _ ) )

for _ in one_particle_2pt.Chi2norms : 
    print( np.mean( _ ), "+/-", jk.jk_std( _ ) )

#one_particle_2pt.save_data( len(one_particle_2pt.spetrum), 
#                            CHI2dof=True, folder="/Users/markbook/Desktop/LOCAL/NLSM_TESTS/L128/2_Analysis/2pt/1P/" )



